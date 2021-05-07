# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Mask CTC based non-autoregressive speech recognition model (pytorch).

See https://arxiv.org/abs/2005.08700 for the detail.

"""
import pdb
from itertools import groupby
import logging
import math

from distutils.util import strtobool
import numpy
import torch

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform
from espnet.nets.pytorch_backend.maskctc.mask import square_mask
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy


class E2E(E2ETransformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2ETransformer.add_arguments(parser)
        E2E.add_maskctc_arguments(parser)

        return parser

    @staticmethod
    def add_maskctc_arguments(parser):
        """Add arguments for maskctc model."""
        group = parser.add_argument_group("maskctc specific setting")

        group.add_argument(
            "--maskctc-use-conformer-encoder",
            default=False,
            type=strtobool,
        )
        group = add_arguments_conformer_common(group)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        odim += 1  # for the mask token

        super().__init__(idim, odim, args, ignore_id)
        assert 0.0 <= self.mtlalpha < 1.0, "mtlalpha should be [0.0, 1.0)"

        self.mask_token = odim - 1
        self.sos = odim - 2
        self.eos = odim - 2
        self.odim = odim

        if args.maskctc_use_conformer_encoder:
            if args.transformer_attn_dropout_rate is None:
                args.transformer_attn_dropout_rate = args.conformer_dropout_rate
            self.encoder = Encoder(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
                selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
                activation_type=args.transformer_encoder_activation_type,
                macaron_style=args.macaron_style,
                use_cnn_module=args.use_cnn_module,
                cnn_module_kernel=args.cnn_module_kernel,
            )
        self.reset_parameters(args)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # 2. forward decoder
        ys_in_pad, ys_out_pad = mask_uniform(
            ys_pad, self.mask_token, self.eos, self.ignore_id
        )
        ys_mask = square_mask(ys_in_pad, self.eos)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        logging.info("ys_in_pad:{}".format(ys_in_pad[0:3]))
        logging.info("pred_pad:{}".format(pred_pad[0:3].argmax(dim=-1)))
        self.pred_pad = pred_pad

        # 3. compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)

        self.acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )
        # 4. compute ctc loss
        loss_ctc, cer_ctc = None, None
        if self.mtlalpha > 0:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: decoding result
        :rtype: list
        """

        def num2str(char_list, mask_token, mask_char="_"):
            def f(yl):
                cl = [char_list[y] if y != mask_token else mask_char for y in yl]
                return "".join(cl).replace("<space>", " ")

            return f

        n2s = num2str(char_list, self.mask_token)

        self.eval()
        # h = self.encode(x).unsqueeze(0)
        # block_len = recog_args.xl_decode_block_length
        block_len = 32
        # chuck_len = recog_args.xl_decode_chuck_length
        chuck_len = 4
        subsample = 4
        cache_len = 1
        decode_mode = "online"
        logging.info("self.odim is {}".format(self.odim))
        if decode_mode == "online":
            x = torch.as_tensor(x)
            x = x[: x.size(0) // subsample * subsample, :]
            hs_pad = torch.zeros(
                x.size(0) // subsample, self.adim, dtype=x.dtype
            ).unsqueeze(0)
            logging.info(
                "length length//4 {} {}".format(x.size(0), x.size(0) // subsample)
            )

            hyp_new = [self.sos]
            h_pad = hs_pad.clone()
            t0 = 0
            ret = []
            for t in range(x.size(0)):
                # with streaming input, get the hidden output of encoder
                if (t + 1) % (block_len * chuck_len * subsample) == 0 or (
                    t == x.size(0) - 1 and t0 // subsample < t // subsample
                ):
                    logging.info(
                        "chunking feat {} {} {}".format(
                            t0, t + 1, x[t0 : t + 1, :].size()
                        )
                    )
                    if t0 > block_len * subsample:
                        # after first block
                        h_pad[:, t0 // subsample : t // subsample, :] = self.encoder(
                            x[t0 - block_len * subsample : t + 1, :].unsqueeze(0), None
                        )[0][:, block_len:, :]
                        # raise ValueError
                    else:
                        h_pad[:, t0 // subsample : t // subsample, :] = self.encoder(
                            x[t0 : t + 1, :].unsqueeze(0), None
                        )[0]
                    # greedy ctc output
                    logging.info(
                        "intervel: {}, {}".format(t0 // subsample, t // subsample)
                    )
                    # ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h_pad[:, :t//subsample, :])).max(dim=-1)
                    ctc_probs, ctc_ids = torch.exp(
                        self.ctc.log_softmax(
                            h_pad[:, t0 // subsample : t // subsample, :]
                        )
                    ).max(dim=-1)

                    y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
                    y_idx = torch.nonzero(y_hat != 0).squeeze(-1)
                    # calculate token-level ctc probability by taking
                    # the maximum probablity of consecutive frame with
                    # the same ctc symbols
                    probs_hat = []
                    cnt = 0
                    for i, y in enumerate(y_hat.tolist()):
                        probs_hat.append(-1)
                        while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                            if probs_hat[i] < ctc_probs[0][cnt]:
                                probs_hat[i] = ctc_probs[0][cnt].item()
                            cnt += 1
                    probs_hat = torch.from_numpy(numpy.array(probs_hat))

                    # mask ctc output based on ctc probablities
                    p_thres = recog_args.maskctc_probability_threshold
                    mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
                    confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(
                        -1
                    )
                    mask_num = len(mask_idx)

                    y_in = (
                        torch.zeros(1, len(y_idx), dtype=torch.long) + self.mask_token
                    )
                    y_in[0][confident_idx] = y_hat[y_idx][confident_idx]
                    # iterative decoding
                    if not mask_num == 0 and recog_args.maskctc_n_iterations > 0:

                        K = recog_args.maskctc_n_iterations
                        num_iter = K if mask_num >= K and K > 0 else mask_num
                        if t0 > block_len * subsample and cache_len > 0:
                            y_in_concat = torch.cat([y_in_prev, y_in], dim=-1)
                            h_pad_concat = torch.cat(
                                [
                                    h_pad_prev,
                                    h_pad[:, t0 // subsample : t // subsample, :],
                                ],
                                dim=1,
                            )
                            mask_idx = mask_idx + y_in_prev.size(1)
                        else:
                            y_in_concat = y_in.clone()
                            h_pad_concat = h_pad[
                                :, t0 // subsample : t // subsample, :
                            ].clone()

                        for n_iter in range(num_iter - 1):
                            pred, _ = self.decoder(
                                y_in_concat, None, h_pad_concat, None
                            )
                            pred_score, pred_id = pred[0][mask_idx].max(dim=-1)
                            cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                            y_in_concat[0][mask_idx[cand]] = pred_id[cand]
                            mask_idx = torch.nonzero(
                                y_in_concat[0] == self.mask_token
                            ).squeeze(-1)

                            logging.info("msk:{}".format(n2s(y_in_concat[0].tolist())))

                        # [edict lewoftover masks (|masks| < mask_num // num_iter)
                        pred, pred_mask = self.decoder(
                            y_in_concat, None, h_pad_concat, None
                        )
                        y_in_concat[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)
                        ret = (
                            ret[: len(ret) - y_in_prev.size(1)]
                            + y_in_concat.tolist()[0]
                        )

                        y_in = y_in_concat[:, -y_in.size(1) :]
                        logging.info("msk:{}".format(n2s(y_in_concat[0].tolist())))

                    else:
                        y_in[0] = y_hat[y_idx]
                        ret += y_in.tolist()[0]
                        logging.info("ctc:{}".format(n2s(y_in[0].tolist())))

                    y_in_prev = y_in.clone()
                    h_pad_prev = h_pad[:, t0 // subsample : t // subsample, :].clone()

                    t0 = t + 1
                if t == x.size(0) - 1:
                    break
        else:
            h = self.encode(x).unsqueeze(0)

            # greedy ctc outputs
            ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h)).max(dim=-1)
            y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
            y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

            # calculate token-level ctc probabilities by taking
            # the maximum probability of consecutive frames with
            # the same ctc symbols
            probs_hat = []
            cnt = 0
            for i, y in enumerate(y_hat.tolist()):
                probs_hat.append(-1)
                while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                    if probs_hat[i] < ctc_probs[0][cnt]:
                        probs_hat[i] = ctc_probs[0][cnt].item()
                    cnt += 1
            probs_hat = torch.from_numpy(numpy.array(probs_hat))

            # mask ctc outputs based on ctc probabilities
            p_thres = recog_args.maskctc_probability_threshold
            mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
            confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
            mask_num = len(mask_idx)

            y_in = torch.zeros(1, len(y_idx), dtype=torch.long) + self.mask_token
            y_in[0][confident_idx] = y_hat[y_idx][confident_idx]

            logging.info("ctc:{}".format(n2s(y_in[0].tolist())))

            # iterative decoding
            if not mask_num == 0 and recog_args.maskctc_n_iterations > 0:
                K = recog_args.maskctc_n_iterations
                num_iter = K if mask_num >= K and K > 0 else mask_num

                for t in range(num_iter - 1):

                    pred, _ = self.decoder(y_in, None, h, None)
                    pred_score, pred_id = pred[0][mask_idx].max(dim=-1)
                    cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                    y_in[0][mask_idx[cand]] = pred_id[cand]
                    mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)

                    logging.info("msk:{}".format(n2s(y_in[0].tolist())))
                    logging.info(
                        "mask_idx, pred:{}, {}".format(
                            mask_idx, pred[0][mask_idx].argmax(dim=-1)
                        )
                    )

                # predict leftover masks (|masks| < mask_num // num_iter)
                pred, pred_mask = self.decoder(y_in, None, h, None)
                logging.info(
                    "mask_idx, pred:{}, {}".format(
                        mask_idx, pred[0][mask_idx].argmax(dim=-1)
                    )
                )
                y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)

                logging.info("msk:{}".format(n2s(y_in[0].tolist())))
            else:
                y_in[0] = y_hat[y_idx]

            ret = y_in.tolist()[0]
        logging.info("ret:{}".format(ret))
        hyp = {"score": 0.0, "yseq": [self.sos] + ret + [self.eos]}

        return [hyp]
