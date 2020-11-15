# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conformer speech recognition model (pytorch).

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

from distutils.util import strtobool

from espnet.nets.pytorch_backend.mcformer.encoder import Encoder
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer


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
        E2E.add_mcformer_arguments(parser)
        return parser

    @staticmethod
    def add_mcformer_arguments(parser):
        """Add arguments for former model."""
        group = parser.add_argument_group("mcformer model specific setting")
        group.add_argument(
            "--transformer-encoder-pos-enc-layer-type",
            type=str,
            default="abs_pos",
            choices=["abs_pos", "scaled_abs_pos", "rel_pos"],
            help="transformer encoder positional encoding layer type",
        )
        group.add_argument(
            "--macaron-style",
            default=False,
            type=strtobool,
            help="Whether to use macaron style for positionwise layer",
        )
        group.add_argument(
            "--mc-selfattention-layer-type",
            type=str,
            default=None,
            choices=["mcattn", "rel_mcattn"],
            help="mcformer encoder multi conv attention layer type",
        )
        group.add_argument(
            "--mc-n-kernel",
            type=int,
            default=1,
            help="mcformer encoder number of convlution kernel",
        )
        group.add_argument(
            "--mc-kernel-size",
            default="31",
            type=str,
            help="mcformer encoder convlution kernel",
        )
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(idim, odim, args, ignore_id)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
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
            macaron_style=args.macaron_style,
            mc_selfattention_layer_type=args.mc_selfattention_layer_type,
            mc_n_kernel=args.mc_n_kernel,
            mc_kernel_size_str=args.mc_kernel_size,
            padding_idx=-1,
        )
        self.reset_parameters(args)
