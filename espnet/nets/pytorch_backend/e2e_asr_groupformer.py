# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conformer speech recognition model (pytorch).

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

from distutils.util import strtobool

from espnet.nets.pytorch_backend.groupformer.encoder import Encoder
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E as E2EConformer


class E2E(E2EConformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2EConformer.add_arguments(parser)
        E2E.add_groupformer_arguments(parser)
        return parser

    @staticmethod
    def add_groupformer_arguments(parser):
        """Add arguments for former model."""
        group = parser.add_argument_group("xformer model specific setting")
        group.add_argument(
            "--groupconv-encoder-layer-type",
            type=str,
            default=None,
            choices=["grouplightconv", "flgc1d"],
            help="groupformer encoder group convolution layer type",
        )
        group.add_argument(
            "--groupconv-wshare",
            default=4,
            type=int,
            help="tied weight of group convolultion",
        )
        group.add_argument(
            "--groupconv-dim",
            default=256,
            type=int,
            help="number of kernel of group convolultion",
        )
        group.add_argument(
            "--groupconv-dropout-rate",
            default=0.1,
            type=float,
            help="dropout rate of group convolultion",
        )
        group.add_argument(
            "--groupconv-encoder-kernel-length",
            default=31,
            type=int,
            help="kernel size of encoder light convolultion",
        )
        group.add_argument(
            "--use-se-layer",
            default=False,
            type=bool,
            help="whether to add SE layer after convolultion layer",
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
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            macaron_style=args.macaron_style,
            groupconv_layer_type=args.groupconv_encoder_layer_type,
            groupconv_wshare=args.groupconv_wshare,
            groupconv_dim=args.groupconv_dim,
            groupconv_dropout_rate=args.groupconv_dropout_rate,
            groupconv_kernel_length=args.groupconv_encoder_kernel_length,
            use_se_layer=args.use_se_layer,
        )
        self.reset_parameters(args)
