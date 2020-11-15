# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conformer speech recognition model (pytorch).

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

from distutils.util import strtobool

from espnet.nets.pytorch_backend.dynamicformer.encoder import Encoder
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
        E2E.add_dynamicformer_arguments(parser)
        return parser

    @staticmethod
    def add_dynamicformer_arguments(parser):
        """Add arguments for former model."""
        group = parser.add_argument_group("xformer model specific setting")
        group.add_argument(
            "--lightconv-encoder-layer-type",
            type=str,
            default=None,
            choices=["lightconv", "lightconv2d", "dynamicconv", "dynamicconv2d"],
            help="xformer encoder light convolution layer type",
        )
        group.add_argument(
            "--lightconv-decoder-layer-type",
            type=str,
            default=None,
            choices=["lightconv", "lightconv2d", "dynamicconv", "dynamicconv2d"],
            help="xformer decoder light convolution layer type",
        )
        group.add_argument(
            "--lightconv-wshare",
            default=4,
            type=int,
            help="tied weight of light convolultion",
        )
        group.add_argument(
            "--lightconv-dim",
            default=256,
            type=int,
            help="number of kernel of light convolultion",
        )
        group.add_argument(
            "--lightconv-dropout-rate",
            default=0.1,
            type=float,
            help="dropout rate of light convolultion",
        )
        group.add_argument(
            "--lightconv-encoder-kernel-length",
            default="",
            type=str,
            help="kernel size of encoder light convolultion",
        )
        group.add_argument(
            "--lightconv-decoder-kernel-length",
            default="",
            type=str,
            help="kernel size of decoder light convolultion",
        )
        group.add_argument(
            "--lightconv-layer-number",
            default="all",
            type=str,
            help="layers that light convolultion, default to be applied to all layers",
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
            lightconv_layer_type=args.lightconv_encoder_layer_type,
            lightconv_wshare=args.lightconv_wshare,
            lightconv_dim=args.lightconv_dim,
            lightconv_dropout_rate=args.lightconv_dropout_rate,
            lightconv_kernel_length=args.lightconv_encoder_kernel_length,
            lightconv_layer_number_str=args.lightconv_layer_number,
            use_se_layer=args.use_se_layer,
        )
        self.reset_parameters(args)
