# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conformer speech recognition model (pytorch).

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

from distutils.util import strtobool

from espnet.nets.pytorch_backend.newformer.encoder import Encoder
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
        E2E.add_newformer_arguments(parser)
        return parser

    @staticmethod
    def add_newformer_arguments(parser):
        """Add arguments for former model."""
        group = parser.add_argument_group("newformer model specific setting")
        group.add_argument(
            "--conv-encoder-type",
            type=str,
            default=None,
            choices=["stack", "parallel", "alter"],
            help="newformer encoder type",
        )
        group.add_argument(
            "--use-pos-enc-conv",
            type=strtobool,
            default=False,
            help="add position encoding to conv layers",
        )
        group.add_argument(
            "--conv-encoder-layer-type",
            type=str,
            default=None,
            choices=[
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "conformer_conv1d",
                "lightdepthwise",
            ],
            help="newformer encoder convolution layer type",
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
            "--conv-encoder-kernel-length",
            default="",
            type=str,
            help="kernel size of encoder convolultion",
        )
        group.add_argument(
            "--conv-block-number",
            default="all",
            type=str,
            help="block number with convolultion, default to be applied to all layers",
        )
        group.add_argument(
            "--use-se-layer",
            default=False,
            type=strtobool,
            help="whether to add SE layer after convolultion layer",
        )
        group.add_argument(
            "--se-block-number",
            default="",
            type=str,
            help="block number with se block, default not applied",
        )
        group.add_argument(
            "--shuffle-after",
            default=False,
            type=strtobool,
            help="whether to shuffle channels after each block",
        )
        group.add_argument(
            "--shuffle-block-number",
            default="",
            type=str,
            help="block number with shuffle channels, default not applied",
        )
        group.add_argument(
            "--dual-type",
            default="linear",
            type=str,
            help="combine method of MH-SA and LW-Conv",
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
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            conv_encoder_type=args.conv_encoder_type,
            use_pos_enc_conv=args.use_pos_enc_conv,
            conv_encoder_layer_type=args.conv_encoder_layer_type,
            lightconv_wshare=args.lightconv_wshare,
            lightconv_dim=args.lightconv_dim,
            lightconv_dropout_rate=args.lightconv_dropout_rate,
            conv_kernel_length_str=args.conv_encoder_kernel_length,
            conv_block_number_str=args.conv_block_number,
            use_se_layer=args.use_se_layer,
            se_block_number_str=args.se_block_number,
            shuffle_after=args.shuffle_after,
            shuffle_block_number_str=args.shuffle_block_number,
            dual_type=args.dual_type,
        )
        self.reset_parameters(args)
