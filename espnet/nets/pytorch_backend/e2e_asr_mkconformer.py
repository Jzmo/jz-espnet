# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conformer speech recognition model (pytorch).

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

from distutils.util import strtobool

from espnet.nets.pytorch_backend.mkconformer.encoder import Encoder
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
        E2E.add_mkconformer_arguments(parser)
        return parser

    @staticmethod
    def add_mkconformer_arguments(parser):
        """Add arguments for mkconformer model."""
        group = parser.add_argument_group("mkconformer model specific setting")
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
        # CNN module
        group.add_argument(
            "--use-cnn-module",
            default=False,
            type=strtobool,
            help="Use convolution module or not",
        )
        group.add_argument(
            "--cnn-module-kernel",
            default=31,
            type=int,
            help="Kernel size of convolution module.",
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
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            use_se_layer=args.use_se_layer,
        )
        self.reset_parameters(args)
