# Copyright 2020 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer common arguments."""


from distutils.util import strtobool


def add_arguments_ltransformer_common(group):
    """Add Transformer common arguments."""
    group.add_argument(
        "--localized-transformer-encoder-window-size",
        type=str,
        default="0",
        help="localized transformer encoder window size",
    )
    return group
