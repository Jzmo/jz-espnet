#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr.sh \
    --train_set train_nodev \
    --lm_config conf/train_lm.yaml \
    --valid_set train_dev \
    --test_sets "train_dev test" \
    --srctexts "data/train_nodev/text" "$@"
