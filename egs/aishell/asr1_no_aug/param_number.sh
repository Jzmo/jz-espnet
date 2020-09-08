#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=5        # start from 0 if you need to start from data preparation
stop_stage=5
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/tuning/train_pytorch_conformer.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=             # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -u
set -o pipefail

train_set=train
train_dev=dev
recog_set="dev test"


for train_config in `ls conf/tuning/train_pytorch*.yaml`; do
#for train_config in conf/tuning/train_pytorch_pformer_keepone_cnn_1234.yaml; do

    if [ -z ${tag} ]; then
	expname=${train_set}_${backend}_$(basename ${train_config%.*})
	if ${do_delta}; then
            expname=${expname}_delta
	fi
    else
	expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
	if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
	       [[ $(get_yaml.py ${train_config} model-module) = *former* ]]; then
	    # Average ASR model
	    if ${use_valbest_average}; then
		recog_model=model.val${n_average}.avg.best
		opt="--log ${expdir}/results/log"
	    else
		recog_model=model.last${n_average}.avg.best
		opt="--log"
	    fi
	    if [ -d $expdir ]; then
		echo $train_config
		count_checkpoints.py \
		    ${opt} \
		    --backend ${backend} \
		    --snapshots ${expdir}/results/snapshot.ep.* \
		    --out ${expdir}/results/${recog_model} \
		    --num ${n_average}
	    fi
	fi
    fi

done
