#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000
train_set="train_960"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <mfcc-dir> <n-cluster>"
    echo "e.g.: $0 dump/raw/org/train_960_sp 50"
    exit 1
fi

mfcc=$1
n_cluster=$2

data_dir=${LIBRISPEECH}/LibriSpeech/
train_set=train_960_sp

# generate tsv files
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    _nj=4
    _tsv_dir=exp/km/${train_set}/tsv
    _logdir=exp/km/${train_set}/log
    _ext=flac
    mkdir -p ${_logdir}
    for part in train dev; do
	${train_cmd} JOB=1:"${_nj}" "${_logdir}"/tsv.JOB.log \
                     ${python} -m local/manifest.py \
                     --wav-file ${_data_dir} \
		     --dest ${_tsv_dir} \
		     --ext ${_ext} \
		     --path-must-contain ${part} \
		     --valid-percent 0
    done
    if [ ! -e ${_tsv_dir}/dict.ltr.txt ]; then
    	echo "Dowloading dict.lrt.txt"
	wget "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt" -P ${_tsv_dir}
    fi
fi

# k-means clustering
# mfcc or hubert
# if mfcc: check if mfcc exist, if not generate
# if humbert: check if hubert feature exit, if not, check if hubert model exist
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: "
    _nj=4
    _km_dir=exp/km/${train_set}/results
    _logdir=exp/km/${train_set}/log
    _mfcc_dir=dump/${train_set}
    if [ ! -e ${_mfcc_dir} ]; then
	echo "mfcc feature does not exist"
	exit 1
    fi
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/km.JOB.log \
                 ${python} -m local/learn_kmeans.py \
                 --feat-dir ${_mfcc_dir} \
		 --km-path ${_km_dir} \
		 --n-cluster ${n_cluster} 

fi


# k-means application
# mfcc or hubert
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: "
    _nj=4
    _km_dir=exp/km/${train_set}/results
    _logdir=exp/km/${train_set}/log
    if [ ! -e ${_mfcc_dir} ]; then
	echo "mfcc feature does not exist"
	exit 1
    fi
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/km.JOB.log \
                 ${python} -m local/dump_km_label.py \
                 --feat-dir dump/${part}/ \
		 --label-dir ${_km_dir}/plabel

fi
