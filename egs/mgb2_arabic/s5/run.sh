#!/bin/bash

# Copyright 2020 Johns Hopkins University (Tianzi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# preference on how to process xml file [python, xml]
process_xml="python"

# general configuration
backend=pytorch
stage=4
stop_stage=4
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train_mtlalpha0.5.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode_ctcweight1.0.yaml

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000    # effective only for word LMs
lmtag=              # tag for managing LMs
lm_resume=          # specify a snapshot file to resume LM training

# decoding parameter
recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
datadir=~/data
datadir=/export/corpora5
mgb2_root=${datadir}/MGB-2
db_dir=data/DB

mer=80

nj=100  # split training into how many jobs?
nDecodeJobs=80

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_mer${mer}"
train_dev="dev"
lm_test="test"
recog_set="dev_non_overlap dev_overlap test_non_overlap test_overlap"

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Untar and preparing training data"
    mkdir -p $db_dir
    local/mgb_extract_data.sh $mgb2_root $db_dir
    local/mgb_data_prep.sh $db_dir $mer $process_xml
    local/mgb_data_prep_test.sh $db_dir
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${train_dev} ${recog_set}; do
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 8 --write_utt2num_frames true \
				  data/${x} exp/make_fbank/${x} ${fbankdir}
	utils/fix_data_dir.sh data/${x}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
	    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
	    data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${lm_test} ${recog_set}; do
	feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
	dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
		data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
		${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
text=data/train_mer${mer}/text
lmtext_extra=DB/train/lm_text/lm_text_clean_bw
echo "dictionary: ${dict}"
if [ $stage -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cleantext=data/train_mer${mer}/text_large
    cut -d ' ' -f 2- $text | cat - $lmtext_extra > $cleantext || exit 1;
    text2token.py -s 1 -n 1 ${cleantext} | cut -f 2- -d" " | tr " " "\n" \
	| sort | uniq | grep -v -e "^\s*$" | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    
    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
		 data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
		 data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${lm_test} ${recog_set}; do
	feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
	data2json.sh --feat ${feat_recog_dir}/feats.scp \
		     data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
	lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    if [ ${use_wordlm} = true ]; then
	lmdatadir=data/local/wordlm_train
	lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
	mkdir -p ${lmdatadir}

	cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train.txt 
	cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
	cut -f 2- -d" " data/${lm_test}/text > ${lmdatadir}/test.txt

	cat ${lmdatadir}/train.txt ${lmtext_extra} > ${lmdatadir}/text.no_oov
	text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/text.no_oov
    else
	lmdatadir=data/local/lm_train
	lmdict=${dict}
	mkdir -p ${lmdatadir}
	lmdatadir=data/local/lm_train
	lmdict=${dict}
	mkdir -p ${lmdatadir}
	text2token.py -s 1 -n 1 data/${train_set}/text \
	    | cut -f 2- -d" " > ${lmdatadir}/train.txt
	text2token.py -s 1 -n 1 data/${train_dev}/text \
	    | cut -f 2- -d" " > ${lmdatadir}/valid.txt
	text2token.py -s 1 -n 1 data/${lm_test}/text \
	    | cut -f 2- -d" " > ${lmdatadir}/test.txt
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
		lm_train.py \
		--config ${lm_config} \
		--ngpu ${ngpu} \
		--backend ${backend} \
		--verbose 1 \
		--outdir ${lmexpdir} \
		--tensorboard-dir tensorboard/${lmexpname} \
		--train-label ${lmdatadir}/train.txt \
		--valid-label ${lmdatadir}/valid.txt \
		--resume ${lm_resume} \
		--dict ${lmdict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
	expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
		asr_train.py \
		--config ${train_config} \
		--ngpu ${ngpu} \
		--backend ${backend} \
		--outdir ${expdir}/results \
		--tensorboard-dir tensorboard/${expname} \
		--debugmode ${debugmode} \
		--dict ${dict} \
		--debugdir ${expdir} \
		--minibatches ${N} \
		--verbose ${verbose} \
		--resume ${resume} \
		--train-json ${feat_tr_dir}/data.json \
		--valid-json ${feat_dt_dir}/data.json
fi
