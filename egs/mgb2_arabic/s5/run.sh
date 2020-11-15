#!/bin/bash

# Copyright 2020 Johns Hopkins University (Tianzi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# preference on how to process xml file [python, xml]
process_xml="python"

# general configuration
backend=pytorch
stage=5
stop_stage=5
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train_pytorch_transformer.v1.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode_transformer.yaml

# rnnlm related
lmtag=              # tag for managing LMs
lm_resume=          # specify a snapshot file to resume LM training
# bpemode(unigram or bpe)
nbpe=500
bpemode=unigram

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10
# data
mgb2_root=/export/corpora5/MGB-2
db_dir=data/DB

#FILTER OUT SEGMENTS BASED ON MER (Match Error Rate)
mer=80

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_mer${mer}_tr90"
train_dev="train_mer${mer}_cv10"
lm_test="dev"
recog_set="dev_non_overlap dev_overlap"

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Untar and preparing training data"
    mkdir -p $db_dir
    local/mgb_extract_data.sh $mgb2_root $db_dir
    local/mgb_data_prep.sh $db_dir $mer $process_xml
    utils/subset_data_dir_tr_cv.sh data/train_mer${mer} data/${train_set} data/${train_dev} || exit 1;
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

    for x in ${train_set} ${train_dev}; do
	if [ -d data/${x}_ori ]; then
	    rm -r data/${x}_ori
	fi
	mv data/${x} data/${x}_ori
	remove_longshortdata.sh --maxchars 800 data/${x}_ori data/${x}
	utils/fix_data_dir.sh data/${x}
    done
    
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
	    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
	    data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
	feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
	dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
		data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
		${feat_recog_dir}
    done
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_unit.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
text=data/train_mer${mer}/text
lmtext_extra=data/DB/train/lm_text/lm_text_clean_bw
echo "dictionary: ${dict}"
if [ $stage -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cleantext=data/train_mer${mer}/text_large
    cut -d ' ' -f 2- $text | cat - $lmtext_extra > $cleantext || exit 1;
    spm_train --input=$cleantext --vocab_size=${nbpe} --model_type=${bpemode} \
	      --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < ${cleantext} | \
	tr ' ' '\n' | sort | uniq | awk -v count=1 '{if($0) {print $0 " " ++count}}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
		 data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
		 data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
	feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
	data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
		     data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    [ ! -e ${lmdatadir} ] && mkdir -p ${lmdatadir}

    cut -f 2- -d" " data/${train_set}/text | \
	spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt 
    cut -f 2- -d" " data/${train_dev}/text | \
	spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/valid.txt
        
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
		--dict ${dict}
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
		--ngpu ${ngpu} \
		--config ${train_config} \
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

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
	recog_model=model.last${n_average}.avg.best
	average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
	(
	    decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
	    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

	    # split data
	    splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

	    #### use CPU for decoding
	    ngpu=0
	    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
			  asr_recog.py \
			  --config ${decode_config} \
			  --ngpu ${ngpu} \
			  --backend ${backend} \
			  --debugmode ${debugmode} \
			  --verbose ${verbose} \
			  --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
			  --result-label ${expdir}/${decode_dir}/data.JOB.json \
			  --model ${expdir}/results/${recog_model}  \
			  --rnnlm ${lmexpdir}/rnnlm.model.best

	    score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

	) &
	pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
