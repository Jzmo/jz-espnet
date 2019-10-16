#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*"  # Print the command line for logging
. ./path.sh

nj=1
do_delta=false
eval_tts_model=1
db_root=""
voc="GL"
backend=pytorch
api=v1
help_message="Usage: $0 <asr_model> <outdir> <subset>"

. utils/parse_options.sh

asr_model=$1
outdir=$2
name=$3

if [ $# != 3 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

echo "stage 6.0: Model preparation"
# ASR model selection for CER objective evaluation 
# Please add new model if you want to use your ASR model.
asr_model_dir="exp/${asr_model}_asr"
case "${asr_model}" in
    "librispeech.transformer.ngpu4") asr_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" \
      asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
      asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer_large.yaml" \
      recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best" \
      lang_model="${asr_model_dir}/exp/irielm.ep11.last5.avg/rnnlm.model.best" ;;

    *) echo "No such models: ${asr_model}"; exit 1 ;;
esac

# ASR model download (librispeech)
if [ ! -e ${asr_model_dir}/.complete ]; then
    mkdir -p ${asr_model_dir}
    download_from_google_drive.sh ${asr_url} ${asr_model_dir} ".tar.gz"
    touch ${asr_model_dir}/.complete
fi
echo "ASR model: ${asr_model_dir} exits."

# Select TTS model
if [ ${eval_tts_model} == 1 ]; then
    echo "Evaluate: TTS model"
else
    echo "Evaluate: ground truth"
    expdir=exp/ground_truth
    outdir=${expdir}/sym_link
    
    if [ ${db_root} == "" ]; then
        echo "Please set --db_root"; exit 1
    fi
    mkdir -p ${outdir}_denorm/${name}/wav
    cat < data/${name}/wav.scp | awk '{print $1}' | while read -r filename; do
        if [ -L ${outdir}_denorm/${name}/wav/${filename}.wav ]; then
            unlink ${outdir}_denorm/${name}/wav/${filename}.wav
        fi
        ln -s ${db_root}/LJSpeech-1.1/wavs/${filename}.wav ${outdir}_denorm/${name}/wav/${filename}.wav
    done
fi

# Select decoder
if [ ${voc} == "GL" ]; then
    dir_tail=""
elif [ ${voc} == "WNV_softmax" ]; then
    dir_tail="_wnv_nsf"
elif [ ${voc} == "WNV_mol" ]; then
    dir_tail="_wnv_mol"
fi

# setting dir
asr_data_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.data${dir_tail}"
asr_fbank_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.fbank${dir_tail}"
asr_feat_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.dump${dir_tail}"
asr_result_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.result${dir_tail}"


echo "stage 6.1: Data preparation for ASR"
# Data preparation for ASR
local/ob_eval/data_prep_for_asr.sh ${outdir}_denorm/${name}/wav${dir_tail} ${asr_data_dir}/${name}
cp data/${name}/text ${asr_data_dir}/${name}/text
utils/validate_data_dir.sh --no-feats ${asr_data_dir}/${name}


echo "stage 6.2: Feature extraction for ASR"
# Feature extraction for ASR
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
  --write_utt2num_frames true \
  --write_utt2dur false \
  ${asr_data_dir}/${name} \
  ${outdir}_denorm.ob_eval/${asr_model}_asr.make_fbank${dir_tail}/${name} \
  ${asr_fbank_dir}/${name}

utils/fix_data_dir.sh ${asr_data_dir}/${name}

dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
  ${asr_data_dir}/${name}/feats.scp ${asr_cmvn} ${outdir}_denorm.ob_eval/${asr_model}_asr.dump_feats${dir_tail}/${name} \
  ${asr_feat_dir}/${name}


echo "stage 6.3: Dictionary and Json Data Preparation for ASR"
# Dictionary and Json Data Preparation for ASR
asr_dict="data/decode_dict/X.txt"; mkdir -p ${asr_dict%/*}
echo "<unk> 1" > ${asr_dict}

data2json.sh --feat ${asr_feat_dir}/${name}/feats.scp \
  ${asr_data_dir}/${name} ${asr_dict} > ${asr_feat_dir}/${name}/data.json


echo "stage 6.4: ASR decoding"
# ASR decoding
asr_decode_config="conf/tuning/decode_asr.yaml"
cat < ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}

# split data
splitjson.py --parts ${nj} ${asr_feat_dir}/${name}/data.json

# set batchsize 0 to disable batch decoding    
${decode_cmd} JOB=1:${nj} ${asr_result_dir}.${api}/${name}/log/decode.JOB.log \
    asr_recog.py \
      --config ${asr_decode_config} \
      --ngpu 0 \
      --backend ${backend} \
      --batchsize 0 \
      --recog-json ${asr_feat_dir}/${name}/split${nj}utt/data.JOB.json \
      --result-label ${asr_result_dir}.${api}/${name}/data.JOB.json \
      --model ${recog_model} \
      --api ${api} \
      --rnnlm ${lang_model}

# calculate CER
score_sclite_wo_dict.sh --wer true ${asr_result_dir}.${api}/${name}
