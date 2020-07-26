#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=7         # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
decode_nj=32         # The number of parallel jobs in decoding.
gpu_decode=false     # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.

# Data preparation related
local_data_opts= # Options to be passed to local/data.sh.

# Feature extraction related
feats_type=raw              # Feature type (fbank or stft or raw).
audio_format=flac           # Audio format (only in feats_type=raw).
min_wav_duration=0.1        # Minimum duration in second
max_wav_duration=20         # Maximum duration in second
write_collected_feats=false # Whether to dump features in stats collection.
# Only used for feats_type != raw
fs=16000          # Sampling rate.
fmin=80           # Minimum frequency of Mel basis.
fmax=7600         # Maximum frequency of Mel basis.
n_mels=80         # The number of mel basis.
n_fft=1024        # The number of fft points.
n_shift=256       # The number of shift points.
win_length=null   # Window length.
f0min=80          # Maximum f0 (only for FastSpeech2)
f0max=800         # Minimum f0 (only for FastSpeech2)

oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole

# Training related
train_config=      # Config for training.
train_args=        # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
tts_exp=           # Specify the direcotry path for experiment. If this option is specified, tag is ignored.
num_splits=1       # Number of splitting for tts corpus
teacher_dumpdir="" # Directory of teacher outputs (needed if tts=fastspeech).

# Decoding related
decode_config= # Config for decoding.
decode_args=   # Arguments for decoding, e.g., "--threshold 0.75".
               # Note that it will overwrite args in decode config.
decode_tag=""  # Suffix for decoding directory.
decode_model=train.loss.best.pth # Model path for decoding e.g.,
                                 # decode_model=train.loss.best.pth
                                 # decode_model=3epoch.pth
                                 # decode_model=valid.acc.best.pth
                                 # decode_model=valid.loss.ave.pth
griffin_lim_iters=4 # the number of iterations of Griffin-Lim.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
srctexts=        # Texts to create token list. Multiple items can be specified.
nlsyms_txt=none  # Non-linguistic symbol list (needed if existing).
token_type=phn   # Transcription type.
cleaner=tacotron # Text cleaner.
g2p=g2p_en       # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus
text_fold_length=150   # fold_length for text data
speech_fold_length=800 # fold_length for speech data

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>" --srctexts "<srctexts>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes
    --nj             # The number of parallel jobs (default="${nj}").
    --decode_nj      # The number of parallel jobs in decoding (default="${decode_nj}").
    --gpu_decode     # Whether to perform gpu decoding (default="${gpu_decode}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type     # Feature type (fbank or stft or raw, default="${feats_type}").
    --audio_format   # Audio format (only in feats_type=raw, default="${audio_format}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --fs             # Sampling rate (default="${fs}").
    --fmax           # Maximum frequency of Mel basis (default="${fmax}").
    --fmin           # Minimum frequency of Mel basis (default="${fmin}").
    --n_mels         # The number of mel basis (default="${n_mels}").
    --n_fft          # The number of fft points (default="${n_fft}").
    --n_shift        # The number of shift points (default="${n_shift}").
    --win_length     # Window length (default="${win_length}").
    --oov            # Out of vocabrary symbol (default="${oov}").
    --blank          # CTC blank symbol (default="${blank}").
    --sos_eos=       # sos and eos symbole (default="${sos_eos}").

    # Training related
    --train_config # Config for training (default="${train_config}").
    --train_args   # Arguments for training, e.g., "--max_epoch 1" (default="${train_args}").
                   # Note that it will overwrite args in train config.
    --tag          # Suffix for training directory (default="${tag}").
    --tts_exp      # Specify the direcotry path for experiment. If this option is specified, tag is ignored (default="${tts_exp}").
    --num_splits   # Number of splitting for tts corpus (default="${num_splits}").

    # Decoding related
    --decode_config     # Config for decoding (default="${decode_config}").
    --decode_args       # Arguments for decoding, e.g., "--threshold 0.75" (default="${decode_args}").
                        # Note that it will overwrite args in decode config.
    --decode_tag        # Suffix for decoding directory (default="${decode_tag}").
    --decode_model      # Model path for decoding (default=${decode_model}).
    --griffin_lim_iters # The number of iterations of Griffin-Lim (default=${griffin_lim_iters}).

    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set         # Name of training set (required).
    --valid_set         # Name of validation set used for monitoring/tuning network training (required).
    --test_sets         # Names of test sets (required).
                        # Note that multiple items (e.g., both dev and eval sets) can be specified.
    --srctexts          # Texts to create token list (required).
                        # Note that multiple items can be specified.
    --nlsyms_txt        # Non-linguistic symbol list (default="${nlsyms_txt}").
    --token_type        # Transcription type (default="${token_type}").
    --cleaner           # Text cleaner (default="${cleaner}").
    --g2p               # g2p method (default="${g2p}").
    --lang              # The language type of corpus (default=${lang}).
    --text_fold_length   # fold_length for text data
    --speech_fold_length # fold_length for speech data
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Check feature type
if [ "${feats_type}" = fbank ]; then
    data_feats="${dumpdir}/fbank"
elif [ "${feats_type}" = stft ]; then
    data_feats="${dumpdir}/stft"
elif [ "${feats_type}" = raw ]; then
    data_feats="${dumpdir}/raw"
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Check token list type
token_listdir="data/token_list/${token_type}"
if [ "${cleaner}" != none ]; then
    token_listdir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    token_listdir+="_${g2p}"
fi
token_list="${token_listdir}/tokens.txt"

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)_${feats_type}_${token_type}"
    else
        tag="train_${feats_type}_${token_type}"
    fi
    if [ "${cleaner}" != none ]; then
        tag+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        tag+="_${g2p}"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${decode_tag}" ]; then
    if [ -n "${decode_config}" ]; then
        decode_tag="$(basename "${decode_config}" .yaml)"
    else
        decode_tag=decode
    fi
    # Add overwritten arg's info
    if [ -n "${decode_args}" ]; then
        decode_tag+="$(echo "${decode_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    decode_tag+="_$(echo "${decode_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for collect-stats mode
tts_stats_dir="${expdir}/tts_stats_${feats_type}_${token_type}"
if [ "${cleaner}" != none ]; then
    tts_stats_dir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    tts_stats_dir+="_${g2p}"
fi
# The directory used for training commands
if [ -z "${tts_exp}" ]; then
    tts_exp="${expdir}/tts_${tag}"
fi


# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi


    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # TODO(kamo): Change kaldi-ark to npy or HDF5?
        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        if [ "${feats_type}" = raw ]; then
            log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    _opts+="--segments data/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank ] || [ "${feats_type}" = stft ] ; then
            log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}/"

            # Generate the fbank features; by default 80-dimensional fbanks on each frame
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1. Copy datadir
                utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"

                # 2. Feature extract
                # TODO(kamo): Wrap (nj->_nj) in make_fbank.sh
                _nj=$(min "${nj}" "$(<${data_feats}${_suf}/${dset}/utt2spk wc -l)")
                _opts=
                if [ "${feats_type}" = fbank ] ; then
                    _opts+="--fmax ${fmax} "
                    _opts+="--fmin ${fmin} "
                    _opts+="--n_mels ${n_mels} "
                fi

                # shellcheck disable=SC2086
                scripts/feats/make_"${feats_type}".sh --cmd "${train_cmd}" --nj "${_nj}" \
                    --fs "${fs}" \
                    --n_fft "${n_fft}" \
                    --n_shift "${n_shift}" \
                    --win_length "${win_length}" \
                    ${_opts} \
                    "${data_feats}${_suf}/${dset}"
                utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

                # 3. Derive the the frame length and feature dimension
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                # 4. Write feats_dim
                head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' \
                    | cut -d, -f2 > ${data_feats}${_suf}/${dset}/feats_dim

                # 5. Write feats_type
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done
        fi
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

                # utt2num_samples is created by format_wav_scp.sh
                <"${data_feats}/org/${dset}/utt2num_samples" \
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                        >"${data_feats}/${dset}/utt2num_samples"
                <"${data_feats}/org/${dset}/wav.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/wav.scp"
            else
                # Get frame shift in ms from conf/fbank.conf
                _frame_shift=
                if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                    # Assume using conf/fbank.conf for feature extraction
                    _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
                fi
                if [ -z "${_frame_shift}" ]; then
                    # If not existing, use the default number in Kaldi (=10ms).
                    # If you are using different number, you have to change the following value manually.
                    _frame_shift=10
                fi

                _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

                cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
                <"${data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                    | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                        >"${data_feats}/${dset}/feats_shape"
                <"${data_feats}/org/${dset}/feats.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/feats_shape"  \
                    >"${data_feats}/${dset}/feats.scp"
            fi

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"
        done

        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Generate token_list from ${srctexts}"
        # "nlsyms_txt" should be generated by local/data.sh if need

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

        python3 -m espnet2.bin.tokenize_text \
              --token_type "${token_type}" -f 2- \
              --input "${data_feats}/srctexts" --output "${token_list}" \
              --non_linguistic_symbols "${nlsyms_txt}" \
              --cleaner "${cleaner}" \
              --g2p "${g2p}" \
              --write_vocabulary true \
              --add_symbol "${blank}:0" \
              --add_symbol "${oov}:1" \
              --add_symbol "${sos_eos}:-1"
    fi
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================



if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: TTS collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _feats_type="$(<${_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            _type=sound
            _opts+="--feats_extract fbank "
            _opts+="--feats_extract_conf fs=${fs} "
            _opts+="--feats_extract_conf n_fft=${n_fft} "
            _opts+="--feats_extract_conf fmin=${fmin} "
            _opts+="--feats_extract_conf fmax=${fmax} "
            _opts+="--feats_extract_conf n_mels=${n_mels} "
            _opts+="--feats_extract_conf hop_length=${n_shift} "
            _opts+="--feats_extract_conf win_length=${win_length} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _odim="$(<${_train_dir}/feats_dim)"
            _opts+="--odim=${_odim} "
        fi

        # Add extra configs for FastSpeech2
        # NOTE(kan-bayashi): We always pass this options but not used in default
        _opts+="--pitch_extract_conf fs=${fs} "
        _opts+="--pitch_extract_conf n_fft=${n_fft} "
        _opts+="--pitch_extract_conf hop_length=${n_shift} "
        _opts+="--pitch_extract_conf f0max=${f0max} "
        _opts+="--pitch_extract_conf f0min=${f0min} "
        _opts+="--energy_extract_conf fs=${fs} "
        _opts+="--energy_extract_conf n_fft=${n_fft} "
        _opts+="--energy_extract_conf hop_length=${n_shift} "
        _opts+="--energy_extract_conf win_length=${win_length} "

        if [ -n "${teacher_dumpdir}" ]; then
            _teacher_train_dir="${teacher_dumpdir}/${train_set}"
            _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/durations,durations,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/durations,durations,text_int "
        fi

        # 1. Split the key file
        _logdir="${tts_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_train_dir}/${_scp} wc -l)" "$(<${_valid_dir}/${_scp} wc -l)")

        key_file="${_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "TTS collect_stats started... log: '${_logdir}/stats.*.log'"
        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            python3 -m espnet2.bin.tts_train \
                --collect_stats true \
                --write_collected_feats "${write_collected_feats}" \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize none \
                --pitch_normalize none \
                --energy_normalize none \
                --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
                --train_data_path_and_name_and_type "${_train_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_valid_dir}/text,text,text" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${_scp},speech,${_type}" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${train_args}

        # 3. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        python3 -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${tts_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${tts_stats_dir}/train/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${tts_stats_dir}/train/text_shape.${token_type}"

        <"${tts_stats_dir}/valid/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${tts_stats_dir}/valid/text_shape.${token_type}"
    fi


    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: TTS Training: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        if [ -z "${teacher_dumpdir}" ]; then
            # CASE 1: Standard training
            _feats_type="$(<${_train_dir}/feats_type)"

            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                # "sound" supports "wav", "flac", etc.
                _type=sound
                _fold_length="$((speech_fold_length * n_shift))"
                _opts+="--feats_extract fbank "
                _opts+="--feats_extract_conf fs=${fs} "
                _opts+="--feats_extract_conf fmin=${fmin} "
                _opts+="--feats_extract_conf fmax=${fmax} "
                _opts+="--feats_extract_conf n_mels=${n_mels} "
                _opts+="--feats_extract_conf hop_length=${n_shift} "
                _opts+="--feats_extract_conf n_fft=${n_fft} "
                _opts+="--feats_extract_conf win_length=${win_length} "
            else
                _scp=feats.scp
                _type=kaldi_ark
                _fold_length="${speech_fold_length}"
                _odim="$(<${_train_dir}/feats_dim)"
                _opts+="--odim=${_odim} "
            fi

            if [ "${num_splits}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${tts_stats_dir}/splits${num_splits}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    python3 -m espnet2.bin.split_scps \
                      --scps \
                          "${_train_dir}/text" \
                          "${_train_dir}/${_scp}" \
                          "${tts_stats_dir}/train/speech_shape" \
                          "${tts_stats_dir}/train/text_shape.${token_type}" \
                      --num_splits "${num_splits}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
                _opts+="--train_shape_file ${_split_dir}/speech_shape "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},speech,${_type} "
                _opts+="--train_shape_file ${tts_stats_dir}/train/text_shape.${token_type} "
                _opts+="--train_shape_file ${tts_stats_dir}/train/speech_shape "
            fi
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},speech,${_type} "
            _opts+="--valid_shape_file ${tts_stats_dir}/valid/text_shape.${token_type} "
            _opts+="--valid_shape_file ${tts_stats_dir}/valid/speech_shape "
        else
            # CASE 2: Knowledge distillation training
            _teacher_train_dir="${teacher_dumpdir}/${train_set}"
            _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
            _fold_length="${speech_fold_length}"

            if [ "${num_splits}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${teacher_dumpdir}/splits${num_splits}"
                _scps=""
                if [ -e ${_teacher_train_dir}/probs ]; then
                    # Knowledge distillation case
                    _scp=feats.scp
                    _type=npy
                    _scps+="${_teacher_train_dir}/denorm/${_scp} "
                    _scps+="${_teacher_train_dir}/speech_shape "
                    _odim="$(head -n 1 "${_teacher_train_dir}/speech_shape" | cut -f 2 -d ",")"
                    _opts+="--odim=${_odim} "
                else
                    # Teacher forcing case
                    if [ "${feats_type}" = raw ]; then
                        _scp=wav.scp
                        _type=sound
                        _fold_length="$((speech_fold_length * n_shift))"
                        _opts+="--feats_extract fbank "
                        _opts+="--feats_extract_conf fs=${fs} "
                        _opts+="--feats_extract_conf fmin=${fmin} "
                        _opts+="--feats_extract_conf fmax=${fmax} "
                        _opts+="--feats_extract_conf n_mels=${n_mels} "
                        _opts+="--feats_extract_conf hop_length=${n_shift} "
                        _opts+="--feats_extract_conf n_fft=${n_fft} "
                        _opts+="--feats_extract_conf win_length=${win_length} "
                    else
                        _scp=feats.scp
                        _type=kaldi_ark
                        _odim="$(head -n 1 "${tts_stats_dir}/train/speech_shape" | cut -f 2 -d ",")"
                        _opts+="--odim=${_odim} "
                    fi
                    _scps+="${_train_dir}/${_scp} "
                    _scps+="${tts_stats_dir}/speech_shape "
                fi
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    python3 -m espnet2.bin.split_scps \
                      --scps \
                          "${_train_dir}/text" \
                          "${_teacher_train_dir}/durations" \
                          "${_teacher_train_dir}/focus_rates" \
                          "${tts_stats_dir}/text_shape.${token_type}" \
                          ${_scps} \
                      --num_splits "${num_splits}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
                _opts+="--train_shape_file ${_split_dir}/speech_shape "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
                _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/durations,durations,text_int "
                _opts+="--train_shape_file ${tts_stats_dir}/train/text_shape.${token_type} "
                if [ -e ${_teacher_train_dir}/probs ]; then
                    # Knowledge distillation case
                    _scp=feats.scp
                    _type=npy
                    _odim="$(head -n 1 "${_teacher_train_dir}/speech_shape" | cut -f 2 -d ",")"
                    _opts+="--odim=${_odim} "
                    _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/denorm/${_scp},speech,${_type} "
                    _opts+="--train_shape_file ${_teacher_train_dir}/speech_shape "
                else
                    # Teacher forcing case
                    if [ "${feats_type}" = raw ]; then
                        _scp=wav.scp
                        _type=sound
                        _fold_length="$((speech_fold_length * n_shift))"
                        _opts+="--feats_extract fbank "
                        _opts+="--feats_extract_conf fs=${fs} "
                        _opts+="--feats_extract_conf fmin=${fmin} "
                        _opts+="--feats_extract_conf fmax=${fmax} "
                        _opts+="--feats_extract_conf n_mels=${n_mels} "
                        _opts+="--feats_extract_conf hop_length=${n_shift} "
                        _opts+="--feats_extract_conf n_fft=${n_fft} "
                        _opts+="--feats_extract_conf win_length=${win_length} "
                    else
                        _scp=feats.scp
                        _type=kaldi_ark
                        _odim="$(head -n 1 "${tts_stats_dir}/train/speech_shape" | cut -f 2 -d ",")"
                        _opts+="--odim=${_odim} "
                    fi
                    _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},speech,${_type} "
                    _opts+="--train_shape_file ${tts_stats_dir}/train/speech_shape "
                fi
            fi
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
            _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/durations,durations,text_int "
            _opts+="--valid_shape_file ${tts_stats_dir}/valid/text_shape.${token_type} "
            if [ -e ${_teacher_train_dir}/probs ]; then
                # Knowledge distillation case
                _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/denorm/${_scp},speech,${_type} "
                _opts+="--valid_shape_file ${_teacher_valid_dir}/speech_shape "
            else
                # Teacher forcing case
                _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},speech,${_type} "
                _opts+="--valid_shape_file ${tts_stats_dir}/valid/speech_shape "
            fi
        fi

        # Check extra inputs (For FastSpeech2)
        if [ -e "${tts_stats_dir}/train/collect_feats/pitch.scp" ]; then
            _scp=pitch.scp
            _type=npy
            _train_collect_dir=${tts_stats_dir}/train/collect_feats
            _valid_collect_dir=${tts_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},pitch,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},pitch,${_type} "
        fi
        if [ -e "${tts_stats_dir}/train/collect_feats/energy.scp" ]; then
            _scp=energy.scp
            _type=npy
            _train_collect_dir=${tts_stats_dir}/train/collect_feats
            _valid_collect_dir=${tts_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},energy,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},energy,${_type} "
        fi

        # Check extra statistics (For FastSpeech2)
        if [ -e "${tts_stats_dir}/train/pitch_stats.npz" ]; then
            _opts+="--pitch_normalize_conf stats_file=${tts_stats_dir}/train/pitch_stats.npz "
        fi
        if [ -e "${tts_stats_dir}/train/energy_stats.npz" ]; then
            _opts+="--energy_normalize_conf stats_file=${tts_stats_dir}/train/energy_stats.npz "
        fi

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "TTS training started... log: '${tts_exp}/train.log'"
        # shellcheck disable=SC2086
        python3 -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${tts_exp}/train.log" \
            --log "${tts_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${tts_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            python3 -m espnet2.bin.tts_train \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize global_mvn \
                --normalize_conf "stats_file=${tts_stats_dir}/train/feats_stats.npz" \
                --resume true \
                --fold_length "${text_fold_length}" \
                --fold_length "${_fold_length}" \
                --output_dir "${tts_exp}" \
                ${_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Decoding: training_dir=${tts_exp}"

    if ${gpu_decode}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _opts=
    if [ -n "${decode_config}" ]; then
        _opts+="--config ${decode_config} "
    fi

    if [ -z "${teacher_dumpdir}" ]; then
        _feats_type="$(<${data_feats}/${train_set}/feats_type)"
    else
        # TODO(kan-bayashi): Fix hard coding
        if [ -e "${teacher_dumpdir}/${train_set}/probs" ]; then
            _feats_type=fbank
        else
            _feats_type="$(<${data_feats}/${train_set}/feats_type)"
        fi
    fi

    # NOTE(kamo): If feats_type=raw, vocoder_conf is unnecessary
    _scp=wav.scp
    _type=sound
    if [ "${_feats_type}" = fbank ] || [ "${_feats_type}" = stft ]; then
        _opts+="--vocoder_conf n_fft=${n_fft} "
        _opts+="--vocoder_conf n_shift=${n_shift} "
        _opts+="--vocoder_conf win_length=${win_length} "
        _opts+="--vocoder_conf fs=${fs} "
        _scp=feats.scp
        _type=kaldi_ark
    fi
    if [ "${_feats_type}" = fbank ]; then
        _opts+="--vocoder_conf n_mels=${n_mels} "
        _opts+="--vocoder_conf fmin=${fmin} "
        _opts+="--vocoder_conf fmax=${fmax} "
    fi

    for dset in ${test_sets}; do
        _data="${data_feats}/${dset}"
        _speech_data="${_data}"
        _dir="${tts_exp}/${decode_tag}/${dset}"
        _logdir="${_dir}/log"
        mkdir -p "${_logdir}"

        # NOTE(kan-bayashi): Overwrite speech arguments if teacher dumpdir is provided
        if [ -n "${teacher_dumpdir}" ]; then
            # TODO(kan-bayashi): Make this part more flexible
            if [ -e "${teacher_dumpdir}/${train_set}/probs" ]; then
                _speech_data="${teacher_dumpdir}/${dset}/denorm"
                _scp=feats.scp
                _type=npy
            fi
        fi

        # 0. Copy feats_type
        cp "${_data}/feats_type" "${_dir}/feats_type"

        # 1. Split the key file
        key_file=${_data}/text
        split_scps=""
        _nj=$(min "${decode_nj}" "$(<${key_file} wc -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/tts_inference.*.log'"
        # shellcheck disable=SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/tts_inference.JOB.log \
            python3 -m espnet2.bin.tts_inference \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/text,text,text" \
                --data_path_and_name_and_type ${_speech_data}/${_scp},speech,${_type} \
                --key_file "${_logdir}"/keys.JOB.scp \
                --model_file "${tts_exp}"/"${decode_model}" \
                --train_config "${tts_exp}"/config.yaml \
                --output_dir "${_logdir}"/output.JOB \
                --vocoder_conf griffin_lim_iters="${griffin_lim_iters}" \
                ${_opts} ${decode_args}

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"/{norm,denorm,wav}
        for i in $(seq "${_nj}"); do
             cat "${_logdir}/output.${i}/norm/feats.scp"
        done | LC_ALL=C sort -k1 > "${_dir}/norm/feats.scp"
        for i in $(seq "${_nj}"); do
             cat "${_logdir}/output.${i}/denorm/feats.scp"
        done | LC_ALL=C sort -k1 > "${_dir}/denorm/feats.scp"
        for i in $(seq "${_nj}"); do
             cat "${_logdir}/output.${i}/speech_shape/speech_shape"
        done | LC_ALL=C sort -k1 > "${_dir}/speech_shape"
        for i in $(seq "${_nj}"); do
            mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
            rm -rf "${_logdir}/output.${i}"/wav
        done
        if [ -e "${_logdir}/output.${_nj}/att_ws" ]; then
            mkdir -p "${_dir}"/att_ws
            for i in $(seq "${_nj}"); do
                 cat "${_logdir}/output.${i}/durations/durations"
            done | LC_ALL=C sort -k1 > "${_dir}/durations"
            for i in $(seq "${_nj}"); do
                 cat "${_logdir}/output.${i}/focus_rates/focus_rates"
            done | LC_ALL=C sort -k1 > "${_dir}/focus_rates"
            for i in $(seq "${_nj}"); do
                mv -u "${_logdir}/output.${i}"/att_ws/*.png "${_dir}"/att_ws
                rm -rf "${_logdir}/output.${i}"/att_ws
            done
        fi
        if [ -e "${_logdir}/output.${_nj}/probs" ]; then
            mkdir -p "${_dir}"/probs
            for i in $(seq "${_nj}"); do
                mv -u "${_logdir}/output.${i}"/probs/*.png "${_dir}"/probs
                rm -rf "${_logdir}/output.${i}"/probs
            done
        fi
    done
fi

packed_model="${tts_exp}/${tts_exp##*/}_${decode_model%.*}.zip"
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "[Option] Stage 8: Pack model: ${packed_model}"

    python -m espnet2.bin.pack tts \
        --dirname "$(basename ${packed_model} .zip)" \
        --config.yaml "${tts_exp}"/config.yaml \
        --pretrain.pth "${tts_exp}"/"${decode_model}" \
        --option ${tts_stats_dir}/train/feats_stats.npz  \
        --outpath "${packed_model}"

    # NOTE(kamo): If you'll use packed model to decode in this script, do as follows
    #   % unzip ${packed_model}
    #   % ./run.sh --stage 8 --tts_exp $(basename ${packed_model} .zip) --decode_model pretrain.pth
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    log "[Option] Stage 9: Upload model to Zenodo: ${packed_model}"

    # To upload your model, you need to do:
    #   1. Signup to Zenodo: https://zenodo.org/
    #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
    #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="
git checkout $(git show -s --format=%H)"
    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/tts1/ -> foo/tt1
    _task="$(pwd | rev | cut -d/ -f1-2 | rev)"
    # foo/asr1 -> foo
    _corpus="${_task%/*}"

    # Generate description file
    cat << EOF > "${tts_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">Coming soon...</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
# Download the model file here
unzip $(basename ${packed_model})
./run.sh --skip_data_prep false --skip_train true --asr_exp $(basename ${packed_model} .zip)/asr --decode_asr_model pretrain.pth --lm_exp $(basename ${packed_model} .zip)/lm --decode_lm pretrain.pth</code>
</pre></li>
<li><strong>Config</strong><pre><code>$(cat "${tts_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

    # NOTE(kamo): The model file is uploaded here, but not published yet.
    #   Please confirm your record at Zenodo and publish by youself.

    # shellcheck disable=SC2086
    python -m espnet2.bin.zenodo_upload \
        --file "${packed_model}" \
        --title "ESPnet2 pretrained model, ${_creator_name}/${_corpus}_$(basename ${packed_model} .zip), fs=${fs}, lang=${lang}" \
        --description_file "${tts_exp}"/description \
        --creator_name "${_creator_name}" \
        --license "CC-BY-4.0" \
        --use_sandbox false \
        --publish false
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
