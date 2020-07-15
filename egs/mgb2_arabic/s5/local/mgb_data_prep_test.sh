#!/usr/bin/env bash

# Copyright (C) 2016, Qatar Computing Research Institute, HBKU
#               2016-2019  Vimal Manohar
#               2019 Dongji Gao
#               2020 Tianzi Wang

if [ $# -ne 1 ]; then
    echo "Usage: $0 <DB-dir>"
    exit 1;
fi

db_dir=$1

test_dir=data/test

# backup
for x in $test_dir; do
    mkdir -p $x
    if [ -f ${x}/wav.scp ]; then
	mdkir -p ${x}/.backup
	mv ${x}/{wav.scp,feats.scp,utt2spk,spk2utt,segments,text} ${x}/.backup
    fi
done

set -e -o pipefail

for x in text segments; do
    awk '(NF)>1' $db_dir/test/${x}.all > $test_dir/${x}
done

find $db_dir/test/wav -type f -name "*.wav" | \
    awk -F/ '{print $NF}' | perl -pe 's/\.wav//g' > \
				 $test_dir/wav_list

for x in $(cat $test_dir/wav_list); do
    echo $x $db_dir/test/wav/$x.wav >> $test_dir/wav.scp
done

#Creating a file reco2file_and_channel which is used by convert_ctm.pl in local/score.sh script
awk '{print $1" "$1" A"}' $test_dir/wav.scp > $test_dir/reco2file_and_channel

# Creating utt2spk for test from segments
if [ ! -f $test_dir/utt2spk ]; then
    cut -d ' ' -f1 $test_dir/segments > $test_dir/utt_id
    cut -d '_' -f1-2 $test_dir/utt_id | paste -d ' ' $test_dir/utt_id - > $test_dir/utt2spk
fi

for list in overlap non_overlap; do
    rm -rf ${test_dir}_$list || true
    cp -r $test_dir ${test_dir}_$list
    for x in segments text utt2spk; do
	utils/filter_scp.pl $db_dir/test/${list}_speech.lst $test_dir/$x > ${test_dir}_$list/${x}
    done
done

for dir in $test_dir ${test_dir}_overlap ${test_dir}_non_overlap; do
    utils/fix_data_dir.sh $dir
    utils/validate_data_dir.sh --no-feats $dir
done

echo "Test data preparation succeeded"
