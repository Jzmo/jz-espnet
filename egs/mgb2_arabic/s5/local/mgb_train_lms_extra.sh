#!/usr/bin/env bash

# Copyright (C) 2020, Tianzi Wang
# To be run from one directory above this script.

text=$1
lmtext_extra=$2

if [ $# -ne 2 ]; then
  echo "Usage: $0 <train-text> <lm-text-extra>"
  exit 1
fi

dir=data/lang_char/lm_large_mer$mer/${train_set}_units.txt
mkdir -p $dir

[ ! -f $text ] && echo "$0: No such file $lmtext" && exit 1;
[ ! -f $lmtext_extra ] && echo "$0: No such file $lmtext_extra" && exit 1;






echo dict creation succeeded
