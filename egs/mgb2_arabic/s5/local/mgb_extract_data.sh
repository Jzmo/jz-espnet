#!/usr/bin/env bash

# Copyright (C) 2020, Tianzi Wang

if [[ ! -e "$1/train.tar.gz" || ! -e "$1/dev.tar.gz" ]]; then
    echo "You need to download the MGB-2 first and copy dev.tar.gz and train.tar.gz to $mgb2_dir"
    echo "check: https://arabicspeech.org/mgb2"
    exit 1
fi

if [ $# -ne 2 ]; then
    echo "Usage: $0 <.tar-dir> <target-dir>"
    exit 0
fi

(cd $2; rm -fr train dev test; for x in $1/*; do tar -xvf $x; done)
