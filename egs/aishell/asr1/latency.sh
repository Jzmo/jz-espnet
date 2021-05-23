#!/bin/bash


rm late/*
recog_set=test
bl=16

for i in {1..20}; do
    echo $i;
    cat exp/train_sp_pytorch_train_pytorch_transformer_maskctc_block_bl16/decode_${recog_set}_decode_pytorch_transformer_maskctc_latency/log/decode.$i.log\
	| grep lat1 |cut -d":" -f6 > late/latency.$recog_set.$i.log
    cat exp/train_sp_pytorch_train_pytorch_transformer_maskctc_block_bl16/decode_${recog_set}_decode_pytorch_transformer_maskctc_latency/log/decode.$i.log\
	| grep lat2 |cut -d":" -f6 >> late/latency.$recog_set.$i.log

done
