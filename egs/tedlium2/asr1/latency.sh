#!/bin/bash


rm late/*
model=transformer
recog_set=dev

for i in {1..8}; do
    echo $i;
    cat exp/train_trim_sp_pytorch_train_pytorch_${model}_maskctc_block_bl16_specaug/decode_${recog_set}_decode_pytorch_transformer_maskctc_latency_nolm_false/log/decode.$i.log\
	| grep lat1 |cut -d":" -f6 > late/latency.$recog_set.$i.log
    cat exp/train_trim_sp_pytorch_train_pytorch_${model}_maskctc_block_bl16_specaug/decode_${recog_set}_decode_pytorch_transformer_maskctc_latency_nolm_false/log/decode.$i.log\
	| grep lat2 |cut -d":" -f6 >> late/latency.$recog_set.$i.log

done
