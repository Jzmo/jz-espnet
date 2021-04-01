#!/bin/bash


recog_set=test
bl=16

for i in {1..8}; do
    echo $i;
    cat exp/train_trim_sp_pytorch_train_pytorch_conformer_maskctc_block_bl16_specaug/decode_test_decode_pytorch_transformer_maskctc_latency_nolm_false/log/decode.$i.log\
	| grep latency1 | cut -d"[" -f3 | cut -d"]" -f1 > late/latency.$i.log
    echo plus $bl
    cat exp/train_trim_sp_pytorch_train_pytorch_conformer_maskctc_block_bl16_specaug/decode_test_decode_pytorch_transformer_maskctc_latency_nolm_false/log/decode.$i.log\
	| grep latency2 | cut -d"[" -f3 | cut -d"]" -f1 >> late/latency.$i.log

done
