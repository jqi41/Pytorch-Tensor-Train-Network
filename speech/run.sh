#!/usr/bin/env bash

set -eu

train_clean_dir=data/segan/clean_trainset_wav_16k
train_noise_dir=data/segan/noisy_trainset_wav_16k
test_clean_dir=data/segan/clean_testset_wav_16k
test_noise_dir=data/segan/noisy_testset_wav_16k
train_noise_scp=train_noise.scp
train_clean_scp=train_clean.scp
test_noise_scp=test_noise.scp 
test_clean_scp=test_clean.scp 
LIST_DIR=lists

mkdir -p $LIST_DIR
find $train_clean_dir -type f -name '*.wav' > $LIST_DIR//$train_clean_scp
find $train_noise_dir -type f -name '*.wav' > $LIST_DIR//$train_noise_scp
find $test_clean_dir  -type f -name '*.wav' > $LIST_DIR//$test_clean_scp 
find $test_noise_dir  -type f -name '*.wav' > $LIST_DIR//$test_noise_scp

mkdir -p lists/train
mkdir -p lists/test
python read_data.py --data_clean_scp=$train_clean_scp --data_type=train --data_noise_scp=$train_noise_scp --data_path=lists --split_num=1100
python read_data.py --data_clean_scp=$test_clean_scp --data_type=test --data_noise_scp=$test_noise_scp --data_path=lists --split_num=1100

mkdir -p exp 
mkdir -p exp/train_feats
mkdir -p exp/test_feats
for i in `seq 0 23`; do 
    data_clean_scp=$LIST_DIR/train/train_clean_${i}.scp 
    data_noise_scp=$LIST_DIR/train/train_noise_${i}.scp
    feat_fn=exp/train_feats/feats_${i}.h5 
    python feat_extract.py --clean_list_fn=${data_clean_scp} --noise_list_fn=${data_noise_scp} --data_fn=${feat_fn}
done
data_clean_scp=$LIST_DIR/test/test_clean_0.scp
data_noise_scp=$LIST_DIR/test/test_noise_0.scp
feat_fn=exp/test_feats/feats_0.h5
python feat_extract.py --clean_list_fn=${data_clean_scp} --noise_list_fn=${data_noise_scp} --data_fn=${feat_fn}
mkdir -p exp/cnn_models
python train_cnn.py --n_data=23 --n_epochs=15 --lr=0.0002

mkdir -p enhanced
python synthesis_cnn.py 

enhanced_scp='lists/enhanced.scp'
find enhanced -type f -name '*.wav' > $enhanced_scp
python pesq_metric.py --data_enhanced_scp=$enhanced_scp --data_clean_scp=$data_clean_scp


