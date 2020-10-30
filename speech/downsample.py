#!/usr/bin/env python3

import os 
import subprocess 
import numpy as np 
import time 

DATA_ROOT_DIR = "data/segan"
CLEAN_TRAIN_DIR = 'clean_testset_wav'  # where original clean train data exist
NOISY_TRAIN_DIR = 'noisy_testset_wav'  # where original noisy train data exist
DST_CLEAN_TRAIN_DIR = 'clean_testset_wav_16k'  # clean preprocessed data folder
DST_NOISY_TRAIN_DIR = 'noisy_testset_wav_16k'  # noisy preprocessed data folder


def downsample_16k():
    """
    Convert all audio files into ones of sampling rate 16k. 
    """
    # clean training sets
    dst_clean_dir = os.path.join(DATA_ROOT_DIR, DST_CLEAN_TRAIN_DIR)
    if not os.path.exists(dst_clean_dir):
        os.makedirs(dst_clean_dir)

    for dirname, dirs, files in os.walk(os.path.join(DATA_ROOT_DIR, CLEAN_TRAIN_DIR)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            out_filepath = os.path.join(dst_clean_dir, filename)
            # use sox to down-sample to 16k
            print('Downsampling : {}'.format(input_filepath))
            subprocess.run(
                    'sox {} -r 16k {}'
                    .format(input_filepath, out_filepath),
                    shell=True, check=True)

    # noisy training sets
    dst_noisy_dir = os.path.join(DATA_ROOT_DIR, DST_NOISY_TRAIN_DIR)
    if not os.path.exists(dst_noisy_dir):
        os.makedirs(dst_noisy_dir)

    for dirname, dirs, files in os.walk(os.path.join(DATA_ROOT_DIR, NOISY_TRAIN_DIR)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            out_filepath = os.path.join(dst_noisy_dir, filename)
            # use sox to down-sample to 16k
            print('Processing : {}'.format(input_filepath))
            subprocess.run(
                    'sox {} -r 16k {}'
                    .format(input_filepath, out_filepath),
                    shell=True, check=True)

if __name__ == "__main__":
    downsample_16k()
