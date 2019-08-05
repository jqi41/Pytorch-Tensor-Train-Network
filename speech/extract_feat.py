#!/usr/bin/env python3
import sys 
import os 

import numpy as np 
import argparse 
import librosa 
import h5py 
import scipy 
import json 
import logging 

parser = argparse.ArgumentParser(description='Log Spectrum Feature Extraction...')
parser.add_argument('--frame_size', default=512, help='Frame size', type=int)
parser.add_argument('--overlap', default=256, help='Overlapping rate between frames', type=int)
parser.add_argument('--fft_size', default=512, help='N for FFT computation', type=int)
parser.add_argument('--sample_rate', default=16000, help='Sampling rate', type=int)
parser.add_argument('--max_data_len', default=2**20, help='Maximum data length', type=int)
parser.add_argument('--framewidth', default=1, help='Length of context frames')
parser.add_argument('--train_clean_list_fn', metavar='DIR', default='clean_list_train', help='A list of training clean wave files')
parser.add_argument('--train_noisy_list_fn', metavar='DIR', default='noisy_list_train', help='A list of training noisy wave files')
parser.add_argument('--test_clean_list_fn', metavar='DIR', default='clean_list_test', help='A list of testing clean wave files')
parser.add_argument('--test_noisy_list_fn', metavar='DIR', default='noisy_list_test', help='A list of testing noisy wave files')
parser.add_argument('--h5_spectrum_fn', metavar='DIR', default='data_spectrum_tensors.h5', help='Feature container in h5 format')
parser.add_argument('--is_tensor', default=0, help='Is tensor feature or not', type=int)
# find ./data/noisy_trainset_56spk_wav -type f -name '*.wav' > noisy_list
args = parser.parse_args()


def make_spectrum_phase(y, framesize, overlap=256, fftsize=512, input=True):
    d = librosa.stft(y, n_fft=framesize, hop_length=overlap, win_length=fftsize, window=scipy.hamming)
    spec = np.log10(abs(d)**2)
    phase = np.exp(1j * np.angle(d))
    mean = np.mean(spec, axis=1).reshape((-1, 1))
    std = np.std(spec, axis=1).reshape((-1, 1)) + 1e-12
    spec = (spec - mean) / std 

    return spec, phase, mean, std 

def recons_spectrum_phase(spec_r, phase, overlap=256, fftsize=512):
    spec_r = np.sqrt(10**spec_r)
    r = np.multiply(spec_r, phase)
    result = librosa.istft(r, hop_length=overlap, win_length=fftsize, window=scipy.hamming)

    return result 

def extract_noisy_feats(noisy_list_fn):
    fbin = args.frame_size // 2 
    noisy_data = np.empty((args.max_data_len, fbin, args.framewidth*2+1), dtype=np.float16)
    noisy_c0 = np.empty((args.max_data_len, 1, args.framewidth*2+1), dtype=np.float16)
    idx = 0
    with open(args.train_noisy_list_fn, 'r') as f:
        for ln in f:
            fn = ln.split('/')[-1][:-1]
            print(fn)
            y, _ = librosa.load(ln[:-1], sr=args.sample_rate)
            S, phase, _, _ = make_spectrum_phase(y, args.frame_size, args.overlap, args.fft_size, input=True)
            if idx + S.shape[1] > args.max_data_len:
                break
            for i in range(args.framewidth, S.shape[1]-args.framewidth):    # 5 frames (long-context feature)
                noisy_data[idx, :, :] = S[1:, i-args.framewidth:i+args.framewidth+1]
                noisy_c0[idx, 0, :] = S[0, i-args.framewidth:i+args.framewidth+1]
                idx = idx + 1
                if idx == args.max_data_len:
                    break
    f.close()
    noisy_data = noisy_data[:idx]
    if args.is_tensor == 0:
        noisy_data = np.reshape(noisy_data, (idx, -1))

    return noisy_data, phase, noisy_c0

def extract_clean_feats(clean_list_fn):
    fbin = args.frame_size // 2 
    clean_data = np.empty((args.max_data_len, fbin), dtype=np.float16)
    clean_c0 = np.empty((args.max_data_len, 1), dtype=np.float16)
    c_idx = 0
    with open(args.train_clean_list_fn, 'r') as f:
        for ln in f:
            fn = ln.split('/')[-1][:-1]
            print(fn)
            y, _ = librosa.load(ln[:-1], sr=args.sample_rate)
            S, phase, _, _ = make_spectrum_phase(y, args.frame_size, args.overlap, args.fft_size, input=False)
            if c_idx + S.shape[1] > args.max_data_len:
                break
            for i in range(args.framewidth, S.shape[1]-args.framewidth):  # 5 frames
                clean_data[c_idx, :] = S[1:, i]
                clean_c0[c_idx, 0] = S[0, i]  
                c_idx = c_idx + 1
                if c_idx == args.max_data_len:
                    break
    f.close()
    clean_data = clean_data[:c_idx]

    return clean_data, phase, clean_c0


if __name__=='__main__':
    if os.path.exists(args.h5_spectrum_fn):
        os.remove(args.h5_spectrum_fn)

    with h5py.File(args.h5_spectrum_fn, 'a') as hf:
        train_noisy_data, _, _ = extract_noisy_feats(args.train_noisy_list_fn)
        hf.create_dataset('train_noisy', data=train_noisy_data)
        train_noisy_data = []

        test_noisy_data, _, _ = extract_noisy_feats(args.test_noisy_list_fn)
        hf.create_dataset('test_noisy', data=test_noisy_data)
        test_noisy_data = []

        train_clean_data, _, _ = extract_clean_feats(args.train_clean_list_fn)
        hf.create_dataset('train_clean', data=train_clean_data)  
        train_clean_data = []

        test_clean_data, _, _ = extract_clean_feats(args.test_clean_list_fn)
        hf.create_dataset('test_clean', data=test_clean_data)
        test_clean_data = []
