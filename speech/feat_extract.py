#!/usr/bin/env python3

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import scipy
import scipy.io.wavfile as wav
from scipy import signal
import h5py
import librosa
import argparse

parser = argparse.ArgumentParser(description='Log Spectrum Feature Extraction...')
parser.add_argument('--frame_size', default=512, help='Frame size', type=int)
parser.add_argument('--overlap', default=256, help='Overlapping rate between frames', type=int)
parser.add_argument('--fft_size', default=512, help='N for FFT computation', type=int)
parser.add_argument('--sample_rate', default=16000, help='Sampling rate', type=int)
parser.add_argument('--max_data_len', default=500000, help='Maximum data length', type=int)
parser.add_argument('--framewidth', default=8, help='Length of context frames')
parser.add_argument('--is_tensor', default=0, help='Is tensor feature or not', type=int)
parser.add_argument('--data_fn', metavar='DIR', default='feat_data.h5', help='file for feature extraction')
parser.add_argument('--noise_list_fn', metavar='DIR', default='noisy.scp', help='A list of clean wave files')
parser.add_argument('--clean_list_fn', metavar='DIR', default='clean.scp', help='A list of noisy wave files')
# find ./data/noisy_trainset_56spk_wav -type f -name '*.wav' > noisy_list
args = parser.parse_args()


if __name__ == "__main__":

    fbin = args.frame_size//2 + 1
    data_name = args.data_fn                #"/mnt/hd-01/user_sylar/MHINTSYPD_100NS/data_257_spectrum.h5"

    noisylistpath = args.noise_list_fn      #"/mnt/hd-01/user_sylar/MHINTSYPD_100NS/trnoisylist"
    print("Expected data size: {0}".format(args.max_data_len))
    noisydata = np.zeros((args.max_data_len, fbin-1, args.framewidth*2+1), dtype=np.float32)
    idx = 0
    with open(noisylistpath, 'r') as f:
        for line in f:
            filename = line.split('/')[-1][:-1]
            print(filename)
            y, sr = librosa.load(line[:-1], sr=args.sample_rate)
            D = librosa.stft(y, n_fft=args.fft_size, hop_length=args.overlap, win_length=args.frame_size, window=scipy.hamming)
            Sxx = np.log10(abs(D)**2) 
            #print('spec.shape' + str(Sxx.shape))
            mean = np.mean(Sxx, axis=1).reshape(fbin, 1)
            std = np.std(Sxx, axis=1).reshape(fbin, 1) + 1e-12
            Sxx = (Sxx - mean) / std
            for i in range(args.framewidth, Sxx.shape[1]-args.framewidth): # 5 Frmae
                noisydata[idx, :, :] = Sxx[1:, i-args.framewidth:i+args.framewidth+1] # For Noisy data
                idx = idx + 1

    noisydata = noisydata[:idx]
    noisydata = np.reshape(noisydata, (idx, -1))
    #===================================================================================#

    with h5py.File(data_name, 'a') as hf:
        hf.create_dataset('noise', data=noisydata) # For Noisy data
    noisdydata = []

    cleanlistpath = args.clean_list_fn #"/mnt/hd-01/user_sylar/MHINTSYPD_100NS/trcleanlist"
    cleandata = np.zeros((args.max_data_len, fbin-1), dtype=np.float32)
    c_idx = 0
    with open(cleanlistpath, 'r') as f:
        for line in f:
            filename = line.split('/')[-1][:-1]
            print(filename)
            y, sr=librosa.load(line[:-1],sr=args.sample_rate)
            D = librosa.stft(y, n_fft=args.fft_size, hop_length=args.overlap, win_length=args.frame_size, window=scipy.hamming)
            Sxx = np.log10(abs(D)**2) 
        #   print('spec.shape' + str(Sxx.shape))
            for i in range(args.framewidth, Sxx.shape[1]-args.framewidth):  # 5 Frmae        
                cleandata[c_idx,:] = Sxx[1:, i] # For Clean data
                c_idx = c_idx + 1

    cleandata = cleandata[:c_idx]

    with h5py.File(data_name, 'a') as hf:
        hf.create_dataset('clean', data=cleandata) # For Clean data
