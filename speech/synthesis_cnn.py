#!/usr/bin/env python3 
import os 
#import tc
import torch 
import numpy as np
import librosa 
import sys 
import scipy
import argparse
from train_cnn import cnn_model
import scipy.io.wavfile as wav

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda:0, 1" if torch.cuda.is_available() else "cpu")


def make_spectrum_phase(y, framesize=512, overlap=256, fftsize=512, output_dim=257):
    D=librosa.stft(y, n_fft=framesize, hop_length=overlap, win_length=fftsize, window=scipy.hamming)
    Sxx = np.log10(abs(D)**2) 
    phase = np.exp(1j * np.angle(D))
    mean = np.mean(Sxx, axis=1).reshape((output_dim, 1))
    std = np.std(Sxx, axis=1).reshape((output_dim, 1)) + 1e-12
    Sxx = (Sxx - mean) / std  

    return Sxx, phase, mean, std


def recons_spec_phase(Sxx_r, phase):
    Sxx_r = np.sqrt(10**Sxx_r)
    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     hop_length = 256,
                     win_length = 512,
                     window = scipy.hamming)

    return result

parser = argparse.ArgumentParser(description='Log Spectrum Feature Extraction...')
parser.add_argument('--frame_size', default=512, help='Frame size', type=int)
parser.add_argument('--overlap', default=256, help='Overlapping rate between frames', type=int)
parser.add_argument('--fft_size', default=512, help='N for FFT computation', type=int)
parser.add_argument('--sample_rate', default=16000, help='Sampling rate', type=int)
parser.add_argument('--max_data_len', default=200000, help='Maximum data length', type=int)
parser.add_argument('--framewidth', default=8, help='Length of context frames')
parser.add_argument('--n_chans', default=[32, 64, 128, 256], help='Convolutional channels')
parser.add_argument('--input_tensor', default=17*256, help='input tensor')
parser.add_argument('--hidden_layer', default=2048, help='Neurons in the hidden layer')
parser.add_argument('--output_dim', default=256, type=int, help='output dimension')
parser.add_argument('--noise_list_fn', metavar='DIR', default='lists//test//test_noise_0.scp', help='A list of noisy wave files')
parser.add_argument('--load_model_path', default='exp//cnn_models//model_14.hdf5', help='The path to the saved model')

args = parser.parse_args()


if __name__ == "__main__":
    fbin = args.frame_size // 2 + 1
    checkpoint = torch.load(args.load_model_path)
    # model = load_model("exp/model_final.hdf5")
    model = cnn_model(args.n_chans, args.hidden_layer, args.output_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    with open(args.noise_list_fn, 'r') as f:
        for line in f:
            filename = line.split('/')[-1][:-1]
            print(filename)
            y, sr=librosa.load(line[:-1], sr=args.sample_rate)
            training_data = np.empty((args.max_data_len, fbin-1, args.framewidth*2 + 1)) # For Noisy dataß
            Sxx, phase, mean, std = make_spectrum_phase(y, args.frame_size, args.overlap, args.fft_size)
            idx = 0     
            for i in range(args.framewidth, Sxx.shape[1] - args.framewidth): # 5 Frmae
                training_data[idx, :, :] = Sxx[1:, i-args.framewidth : i+args.framewidth+1] # For Noisy data
                idx = idx + 1

            X_train = training_data[:idx, :, :]
            #print("X_train.shape={}".format(X_train.shape))
            X_train = np.reshape(X_train, (idx, -1))
            # predict = model.predict(X_train)
            predict = model.forward(torch.tensor(X_train).type(dtype))
            count=0
            for i in range(args.framewidth, Sxx.shape[1] - args.framewidth):
                Sxx[1:,i] = predict[count].cpu().detach().numpy()
                count+=1
            # # The un-enhanced part of spec should be un-normalized
            Sxx[:, :args.framewidth] = (Sxx[:, :args.framewidth] * std) + mean
            Sxx[:, -args.framewidth:] = (Sxx[:, -args.framewidth:] * std) + mean    

            recons_y = recons_spec_phase(Sxx, phase)
            output = librosa.util.fix_length(recons_y, y.shape[0])
            wav.write(os.path.join("enhanced", filename), args.sample_rate, np.int16(output*32767))
