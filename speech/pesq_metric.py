#!/usr/bin/env python3

import soundfile as sf 
from pypesq import pesq 
from read_data import read_clean_data, read_noise_data
import argparse


parser = argparse.ArgumentParser(description='Measuring PESQ scores of enhanced speech...')
parser.add_argument('--data_clean_scp', metavar='DIR', default='list//test//test_clean.scp', help='A list of clean training data')
parser.add_argument('--data_enhanced_scp', metavar='DIR', default='list//enhanced.scp', help='A list of noisy training data')

args = parser.parse_args()


if __name__ == "__main__":
    scores = []
    fp_clean = open(args.data_clean_scp)
    fp_noise = open(args.data_enhanced_scp)

    clean_dct = read_clean_data(args.data_clean_scp)
    enhanced_dct = read_noise_data(args.data_enhanced_scp)

    for _key in clean_dct.keys():
        clean_file = clean_dct[_key]
        enhanced_file = enhanced_dct[_key]

        ref, sr = sf.read(clean_file)
        deg, sr = sf.read(enhanced_file[0])

        score = pesq(ref, deg, sr)
        print("{0}: {1}".format(_key, score))
        scores.append(float(score))

    avg_score = sum(scores) / len(clean_dct.keys())
    print("PESQ = {}".format(avg_score))
