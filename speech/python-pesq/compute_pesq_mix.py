import soundfile as sf
from pypesq import pesq

import scipy
import mir_eval
import numpy as np
import librosa
from os import listdir
from os.path import isfile, join
from itertools import permutations
import glob
import os
import sys

def read_wav(path):
    sr, data = scipy.io.wavfile.read(path)
    return data


def si_snr(x, s, remove_dc=True):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


def permute_si_snr(xlist, slist):
    """
    Compute SI-SNR between N pairs
    Arguments:
        x: list[vector], enhanced/separated signal
        s: list[vector], reference signal(ground truth)
    """

    def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)

    N = len(xlist)
    if N != len(slist):
        raise RuntimeError(
            "size do not match between xlist and slist: {:d} vs {:d}".format(
                N, len(slist)))
    si_snrs = []
    for order in permutations(range(N)):
        si_snrs.append(si_snr_avg(xlist, [slist[n] for n in order]))
    
    return max(si_snrs)



ref_dir = "../../data_simu/2spk_2s_center_crop112/test" #should have "mix", "spk1", "spk2"
#ref_list = [line.rstrip("\n") for line in open("/data2/wujian/WSJ0/ailab_6ch_wsj0/tt/tt.list")]
enh_dir_spk1="../out_newsnr_center112_epoch48_loss_neg7d5_spk1"
enh_dir_spk2="../out_newsnr_center112_epoch48_loss_neg7d5_spk2"



pesq_total=0.0
ref_list=glob.glob(os.path.join(ref_dir,"mix","*.wav"))

for i in range(len(ref_list)):
    fname = ref_list[i].split('/')[-1] # name of the file
    #ad = int(fname.split("_")[-1].replace("ad", "").replace(".wav", "")) # angle difference

    est_i_1,sr = sf.read(ref_list[i])
    est_i_2,sr = sf.read(ref_list[i])  
    
    ref_i_1,sr = sf.read(ref_dir + "/spk1/" + fname)
    ref_i_2,sr = sf.read(ref_dir + "/spk2/" + fname)
    
    min_len = np.min (( len(est_i_1), len(est_i_2), len(ref_i_1), len(ref_i_2) ))
     
    est_i_1 = est_i_1 [0:min_len]
    est_i_2 = est_i_2 [0:min_len]
    ref_i_1 = ref_i_1 [0:min_len]
    ref_i_2 = ref_i_2 [0:min_len]

    pesq_spk1=pesq(ref_i_1,est_i_1,16000)
    pesq_spk2=pesq(ref_i_2,est_i_2,16000)
    pesq_tmp=(pesq_spk1+pesq_spk2)/2.0
    pesq_total=pesq_total+pesq_tmp

    print("utt: " + str(i) + " pesq is: " +  str(pesq_tmp) )


# stats 
print("============================================")

# PESQ results
print("============================================")
print("PESQ results: ")
print("Average PESQ: ", pesq_total/len(ref_list))

