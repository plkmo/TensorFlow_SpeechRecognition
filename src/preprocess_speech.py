# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:16:42 2019

@author: WT
"""
from scipy.io import wavfile
from scipy import signal
import numpy as np
import os
import pickle
import librosa
import re
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size*sample_rate/1e3))
    noverlap = int(round(step_size*sample_rate/1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def extract_MFCC(args):
    ### Extract MFCC coefficients for all data
    logger.info("Extracting MFCC features...")
    data = []; length_log = []
    fixed_len = args.mfcc_bin_len # fix the MFCC time length
    subfolders = [f.name for f in os.scandir("data") if f.is_dir()]
    labels = {"yes":0, "no":1, "up":2, "down":3, "left":4, "right":5, "on":6, "off":7, "stop":8, "go":9,\
              "silence":10, "unknown":11}
    for idx, name in tqdm(enumerate(subfolders)):
        if name == "_background_noise_":
            for file in os.listdir("./data/" + name):
                if re.match(".+.wav", file) is not None:
                    sample_rate, samples = wavfile.read(os.path.join("./data/%s/" % name, file))
                    mfcc = librosa.feature.mfcc(y=samples.astype(float), sr=sample_rate, n_mfcc=args.n_mfcc,\
                                                n_fft=args.n_fft, hop_length=args.hop_length)
                    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
                    for i in range(len(delta2_mfcc[0,:])//fixed_len):
                        s = delta2_mfcc[:, i*fixed_len:(i + 1)*fixed_len]
                        if len(s[0, :]) == fixed_len:
                            length_log.append(len(s[0,:]))
                            data.append((s, 10)) # "other" label
        else:    
            for file in os.listdir("./data/" + name):
                if re.match(".+.wav", file) is not None:
                    sample_rate, samples = wavfile.read(os.path.join("./data/%s/" % name, file))
                    mfcc = librosa.feature.mfcc(y=samples.astype(float), sr=sample_rate, n_mfcc=args.n_mfcc,\
                                                n_fft=args.n_fft, hop_length=args.hop_length)
                    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
                    if len(delta2_mfcc[0,:]) != fixed_len:
                        s = delta2_mfcc[:, :fixed_len]
                        while len(s[0,:]) < fixed_len:
                            s = np.append(s, np.zeros((len(s),1)), axis=1)                    
                        length_log.append(len(s[0,:]))
                        if name in labels.keys():
                            data.append((s, labels[name]))
                        else:
                            data.append((s, 10)) # "other" label
    save_as_pickle("data.pkl", data)
    save_as_pickle("length_log.pkl", length_log)
    logger.info("Done extracting MFCC features!")
