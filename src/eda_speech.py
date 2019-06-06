# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:16:42 2019

@author: WT
"""
from scipy.io import wavfile
from scipy import signal
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display as display

### Use Recurrent models for on-the-fly speech recognition

data_path = "./data/"
filename = "bird/00b01445_nohash_0.wav"

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

sample_rate, samples = wavfile.read(os.path.join(data_path, filename))
freqs, times, spectrogram = log_specgram(samples, sample_rate)

fig = plt.figure(figsize=(14,8))
ax1 = fig.add_subplot(211)
ax1.set_title("Raw wave of %s" % filename)
ax1.set_ylabel("Amplitude")
ax1.plot(np.linspace(0, len(samples)/sample_rate, len(samples)), samples)

ax2 = fig.add_subplot(212)
ax2.set_title("Log Spectrogram of %s" % filename)
ax2.set_xlabel("t (s)")
ax2.set_ylabel("Log Amplitude")
ax2.imshow(spectrogram.T, aspect="auto", origin="lower",\
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])

S = librosa.feature.melspectrogram(samples.astype(float), sr=sample_rate, n_mels=128)
lg_S = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(13,5))
display.specshow(lg_S, sr=sample_rate, x_axis="time", y_axis="mel")
plt.title("Mel power spectrogram")
plt.colorbar(format="%+02.0f dB")
plt.tight_layout()

mfcc = librosa.feature.mfcc(S=lg_S, n_mfcc=13)
# Let's pad on the first and second deltas while we're at it
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(12, 4))
librosa.display.specshow(delta2_mfcc)
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()
