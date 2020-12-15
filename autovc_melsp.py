# autovc mel spectrogram shared by autovc/dvector/wavenet_vocoder

import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length, hop_length):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
def log_melsp_01(x,
    sr=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    fmin=80,
    fmax=8000):
    '''
    '''
    mel_basis = mel(sr, n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    # 
    # Remove drifting noise
    y = signal.filtfilt(b, a, x)
    # Ddd a little random noise for model roubstness
    prng = RandomState()
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    # Compute spect
    D = pySTFT(wav, fft_length=n_fft, hop_length=hop_length).T
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)    

    return S.astype(np.float32)
    
