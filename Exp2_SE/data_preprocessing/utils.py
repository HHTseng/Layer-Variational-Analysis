import os
import numpy as np
import librosa
import scipy


def get_filepaths(directory, ftype='.wav', sort=True):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)

    file_paths = sorted(file_paths) if sort else file_paths
    return file_paths


def make_spectrum(filename=None, y=None, feature_type='logmag', _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    D = librosa.stft(y, center=False, n_fft=512, hop_length=160, win_length=512, window=scipy.signal.hamming)
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    # select feature types
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    return Sxx, phase, len(y)