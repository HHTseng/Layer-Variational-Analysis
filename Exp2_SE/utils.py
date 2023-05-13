import os
import numpy as np
import librosa
import scipy
import torch
from pypesq import pesq
from pystoi.stoi import stoi
from pathlib import Path
from tqdm import tqdm

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_filepaths(directory, ftype='.wav', sort=True):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)

    file_paths = sorted(file_paths) if sort else file_paths
    return file_paths


def cal_score_wav(clean, enhanced):
    clean = clean/abs(clean).max()
    enhanced = enhanced/abs(enhanced).max()
    s_stoi = stoi(clean, enhanced, 16000)
    s_pesq = pesq(clean, enhanced, 16000)
    return round(s_pesq, 5), round(s_stoi, 5)


def make_spectrum(filename=None, y=None, feature_type='logmag'):
    if filename:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype != 'float32':
            y = np.float32(y)

    D = librosa.stft(y, center=False, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    # convert feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    else:
        Sxx = D
    return Sxx, phase, len(y)


def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)

    R = np.multiply(Sxx_r, phase)
    result = librosa.istft(R, center=False, hop_length=256, win_length=512, window=scipy.signal.hamming, length=length_wav)
    return result


def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)


def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)


def write_score(model, device, criterion, audio_path, clean_path, domain, score_file=None):
    model.eval()

    if os.path.exists(score_file):
        os.remove(score_file)
    with open(score_file, 'a') as file:
        file.write('Filename,noise,SNR,PESQ,STOI,LOSS\n')

    progress_bar = tqdm(get_filepaths(audio_path, sort=True))
    for enhanced_wav in progress_bar:
        struct = Path(enhanced_wav).parts
        filename = os.path.splitext(struct[-1])[0]
        SNR, noise_type = struct[-2], struct[-3]

        # load clean label
        clean, sr = librosa.load(os.path.join(clean_path, struct[-1]), sr=16000)
        clean_spec, _, _ = make_spectrum(y=clean)  # clean -- STFT-->  clean spectrum

        # load noisy
        noisy, sr = librosa.load(enhanced_wav, sr=16000)  # noisy -- STFT-->  noisy spectrum
        noisy_spec, noisy_phase, n_len = make_spectrum(y=noisy)
        noisy_spec = torch.from_numpy(noisy_spec.transpose()).to(device).unsqueeze(0)

        # denoise prediction
        pred = model(noisy_spec)[1].cpu().detach().numpy()
        enhanced = recons_spec_phase(pred.squeeze().transpose(), noisy_phase, n_len)

        # compute scores
        loss = criterion(torch.from_numpy(pred.squeeze().transpose()), torch.from_numpy(clean_spec))
        PESQ, STOI = cal_score_wav(clean, enhanced)
        progress_bar.set_description(f'[{domain}, {noise_type}, {SNR}dB, {filename}] Loss: {loss:.4f}, PESQ: {PESQ:.2f}, STOI: {STOI:.2f}', refresh=True)

        with open(score_file, 'a') as file:
            file.write(f'{filename},{noise_type},{SNR},{PESQ},{STOI},{loss}\n')
