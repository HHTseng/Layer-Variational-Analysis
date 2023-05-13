import os
import numpy as np
import librosa
import random
from tqdm import tqdm
import scipy.io.wavfile
from pathlib import Path
from utils import get_filepaths


def add_noise_(clean_wav_path, noise_wav_path, SNR):
    # read c wav
    c, clean_SR = librosa.load(clean_wav_path, sr=16000)
    n, noise_SR = librosa.load(noise_wav_path, sr=16000)

    # if noise shorter than clean wav
    n_ext = [n for _ in range((len(c) // len(n)) + 1)] if len(n) < len(c) else [n]
    n = np.concatenate(n_ext)

    # random cut noise
    start = random.randint(0, len(n) - len(c))
    n = n[start:(start + len(c))]
    n = np.asarray(n)

    P_c = np.dot(c, c)
    P_n = np.dot(n, n)

    # normalize noise
    n = np.sqrt(P_c / ((10.0 ** (SNR / 10.0)) * P_n)) * n
    #     print("Check SNR:", 10 * np.log10(P_c / P_n) )

    x = c + n   # mix noise [x = clean + noise]
    #     x = x / np.max(abs(x))
    return x, clean_SR


clean_paths = {'train': "/mnt/Datasets/transfer/data/train/clean",
               'test': "/mnt/Datasets/transfer/data/test/clean"}

noise_data = '/mnt/Datasets/transfer/data/noise/'

noise_paths = {'train': {'source': ['white_noise', 'sea', 'takeoff', 'train', 'cabin'],
                         'target': ['babycry', 'BELL', 'SIREN']},
               'test': {'source': ['white_noise', 'sea', 'takeoff', 'train', 'cabin'],
                        'target': ['babycry', 'BELL', 'SIREN']}}

SNR_lists = {'train': {'source': [-5, 0, 5, 10], 'target': [-1, 1]},
             'test': {'source': [-5, 0, 5, 10], 'target': [-1, 1]}}

if __name__ == '__main__':
    for stage in ['train', 'test']:
        for k, noises in noise_paths[stage].items():
            for noise in noises:
                for SNR in SNR_lists[stage][k]:
                    progress_bar = tqdm(get_filepaths(clean_paths[stage], sort=True))
                    for clean_file in progress_bar:
                        noisy, clean_SR = add_noise_(clean_file, os.path.join(noise_data, noise + '.wav'), SNR)  # noisy := clean + noise

                        # split clean path
                        struct = Path(clean_file).parts   # folder structures
                        noisy_path = os.path.join(*(struct[:-2]), k, noise, f'{SNR}')
                        noisy_file = os.path.join(noisy_path, struct[-1])

                        # show info
                        progress_bar.set_description(f'[{stage}, {k}, SNR:{SNR:3d}] {noise} + {struct[-1]}', refresh=True)

                        # output noisy
                        Path(noisy_path).mkdir(parents=True, exist_ok=True)
                        scipy.io.wavfile.write(noisy_file, clean_SR, np.int16(noisy * 32767))
