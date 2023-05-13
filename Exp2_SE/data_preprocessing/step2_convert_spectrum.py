import librosa
import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from utils import get_filepaths, make_spectrum

epsilon = np.finfo(float).eps
SR = 16000   # sampling rate

data_root = '/mnt/Datasets/transfer/data/'

if __name__ == '__main__':
    for stage in ['train']:
        print(f'converting {stage} files')
        train_path = os.path.join(data_root, stage)
        train_convert_save_path = os.path.join(train_path + '_log1p_pt')

        n_frame = 128
        wav_files = get_filepaths(train_path)
        for wav_file in tqdm(wav_files):
            wav, sr = librosa.load(wav_file, sr=SR)
            out_path = wav_file.replace(train_path, train_convert_save_path).split('.w')[0]
            data, _, _ = make_spectrum(y=wav)
            for i in np.arange(data.shape[1] // n_frame):
                Path(os.path.join(*(Path(out_path).parts[:-1]))).mkdir(parents=True, exist_ok=True)
                out_name = out_path + f'_{i}.pt'
                torch.save(torch.from_numpy(data.transpose()[i * n_frame: (i + 1) * n_frame]), out_name)