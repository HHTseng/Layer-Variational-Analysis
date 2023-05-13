import os
import numpy as np
import torch
from torch.utils.data import Dataset


def image_paths(scale, mode, path='./'):
    if scale == 'multiple':
        scale = [3, 4] if mode == 'train' else [4, 6]

    input = [os.path.join(path, f'blur_scale_{s}/X_{mode}.npy') for s in scale]
    label = os.path.join(path, f'blur_scale_1/X_{mode}.npy')
    return input, label


class TrainDataset(Dataset):
    def __init__(self, input_path, label_path, N=None):
        super(TrainDataset, self).__init__()
        k = len(input_path)

        # number of samples
        if N==None: N = np.load(input_path[0]).shape[0]

        # mixing samples of k blurring scales
        self.input = np.concatenate([np.load(_)[(i * (N // k)): ((i + 1) * (N // k))] for i, _ in enumerate(input_path)], axis=0)
        self.label = np.load(label_path)[:N]

        # to Torch tensor
        self.input = torch.from_numpy(self.input)
        self.label = torch.from_numpy(self.label)

    def __getitem__(self, idx):
        X = self.input[idx]
        y = self.label[idx]

        return X, y

    def __len__(self):
        return len(self.input)