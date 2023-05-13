import os
import numpy as np
import torch
from torch.utils.data import Dataset


class AdaptationData(Dataset):
    def __init__(self, source_files, clean_path, target_path=None, N=None):
        super(AdaptationData, self).__init__()
        self.source_files = source_files[:N] if N else source_files
        self.clean_files = [os.path.join(clean_path, os.path.basename(file)) for file in self.source_files]
        self.target_path = target_path

        if target_path:
            self.target_files = [os.path.join(target_path, os.path.basename(file)) for file in self.source_files]

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source = torch.load(self.source_files[idx])
        try:
            clean = torch.load(self.clean_files[idx])
        except:
            raise ValueError(f'{self.clean_files[idx]}')

        if self.target_path:
            target = torch.load(self.target_files[idx])
            return source, clean, target
        else:
            return source, clean
