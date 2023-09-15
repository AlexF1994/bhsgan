from torch.utils.data import Dataset
import torch
import numpy as np


class SampleDataset(Dataset):
    def __init__(self, sample: np.ndarray):
        self.sample = torch.from_numpy(sample)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return torch.select(self.sample, 0, idx)