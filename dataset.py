from torch.utils.data import Dataset
import torchvision
import torch
import numpy as np


class SampleDataset(Dataset):
    def __init__(self, sample: np.ndarray):
        self.sample = torch.from_numpy(sample)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return torch.select(self.sample, 0, idx)


class SubCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, include_list=[], **kwargs):
        super().__init__(*args, **kwargs)

        if include_list == []:
            return

        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = (labels.reshape(-1, 1) == include).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()
