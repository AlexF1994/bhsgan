import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        )
    
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples,z_dim,device=device)