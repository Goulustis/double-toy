from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class XDer(Dataset):
    def __init__(self, min_val=-1e2, max_val = 1e2, sz = 1e4):
        self.max_val = max_val
        self.min_val = min_val

        self.gap = self.max_val - self.min_val

        self.size = int(sz)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.gap*torch.rand(1) + self.min_val
        # return x, torch.tensor([1.]).float()
        return x, 2*x


