import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Generic(nn.Module):
    def __init__(self):
        super(Generic, self).__init__()

    def forward(self, x):
        return x**2

def build_model(h_dim = 64, h_lyers = 3, act_func = Sine):#nn.ReLU):
    lyer_ls = [nn.Linear(1,h_dim), act_func()]

    for _ in range(h_lyers):
        lyer_ls.append(nn.Linear(h_dim, h_dim))
        lyer_ls.append(act_func())

    lyer_ls.append(nn.Linear(h_dim, 1))

    return nn.Sequential(*lyer_ls)

