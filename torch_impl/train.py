import torch
from torch.nn.functional import cosine_similarity as cos_sim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import XDer
from model import build_model

import matplotlib.pyplot as plt
from tqdm import tqdm


def calc_grad(y, x, grad_out = None):
    if grad_out is None:
        grad_out = torch.ones_like(y)

    grad = torch.autograd.grad(y, [x], grad_out, create_graph=True)[0]
    return grad

def grad_mse_loss(inp, mod_out, fx_grad):
    grad = calc_grad(mod_out, inp)
    return F.mse_loss(grad, fx_grad)


def train(num_epochs=30, batch_size = 128, use_cuda=True):
    tr_set = XDer()
    tr_loader = DataLoader(tr_set, batch_size = batch_size, shuffle=True)

    model = build_model()
    if use_cuda:
        model = model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    
    num_iter = len(tr_set)//batch_size
    for epoch in range(num_epochs):
        print("==================", epoch, "==============")
        tot_loss = 0
        for x, x_der in tqdm(tr_loader, total = num_iter):
            if use_cuda:
                x, x_der = x.cuda(), x_der.cuda()
            x = x.requires_grad_(True)

            optimizer.zero_grad()
            out = model(x)
            loss = grad_mse_loss(x, out, x_der)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()

        epoch_loss = tot_loss/num_iter
        losses.append(loss.item())#epoch_loss)
        print("epoch_loss:", loss.item())#epoch_loss)
    
    torch.save(model.state_dict(), "ckpoint.pth")

    plt.plot(losses)
    plt.savefig("loss_train.png")


def main():
    train()

def test_calc_grad():
    x = torch.tensor(list(range(10))).float()
    x_inp = x.clone().requires_grad_(True)
    y = x_inp**2
    grad = calc_grad(y, x_inp)
    print(x)
    print(2*x)
    print(grad)

if __name__ == "__main__":
    main()
    #test_calc_grad()
