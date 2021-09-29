import torch

import matplotlib.pyplot as plt

from model import build_model


def test():
    X = torch.linspace(-30,30, steps=30).reshape(-1,1)

    model = build_model()
    model.load_state_dict(torch.load("ckpoint.pth"))

    with torch.no_grad():
        out = model(X)

    X = X.squeeze()
    out = out.squeeze()

    plt.plot(X, out)
#    plt.plot(X,X)
    plt.plot(X, X**2)
    plt.show()

if __name__ == "__main__":
    test()
