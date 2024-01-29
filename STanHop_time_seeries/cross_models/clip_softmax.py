import torch
import torch.nn as nn
from cross_models.softmax_1 import softmax_1

def clipped_softmax(data, dim=1, eta=1.1, gamma=-0.1, **kw):
    sm_out = torch.nn.functional.softmax(data, dim=dim, **kw)
    stretched_out = sm_out * (eta - gamma) + gamma
    return torch.clip(stretched_out, 0, 1)

def clipped_softmax_1(data, dim=1, eta=1.1, gamma=-0.1, **kw):
    sm_out = softmax_1(data, dim=dim, **kw)
    stretched_out = sm_out * (eta - gamma) + gamma
    return torch.clip(stretched_out, 0, 1)

class ClipSoftmax(nn.Module):
    __constants__ = ["dim"]

    def __init__(self, dim=-1, eta=1.1, gamma=-0.1):
        """
        dim: The dimension we want to cast the operation over. Default -1
        """
        super(ClipSoftmax, self).__init__()
        self.dim = dim
        self.eta = eta
        self.gamma = gamma

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input):
        a = clipped_softmax(input, self.dim, self.eta, self.gamma) 
        return a

    def extra_repr(self):
        return f"dim={self.dim}"

class ClipSoftmax_1(nn.Module):
    __constants__ = ["dim"]

    def __init__(self, dim=-1, eta=1.1, gamma=-0.1):
        """
        dim: The dimension we want to cast the operation over. Default -1
        """
        super(ClipSoftmax, self).__init__()
        self.dim = dim
        self.eta = eta
        self.gamma = gamma

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input):
        a = clipped_softmax_1(input, self.dim, self.eta, self.gamma) 
        return a

    def extra_repr(self):
        return f"dim={self.dim}"