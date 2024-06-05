import torch
import torch.nn as nn

def softmax_n(x, n = None, dim=-1, dtype=None):
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    if n is None:
        n = 0.
    if dim is None:
        dim = -1
    shift = x.max(dim=dim, keepdim=True).values.detach()
    numerator = torch.exp(x - shift)
    output = numerator / (n * torch.exp(-shift) + numerator.sum(dim=dim, keepdim=True))
    return output if dtype is None else output.type(dtype=dtype)



def softmax_1(x, dim=-1, _stacklevel=3, dtype=None):
    #subtract the max for stability
    return softmax_n(x, 1, dim=dim)
    
class Softmax_1(nn.Module):
    __constants__ = ["dim"]

    def __init__(self, dim=-1):
        """
        dim: The dimension we want to cast the operation over. Default -1
        """
        super(Softmax_1, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input):
        a = softmax_1(input, self.dim) 
        return a

    def extra_repr(self):
        return f"dim={self.dim}"

