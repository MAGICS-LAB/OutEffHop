import torch
import numpy as np
import scipy.io
import os
import pickle
import pandas as pd
import sklearn.model_selection
import torch
import torch.utils.data
# from torchvision import datasets, transforms
import torch.nn.functional as F
import random

### Overlap functions


def softmax_n_shifted_zeros(input: torch.Tensor, n: int, dim=-1) -> torch.Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0
    """
    # compute the maxes along the last dimension
    input_maxes = input.max(dim=dim, keepdim=True).values
    # shift the input to prevent overflow (and underflow in the denominator)
    shifted_inputs = torch.subtract(input, input_maxes)
    # compute the numerator and softmax_0 denominator using the shifted input
    numerator = torch.exp(shifted_inputs)
    original_denominator = numerator.sum(dim=dim, keepdim=True)
    # we need to shift the zeros in the same way we shifted the inputs
    shifted_zeros = torch.multiply(input_maxes, -1)
    # and then add this contribution to the denominator
    denominator = torch.add(original_denominator, torch.multiply(torch.exp(shifted_zeros), n))
    return torch.divide(numerator, denominator)

def softmax_1(input: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (1 + \sum_j exp(x_j))$
    """
    return softmax_n_shifted_zeros(input, 1, dim=dim)

def random_mask_02(x, dim=-1):
    return F.softmax( torch.dropout(x, p=0.2, train=True), dim=dim)

def random_mask_05(x, dim=-1):
    return F.softmax( torch.dropout(x, p=0.5, train=True), dim=dim)

def random_mask_08(x, dim=-1):
    return F.softmax( torch.dropout(x, p=0.8, train=True), dim=dim)

def topk_80(x, dim=-1):
    # x = x * param
    vals, idx = x.topk(int(0.8*len(x)))

    topk = torch.zeros_like(x)
    topk[idx] = vals

    return F.softmax(topk, dim=dim)

def topk_50(x, dim=-1):

    vals, idx = x.topk(int(0.5*len(x)))

    topk = torch.zeros_like(x)
    topk[idx] = vals

    return F.softmax(topk, dim=dim)

def topk_20(x, dim=-1):

    vals, idx = x.topk(int(0.2*len(x)))

    topk = torch.zeros_like(x)
    topk[idx] = vals

    return F.softmax(topk, dim=dim)

def get_kernel_function(kernel):
    return kernel.forward

def kernel_function(u, v, kernel):
    return kernel(u).T @ kernel(v)
    
def dot_product(u, v):
    return u.T @ v

def manhhatan_distance(u, v):

    v = v.unsqueeze(-1).repeat(1, u.size(-1))
    return torch.abs(u-v).sum(0)

def l2_distance(u, v):

    v = v.unsqueeze(-1).repeat(1, u.size(-1))
    return torch.sqrt(torch.square(u-v).sum(0))



def polynomial(x,param=10, dim=-1):
    return torch.pow(x, param)

def MHN_energy(Xi, x, beta=1):
    # x: D, Xi: (D, M)
    e = -torch.logsumexp(beta*(Xi.T @ x), dim=0) + 0.5*(torch.dot(x,x)) + torch.log(torch.tensor(Xi.size(-1))) + 0.5
    return e

def kernel_fn(W, x):

    # W: (D, D)
    # x: (D, n)
    return W@x

def LMHN_energy(Xi, x, w, beta=1):
    # x: D, Xi: (D, M)
    phi_Xi = kernel_fn(w, Xi)
    phi_x = kernel_fn(w, x)
    e = -torch.logsumexp(beta*(phi_Xi.T @ phi_x), dim=0) + 0.5*(torch.dot(phi_x,phi_x)) + torch.log(torch.tensor(Xi.size(-1))) + 0.5
    return e

def LMHN_update_rule(Xi, x, W, beta=1, steps=1):

    # W: (D, D)
    # Xi: (D, M)
    # x: (D)

    for _ in range(steps):
        phi_x = kernel_fn(W, x)
        phi_Xi = kernel_fn(W, Xi)
        score = beta * F.softmax(phi_Xi.T @ phi_x, dim=-1)
        x =  Xi @ score

    return x

def MHN_update_rule(Xi, x, beta, steps, activation=F.softmax, overlap=dot_product):

    for _ in range(steps):
        score = beta * activation(overlap(Xi, x), dim=-1)
        x = Xi @ score
    return x

def UMHN_update_rule(Xi, x, beta, steps, overlap, activation=F.softmax):

    # overlap function here is a kernel
    for _ in range(steps):
        score = beta * activation(overlap(Xi, x), dim=-1)
        x = Xi @ score
    return x