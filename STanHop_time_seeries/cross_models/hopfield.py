import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt
from cross_models.entmax import EntmaxAlpha
from cross_models.softmax_1 import Softmax_1
from cross_models.sparse_max import Sparsemax
from cross_models.clip_softmax import ClipSoftmax, ClipSoftmax_1


def logit(p, eps=1e-16):
    p = np.clip(p, eps, 1 - eps)
    return -np.log(1 / p - 1)


class Association(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, eta = 1.1, gamma = -0.1, mode = 'entmax'):
        super(Association, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        if mode == 'entmax':
            self.softmax = EntmaxAlpha()
        elif mode == 'softmax1':
            self.softmax = Softmax_1()
        elif mode == 'softmax':
            self.softmax = nn.Softmax(dim=-1)
        elif mode == 'sparsemax':
            self.softmax = Sparsemax()
        elif mode == 'clip':
            self.softmax = ClipSoftmax(eta = eta, gamma = gamma)
        elif mode == 'clip_softmax1':
            self.softmax = ClipSoftmax_1(eta = eta, gamma = gamma)
        
        
        
        
        # self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(self.softmax(scale * scores))
        attn_output = torch.einsum("bhls,bshd->blhd", A, values)
        
        return attn_output.contiguous()

class Hopfield(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1, eta = 1.1, gamma = -0.1, mode = 'entmax'):
        super(Hopfield, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = Association(scale=None, attention_dropout = dropout, eta = eta, gamma = gamma,  mode = mode)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_keys * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys)
        values = self.value_projection(self.key_projection(keys)).view(B, S, H, -1)
        keys = k.view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)

class HopfieldPooling(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, num_pattern=1, d_keys=None, d_values=None, mix=True, dropout = 0.1, eta = 1.1, gamma = -0.1, mode = 'entmax',):
        super(HopfieldPooling, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = Association(scale=None, attention_dropout = dropout, eta = eta, gamma = gamma, mode = mode)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_keys * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

        pooling_weight_size = d_model
        self.key = nn.Parameter(
            torch.empty(
                size=(
                    *
                    (
                        (1,
                         num_pattern)),
                    d_model if pooling_weight_size is None else pooling_weight_size)),
            requires_grad=True)

    def forward(self, query):

        B, L, _ = query.shape
        _, S, _ = self.key.shape
        H = self.n_heads

        k = self.key.repeat((*((B, 1)), 1))
        queries = self.query_projection(query).view(B, L, H, -1)
        keys = self.key_projection(k)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out)

class STHMLayer(nn.Module):
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, dropout=0.1, eta = 1.1, gamma = -0.1,  mode = 'entmax'):
        super(STHMLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.cross_time = Hopfield(d_model, n_heads, dropout = dropout, eta = eta, gamma = gamma, mode = mode)
        self.cross_series = HopfieldPooling(d_model, n_heads, num_pattern=factor, dropout = dropout, eta=eta, gamma = gamma, mode = mode)
        self.hopfield = Hopfield(d_model, n_heads, dropout = dropout, eta = eta, gamma = gamma, mode = mode)
        
        self.dropout = nn.Dropout(dropout)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        #Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.cross_time(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        # dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        # dim_in = self.norm2(dim_in)

        series_in = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)
        series_h = self.cross_series(series_in)
        pooled_h = self.hopfield(series_in, series_in, series_h)
        dim_enc = series_h + self.dropout(pooled_h)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        # dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)

        return final_out