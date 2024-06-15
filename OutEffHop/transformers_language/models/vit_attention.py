
import math
from functools import partial
from typing import Optional, Tuple
import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.utils.checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from quantization.utils import BaseEnumOptions
from transformers_language.models.softmax import clipped_softmax, clipped_softmax1
# import torch.nn.Function as F
# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False

# Set to True if wanting to use torch.jit.script on a model
_SCRIPTABLE = False


# use torch.scaled_dot_product_attention where possible
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if 'TIMM_FUSED_ATTN' in os.environ:
    _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
else:
    _USE_FUSED_ATTN = 1  # 0 == off, 1 == on (for tested use), 2 == on (for experimental use)

def logit(p, eps=1e-16):
    p = np.clip(p, eps, 1 - eps)
    return -np.log(1 / p - 1)


class AttentionGateType(BaseEnumOptions):
    none = 0
    unconditional_per_head = 1
    conditional_per_head = 2
    conditional_per_token = 3

def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0
  
def scaled_dot_product_attention(query, key, value, softmax_fn, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = softmax_fn(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class ViTSelfAttentionWithExtras(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            softmax_fn=torch.nn.functional.softmax,
            gamma=None,
            ssm_eps=None,
            tau=None,
            skip_attn=False,
            attn_gate_type=AttentionGateType.none,
            attn_gate_init=None,
            attn_gate_mlp=False,
            attn_gate_mlp2=False,
            attn_gate_linear_all_features=False,
            fine_tuning=False,
            max_seq_length=None,

    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_attention_heads = num_heads
        self.attention_head_size = dim // num_heads
        self.scale = self.attention_head_size ** -0.5
        self.fused_attn = use_fused_attn()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.attention_head_size) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.attention_head_size) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.attn_scores = nn.Identity()  # before attention mask
        self.attn_probs_before_dropout = nn.Identity()
        self.attn_probs_after_dropout = nn.Identity()

        self.gamma = gamma
        self.ssm_eps = ssm_eps
        self.tau = tau
        self.max_seq_length = max_seq_length

        # define softmax function
        
        self.softmax_fn = softmax_fn

        self.skip_attn = skip_attn

        # attention gating
        self.last_gate_avg_prob = None
        self.last_gate_all_probs = None

        self.attn_gate_type = attn_gate_type
        self.attn_gate_init = attn_gate_init
        self.attn_gate_mlp = attn_gate_mlp
        self.attn_gate_mlp2 = attn_gate_mlp2
        self.attn_gate_linear_all_features = attn_gate_linear_all_features

        self.alpha = None
        self.gate_fn = torch.sigmoid
        self.pooling_fn = partial(torch.mean, dim=1, keepdims=True)

        self.fine_tuning = fine_tuning

        # gate scaling factor
        self.gate_scaling_factor = 1.0
        if self.fine_tuning and self.attn_gate_init is not None:
            self.gate_scaling_factor = 1.0 / self.attn_gate_init

        # define gate
        if self.attn_gate_type == AttentionGateType.unconditional_per_head:
            init_alpha = torch.zeros(size=(self.num_attention_heads,))
            self.alpha = nn.Parameter(init_alpha, requires_grad=True)

        elif self.attn_gate_type in (
            AttentionGateType.conditional_per_head,
            AttentionGateType.conditional_per_token,
        ):
            if self.attn_gate_linear_all_features:
                self.alpha = nn.Linear(self.all_head_size, self.num_attention_heads, bias=True)

            else:  # separate predictors for each head
                module_list = []
                for _ in range(self.num_attention_heads):
                    if self.attn_gate_mlp:
                        fc = nn.Sequential(
                            nn.Linear(
                                self.attention_head_size, self.attention_head_size // 4, bias=True
                            ),
                            nn.ReLU(),
                            nn.Linear(self.attention_head_size // 4, 1, bias=True),
                        )
                    elif self.attn_gate_mlp2:
                        fc = nn.Sequential(
                            nn.Linear(
                                self.attention_head_size, self.attention_head_size, bias=True
                            ),
                            nn.ReLU(),
                            nn.Linear(self.attention_head_size, 1, bias=True),
                        )
                    else:
                        fc = nn.Linear(self.attention_head_size, 1, bias=True)

                        if self.attn_gate_init is not None:
                            init_bias = logit(self.attn_gate_init)
                            torch.nn.init.constant_(fc.bias, init_bias)

                        if self.fine_tuning:
                            # init to a very small values
                            torch.nn.init.normal_(fc.weight, mean=0.0, std=0.01)

                    module_list.append(fc)
                self.alpha = nn.ModuleList(module_list)
                
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = x
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_attention_heads, self.attention_head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            context_layer = scaled_dot_product_attention(
                q, k, v, self.softmax_fn,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            
            attn = self.softmax_fn(attn, dim=-1)
            attn = self.attn_probs_before_dropout(attn)
            attn = self.attn_drop(attn)
            attn = self.attn_probs_after_dropout(attn)
            context_layer = attn @ v

            
        # *** Gating ***
        if self.attn_gate_type == AttentionGateType.unconditional_per_head:
            gate = self.gate_fn(self.alpha)  # (H,)
            context_layer *= gate.view(-1, 1, 1)  # (B, H, T, d_head)

            self.last_gate_avg_prob = gate.view(-1)

        elif self.attn_gate_type in (
            AttentionGateType.conditional_per_head,
            AttentionGateType.conditional_per_token,
        ):
            
            x = hidden_states
            
            if self.attn_gate_linear_all_features:  # assume per_token
                alpha = self.alpha(x)  # (B, T, H)
                gate = self.gate_fn(alpha)
                gate = gate.permute(0, 2, 1).contiguous()  # (B, H, T)
                gate = gate.unsqueeze(3)  # (B, H, T, 1)

            else:
                x = self.transpose_for_scores(x)  # (B, H, T, d_head)

                alpha = []
                for head_idx in range(self.num_attention_heads):
                    x_head = x[:, head_idx, ...]  # (B, T, d_head)
                    fc_head = self.alpha[head_idx]
                    alpha_head = fc_head(x_head)  # (B, T, 1)
                    if self.attn_gate_type == AttentionGateType.conditional_per_head:
                        alpha_head = self.pooling_fn(alpha_head)  # (B, 1, 1)
                    alpha.append(alpha_head)
                alpha = torch.stack(alpha, dim=1)  # (B, H, *, 1)
                gate = self.gate_fn(alpha)

            context_layer *= gate * self.gate_scaling_factor

            self.last_gate_all_probs = gate  # all gates to see the distributions
            avg_gate = gate.mean(dim=0)
            self.last_gate_avg_prob = avg_gate.view(self.num_attention_heads, -1).mean(dim=1)


        x = context_layer.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
