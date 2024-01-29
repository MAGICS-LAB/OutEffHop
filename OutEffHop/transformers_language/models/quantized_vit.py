# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import math
from functools import partial
from typing import Any, Callable, Dict, Sequence, Iterator, Tuple, Type, Union, List, Optional, Set
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.jit import Final
from quantization.utils import BaseEnumOptions
from transformers_language.models.softmax import clipped_softmax
import torch.nn.Function as F
import random
import warnings
from torch.utils.checkpoint import checkpoint
from itertools import chain
import os
from vision_transformer import init_weights_vit_moco, get_init_weights_vit, _load_weights, init_weights_vit_jax
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_classes import (
    FP32Acts,
    QuantizedActivation,
    QuantizedModule,
)
from quantization.base_quantized_model import QuantizedModel
from quantization.quantizers import QMethods
from quantization.quantizers.uniform_quantizers import SymmetricUniformQuantizer
from quantization.range_estimators import CurrentMinMaxEstimator
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertSelfAttentionWithExtras,
)
from transformers_language.utils import DotDict

from timm.layers import trunc_normal_, lecun_normal_, resample_abs_pos_embed, use_fused_attn

def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x

def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

DEFAULT_QUANT_DICT = {
    # Attention
    "attn_mask_type": "add",
    # Clip `h` tensor
    "k_std": None,
    # LayerNorm
    "layer_norm_ver": "v1",
    "layer_norm_embd": False,
    "layer_norm_res_self_output": False,
    "layer_norm_res_output": False,
    "layer_norm_n_bits_unary": 8,
    "layer_norm_n_bits_binary": 8,
    "layer_norm_n_bits_params": 8,
}

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

def _make_quant_dict(partial_dict):
    quant_dict = DEFAULT_QUANT_DICT.copy()
    quant_dict.update(partial_dict)
    return DotDict(quant_dict)

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

def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
        
  
def scaled_dot_product_attention(query, key, value, softmax_fn, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
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
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def trunc_normal_tf_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsquently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)
    return tensor





def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

class QuantizedAttentionPoolLatent(QuantizedModel):
    """ Attention pooling w/ latent query
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            org_model,
            qk_norm: bool = False,
            pos_embed: str = '',
            norm_layer: Optional[nn.Module] = None,
            pool_type: str = 'token',
            drop: float = 0.0,
            **quant_params
    ):
        super().__init__()
        self.embed_dim = org_model.embed_dim or in_features
        self.out_features = org_model.out_features or in_features
        assert org_model.embed_dim % org_model.num_heads == 0
        self.num_heads = org_model.num_heads
        self.head_dim = org_model.embed_dim // org_model.num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()

        if pos_embed == 'abs':
            spatial_len = self.feat_size
            self.pos_embed = nn.Parameter(torch.zeros(spatial_len, org_model.in_features))
        else:
            self.pos_embed = Noneself.pos_embed

        self.latent_dim = org_model.latent_dim or org_model.embed_dim
        self.latent_len = org_model.latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, org_model.embed_dim))

        self.q = quantize_model(org_model.q, **quant_params)
        self.kv = quantize_model(org_model.kv, **quant_params)
        self.q_norm = quantize_model(org_model.q_norm, **quant_params) if qk_norm else nn.Identity()
        self.k_norm = quantize_model(org_model.k_norm, **quant_params) if qk_norm else nn.Identity()
        self.proj = quantize_model(org_model.proj, **quant_params)
        self.proj_drop = nn.Dropout(drop)

        self.norm = quantize_model(org_model.norm, **quant_params) if norm_layer is not None else nn.Identity()
        self.mlp = quantize_model(org_model.mlp, **quant_params)
        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def forward(self, x):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x



class QuantizedViTSelfAttentionWithExtras(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        # copy attributes
        self.num_heads = org_model.num_attention_heads
        self.attention_head_size = org_model.attention_head_size
        self.scale = org_model.scale
        self.alpha = org_model.alpha
        self.ssm_eps = org_model.ssm_eps
        self.tau = org_model.tau
        self.max_seq_length = org_model.max_seq_length
        self.fused_attn = org_model.fused_attn
        self.gamma = org_model.gamma

        self.qkv = quantize_model(org_model.qkv, **quant_params)
        self.q_norm = quantize_model(org_model.q_norm, **quant_params)
        self.k_norm = quantize_model(org_model.k_norm, **quant_params)
        self.attn_drop = org_model.attn_drop
        self.proj = quantize_model(org_model.proj, **quant_params)
        self.proj_drop = org_model.proj_drop
        self.attn_scores = org_model.attn_scores
        self.attn_probs_before_dropout = org_model.attn_probs_before_dropout
        self.attn_probs_after_dropout = org_model.attn_probs_after_dropout

        # softmax fn
        self.softmax_fn = org_model.softmax_fn

        # Activation quantizers
        self.attn_scores_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_probs_act_quantizer = QuantizedActivation(**quant_params)
        self.context_act_quantizer = QuantizedActivation(**quant_params)

        # attention gating
        self.attn_gate_type = org_model.attn_gate_type
        self.attn_gate_init = org_model.attn_gate_init
        self.attn_gate_mlp = org_model.attn_gate_mlp
        self.attn_gate_mlp2 = org_model.attn_gate_mlp2
        self.attn_gate_linear_all_features = org_model.attn_gate_linear_all_features
        
        self.alpha = org_model.alpha  # do not quantize for now
        self.gate_fn = org_model.gate_fn
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, x):
        hidden_states = x
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = scaled_dot_product_attention(
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
            x = attn @ v

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


class QuantizedLayerScale(QuantizedModel):
    def __init__(
            self,
            org_model, **quant_params
    ) -> None:
        super().__init__()
        self.inplace = org_model.inplace
        self.gamma = nn.Parameter(org_model.init_values * torch.ones(org_model.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

class QuantizedBlock(QuantizedModel):
    def __init__(
            self, org_model, **quant_params
    ) -> None:
        super().__init__()
        self.norm1 = quantize_model(org_model.norm1, **quant_params)
        self.attn = QuantizedViTSelfAttentionWithExtras(org_model.attn, **quant_params)
        self.ls1 = quantize_model(org_model.ls1, **quant_params)
        self.drop_path1 = org_model.drop_path1
        self.norm2 = quantize_model(org_model.norm2, **quant_params)
        self.mlp = quantize_model(org_model.mlp, **quant_params)
        self.ls2 = org_model.ls2
        self.drop_path2 = org_model.drop_path2
        
        self.res_act_quantizer_1 = QuantizedActivation(**quant_params)
        self.res_act_quantizer_2 = QuantizedActivation(**quant_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = self.res_act_quantizer_1(x)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = self.res_act_quantizer_2(x)
        return x







class QuantizedVisionTransformer(QuantizedModel):
    def __init__(
            self,
            org_model,
            **quant_params
    ) -> None:
        super().__init__()
        assert org_model.global_pool in ('', 'avg', 'token', 'map')
        assert org_model.has_class_token or org_model.global_pool != 'token'
        # use_fc_norm = org_model.global_pool == 'avg' if org_model.fc_norm is None else org_model.fc_norm
        # norm_layer = quantize_model(org_model.norm_layer, **quant_params)
        # act_layer = quantize_model(org_model.act_layer, **quant_params)
        
        self.num_classes = org_model.num_classes
        self.global_pool = org_model.global_pool
        self.num_features = org_model.embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if org_model.class_token else 0
        self.num_prefix_tokens += org_model.num_reg_tokens
        self.num_reg_tokens = org_model.num_reg_tokens
        self.has_class_token = org_model.has_class_token
        self.no_embed_class = org_model.no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = org_model.dynamic_img_size
        self.grad_checkpointing = False

        # embed_args = {}
        # if self.dynamic_img_size:
        #     # flatten deferred until after pos embed
        #     embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = quantize_model(org_model.embed_layer, **quant_params)
        num_patches = self.patch_embed.num_patches

        self.cls_token = org_model.cls_token 
        self.reg_token = org_model.reg_token
        # embed_len = num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = org_model.pos_embed
        self.pos_drop = org_model.pos_drop
        # if org_model.patch_drop_rate > 0:
        #     self.patch_drop = quantize_model(org_model.patch_drop, **quant_params)
        # else:
        #     self.patch_drop = nn.Identity()
        self.patch_drop = org_model.patch_drop
            
        self.norm_pre = quantize_model(org_model.norm_pre, **quant_params)

        # dpr = [x.item() for x in torch.linspace(0, org_model.drop_path_rate, org_model.depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            QuantizedBlock(org_model.blocks[i], **quant_params)
            for i in range(org_model.depth)])
        
        # self.norm = norm_layer(org_model.embed_dim) if not use_fc_norm else nn.Identity()
        self.norm  = quantize_model(org_model.norm, **quant_params)
        
        # Classifier Head
        if org_model.global_pool == 'map':
            # self.attn_pool = QuantizedAttentionPoolLatent(
            #     self.embed_dim,
            #     norm_layer=norm_layer,
            #     org_model=org_model.attn_pool,
            #     **quant_params
            # )
            self.attn_pool = quantize_model(org_model.attn_pool, **quant_params)
        else:
            self.attn_pool = None
            
            
        self.fc_norm = quantize_model(org_model.fc_norm, **quant_params)
        self.head_drop = nn.Dropout(org_model.drop_rate)
        self.head = quantize_model(org_model.head, **quant_params) if org_model.num_classes > 0 else nn.Identity()

        if org_model.weight_init != 'skip':
            self.init_weights(org_model.weight_init)

    def init_weights(self, mode: Literal['jax', 'jax_nlhb', 'moco', ''] = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    








class ViTSelfAttentionWithExtras(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            org_model,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            softmax_fn=torch.nn.functional.softmax,
            alpha=None,
            ssm_eps=None,
            tau=None,
            max_seq_length=None,
            skip_attn=False,
            attn_gate_type=AttentionGateType.none,
            attn_gate_init=None,
            attn_gate_mlp=False,
            attn_gate_mlp2=False,
            attn_gate_linear_all_features=False,
            fine_tuning=False,
            **quant_params
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = quantize_model(org_model.qkv, **quant_params)
        self.q_norm = quantize_model(org_model.q_norm, **quant_params) if qk_norm else nn.Identity()
        self.k_norm = quantize_model(org_model.k_norm, **quant_params) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = quantize_model(org_model.proj, **quant_params)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_scores = nn.Identity()  # before attention mask
        self.attn_probs_before_dropout = nn.Identity()
        self.attn_probs_after_dropout = nn.Identity()

        self.alpha = alpha
        self.ssm_eps = ssm_eps
        self.tau = tau
        self.max_seq_length = max_seq_length

        # define softmax function
        if self.alpha is not None:
            assert self.max_seq_length is not None
            gamma = -self.alpha / self.max_seq_length
            self.softmax_fn = partial(clipped_softmax, gamma=gamma, eta=1.0)
        else:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = scaled_dot_product_attention(
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
            x = attn @ v
        # *** Gating ***
        if self.attn_gate_type == AttentionGateType.unconditional_per_head:
            gate = self.gate_fn(self.alpha)  # (H,)
            context_layer *= gate.view(-1, 1, 1)  # (B, H, T, d_head)

            self.last_gate_avg_prob = gate.view(-1)

        elif self.attn_gate_type in (
            AttentionGateType.conditional_per_head,
            AttentionGateType.conditional_per_token,
        ):
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


        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x