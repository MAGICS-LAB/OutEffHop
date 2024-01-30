import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import copy
from layers import *


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(.3)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, mode, activation):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(512, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(512, eps=1e-6)
        self.ffn = MLP()
        if mode == 'MHN':
            self.layer = Hopfield(d_model=512, n_heads=4, scale=None, mode=activation, update_steps=1)
        elif mode == 'UMHN':
            self.layer = LearnableHopfield(d_model=512, n_heads=4, update_steps=1, mode=activation, kernel='lin', scale=None)


    def forward(self, x, get_kernel=False):
        h = x
        x = self.attention_norm(x)
        if get_kernel:
            k = self.layer.uniform_forward(x)
        x = self.layer(x, x)

        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        if get_kernel:
            return x, k
        return x

class ThreeLayerViH(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224, n_heads = 1, scale=None, mode = 'MHN', activation='softmax', update_steps=1, n_class=10):
        self.patch_size = patch_size
        super().__init__()

        self.mode = mode
        self.emb = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(512, eps=1e-6)
        for _ in range(3):
            layer = Block(mode, activation)
            self.layer.append(copy.deepcopy(layer))

        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, n_class)

    def kernel_forward(self, x):
        unif_result = []
        if self.mode == 'UMHN':
            x = self.emb(x)
            for l in self.layer:
                x, u = l(x, True)
                unif_result.append(u)
            return unif_result
        else:
            raise Exception('Only UMHN supports kernel forward')

    def forward(self, x):
        x = self.emb(x)

        for l in self.layer:
            x= l(x)
        x = self.encoder_norm(x)
        query, _ = x[:, 0].unsqueeze(dim=1), x[:, 1:]

        cls = self.ln(self.relu(query)).squeeze(1)
        return self.fc(cls)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
    
    def forward(self, x: Tensor) :
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        return x

class ViH(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224, n_heads = 1, scale=None, mode = 'MHN', activation='softmax', update_steps=1, n_class=10):
        self.patch_size = patch_size
        super().__init__()

        self.mode = mode
        self.emb = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        if mode == 'MHN':
            self.layer = Hopfield(d_model=emb_size, n_heads=n_heads, scale=scale, mode=activation, update_steps=update_steps)
        elif mode == 'UMHN':
            self.layer = LearnableHopfield(d_model=emb_size, n_heads=n_heads, update_steps=update_steps, mode=activation, kernel='lin', scale=scale)

        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, n_class)

    def kernel_forward(self, x):
        if self.mode == 'UMHN':
            x = self.emb(x)
            return self.layer.uniform_forward(x)
        else:
            raise Exception('Only UMHN supports kernel forward')

    def forward(self, x):
        x = self.emb(x)
        query, memory = x[:, 0].unsqueeze(dim=1), x[:, 1:]
        cls = self.ln(self.relu(self.layer(query, memory))).squeeze(1)
        return self.fc(cls)
        
class Model(nn.Module):

    def __init__(
            self,
            d_model,
            n_class,
            n_heads=1,
            dropout=0.1,
            mode='MHN',
            kernel='lin',
            scale=None):
        super(Model, self).__init__()

        if mode == 'MHN':
            self.hopfield = Hopfield(d_model=d_model, n_heads=n_heads)
        
        else:
            self.hopfield = LearnableHopfield(d_model=d_model, n_heads=n_heads, kernel=kernel)

        self.fc = nn.Linear(d_model, n_class)

    
    def forward(self, memory, query):

        out = self.hopfield(memory, query)
        return self.fc(out)