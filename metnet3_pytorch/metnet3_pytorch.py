from functools import partial

import torch
from torch import nn, Tensor, einsum
import torch.distributed as dist
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# loss scaling in section 4.3.2

class LossScaleFunction(Function):
    @staticmethod
    def forward(ctx, x):
        assert x.ndim == 4
        return x

    @staticmethod
    def backward(ctx, grads):
        num_channels = grads.shape[1]

        weight = 1. / grads.norm(p = 2, keepdim = True, dim = (-1, -2))
        l1_normed_weight = weight / weight.sum(keepdim = True, dim = 1)

        scaled_grads = num_channels * l1_normed_weight * grads

        return scaled_grads

class LossScaler(Module):
    def forward(self, x):
        return LossScaleFunction.apply(x)

# conditionable resnet block

class Block(Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.ReLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = None

        if exists(time_emb_dim):
            self.mlp = Sequential(
                nn.ReLU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None

        assert not (exists(self.mlp) ^ exists(time_emb))

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

# multi-headed rms normalization, for query / key normalized attention

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        *,
        heads
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# MBConv

class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 32,
        dim_head = 32,
        dropout = 0.,
        window_size = 8,
        num_registers = 1
    ):
        super().__init__()
        assert num_registers > 0
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        dim_inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.q_norm = RMSNorm(dim_head, heads = heads)
        self.k_norm = RMSNorm(dim_head, heads = heads)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        num_rel_pos_bias = (2 * window_size - 1) ** 2

        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        rel_pos_indices = F.pad(rel_pos_indices, (num_registers, 0, num_registers, 0), value = num_rel_pos_bias)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        device, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q, k = self.q_norm(q), self.k_norm(k)

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads out

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MaxViT(Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads = 32,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 8,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3,
        num_register_tokens = 4
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'
        assert num_register_tokens > 0

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # window size

        self.window_size = window_size

        self.register_tokens = nn.ParameterList([])

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample = is_first,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                )

                block_attn = Attention(dim = layer_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)

                grid_attn = Attention(dim = layer_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)                

                register_tokens = nn.Parameter(torch.randn(num_register_tokens, layer_dim))

                self.layers.append(ModuleList([
                    conv,
                    block_attn,
                    grid_attn
                ]))

                self.register_tokens.append(register_tokens)

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        b, w = x.shape[0], self.window_size

        x = self.conv_stem(x)

        for (conv, block_attn, grid_attn), register_tokens in zip(self.layers, self.register_tokens):
            x = conv(x)

            # block-like attention

            x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)

            # prepare register tokens

            r = repeat(register_tokens, 'n d -> b x y n d', b = b, x = x.shape[1],y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = block_attn(x) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')

            r = unpack_one(r, register_batch_ps, '* n d')

            # grid-like attention

            x = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)

            # prepare register tokens

            r = reduce(r, 'b x y n d -> b n d', 'mean')
            r = repeat(r, 'b n d -> b x y n d', x = x.shape[1], y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = grid_attn(x) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (w1 x) (w2 y)')

        return self.mlp_head(x)

# main MetNet3 module

class MetNet3(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
