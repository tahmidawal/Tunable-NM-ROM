"""Four neural-operator families for the linear 3D Table-1 cells (Poisson 3D, Heat 3D), PyTorch.

Copied from experiments/ns3d-operators/ops3d.py (branch exp/2026-09-23-ns3d-operators @ 766c3247) and
changed ONLY where the Dirichlet unit cube differs from the periodic torus (see DESIGN.md):
* in/out channel counts come from the config (Poisson: 4 in -> 1 out; heat: 4 in -> 5 out),
* convolutions use zero padding (Dirichlet boundary) instead of circular padding,
* the DeepONet trunk uses sin/cos(pi f x) (the sine family of the Dirichlet cube) instead of sin/cos(2 pi f x),
* no Leray projector (NS only).
The network sees an n^3 grid whose index i <-> x = i/n (i = 0 is the x = 0 boundary face, value 0),
so every n divisible by 16 works with the four-level U-Net and the 16^3-token Transolver. The problem
adapters (features, cropping to the (n-1)^3 interior, losses) live in problems.py.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 'zeros'   # Dirichlet cube (ns3d-operators used 'circular' on the torus)
DTYPES = {'float32': torch.float32, 'float64': torch.float64}


def configure():
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
        raise RuntimeError('JAX_DEFAULT_MATMUL_PRECISION=highest is mandatory')
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA backend is mandatory')
    from neuralop.layers.spectral_convolution import SpectralConv
    from spectral_conv_f64 import UPSTREAM_SHA256
    torch.set_default_dtype(torch.float64)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual = hashlib.sha256(Path(inspect.getfile(SpectralConv)).read_bytes()).hexdigest()
    if actual != UPSTREAM_SHA256:
        raise RuntimeError(f'SpectralConv source differs from reviewed implementation: {actual}')
    print('torch_backend=cuda', torch.cuda.get_device_name(), flush=True)
    return dict(torch=torch.__version__, neuraloperator=importlib.metadata.version('neuraloperator'),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), spectral_module_sha256=actual)


# ------------------------------------------------------------------- precision wrapper
class Precision(nn.Module):
    """Cast float64 features to the network dtype and the output back to float64."""

    def __init__(self, network, dtype):
        super().__init__()
        self.network = network
        self.parameter_dtype = dtype
        self.network.to(dtype=dtype)

    def forward(self, x):
        assert x.dtype == torch.float64
        return self.network(x.to(self.parameter_dtype)).to(torch.float64)


def to_f64(model):
    return model._apply(lambda x: x.to(dtype=torch.complex128 if x.is_complex() else torch.float64)
                        if x.is_floating_point() or x.is_complex() else x)


# ----------------------------------------------------------------------------- U-Net
def conv_block(cin, cout, groups):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, padding_mode=PAD, bias=False), nn.GroupNorm(groups, cout), nn.GELU(),
        nn.Conv3d(cout, cout, 3, padding=1, padding_mode=PAD, bias=False), nn.GroupNorm(groups, cout), nn.GELU())


class UNet3d(nn.Module):
    """The 2D cells' PDEBench-topology U-Net in 3D: four levels, two 3^3 convolutions per
    level (circular padding: periodic domain), channel doubling to 16*base at the
    bottleneck, 2^3 max-pooling, 2^3 transposed convolution up, skip concatenation,
    GroupNorm(8) + GELU. The grid side must be divisible by 16."""

    def __init__(self, cin, cout, base, groups=8, levels=4):
        super().__init__()
        assert levels == 4
        f = base
        self.enc = nn.ModuleList([conv_block(cin, f, groups), conv_block(f, 2 * f, groups),
                                  conv_block(2 * f, 4 * f, groups), conv_block(4 * f, 8 * f, groups)])
        self.bottleneck = conv_block(8 * f, 16 * f, groups)
        self.up = nn.ModuleList([nn.ConvTranspose3d(16 * f, 8 * f, 2, stride=2), nn.ConvTranspose3d(8 * f, 4 * f, 2, stride=2),
                                 nn.ConvTranspose3d(4 * f, 2 * f, 2, stride=2), nn.ConvTranspose3d(2 * f, f, 2, stride=2)])
        self.dec = nn.ModuleList([conv_block(16 * f, 8 * f, groups), conv_block(8 * f, 4 * f, groups),
                                  conv_block(4 * f, 2 * f, groups), conv_block(2 * f, f, groups)])
        self.head = nn.Conv3d(f, cout, 1)

    def forward(self, x):
        assert all(s % 16 == 0 for s in x.shape[-3:])
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            x = block(torch.cat((up(x), skip), dim=1))
        return self.head(x)


# ------------------------------------------------------------------------ Transolver
class PhysicsAttention3d(nn.Module):
    """Physics attention on a structured 3D token grid (Transolver, structured-mesh layer)."""

    def __init__(self, dim, heads, slices, kernel=3):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dim_head = heads, dim // heads
        self.scale = self.dim_head ** -0.5
        pad = kernel // 2
        self.in_project_x = nn.Conv3d(dim, dim, kernel, 1, pad, padding_mode=PAD)
        self.in_project_fx = nn.Conv3d(dim, dim, kernel, 1, pad, padding_mode=PAD)
        self.in_project_slice = nn.Linear(self.dim_head, slices)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

    def heads_first(self, x):  # B C D H W -> B heads N dim_head
        b, c = x.shape[:2]
        return x.reshape(b, self.heads, self.dim_head, -1).transpose(2, 3)

    def forward(self, x):  # B C D H W
        b, c, d, h, w = x.shape
        fx_mid = self.heads_first(self.in_project_fx(x))
        x_mid = self.heads_first(self.in_project_x(x))
        weights = torch.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, 0.1, 5), dim=-1)
        norm = weights.sum(2)
        token = torch.einsum('bhnc,bhng->bhgc', fx_mid, weights) / (norm + 1e-5)[..., None]
        q, k, v = self.to_q(token), self.to_k(token), self.to_v(token)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = torch.einsum('bhgc,bhng->bhnc', attn @ v, weights)
        out = out.transpose(2, 3).reshape(b, c, d, h, w).permute(0, 2, 3, 4, 1)
        return self.to_out(out).permute(0, 4, 1, 2, 3)


class TransolverBlock3d(nn.Module):
    def __init__(self, dim, heads, slices, mlp_ratio):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = PhysicsAttention3d(dim, heads, slices)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):  # B C D H W
        y = self.ln1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = x + self.attn(y)
        y = self.ln2(x.permute(0, 2, 3, 4, 1))
        return x + self.mlp(y).permute(0, 4, 1, 2, 3)


class Transolver3d(nn.Module):
    """Patchified 3D Transolver. Tokens are patch^3 blocks; each token's position enters
    as the unified positional encoding (distances from the token centre to a ref^3
    reference lattice on the unit cube). Decoder: LayerNorm + one linear map to
    patch^3 * cout values, unpatchified."""

    def __init__(self, cin, cout, dim, layers, heads, slices, mlp_ratio, patch, ref):
        super().__init__()
        self.patch, self.ref, self.cout = patch, ref, cout
        fun_dim = cin * patch ** 3
        self.preprocess = nn.Sequential(nn.Linear(fun_dim + ref ** 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.placeholder = nn.Parameter(torch.rand(dim) / dim)
        self.blocks = nn.ModuleList([TransolverBlock3d(dim, heads, slices, mlp_ratio) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, cout * patch ** 3)
        g = torch.linspace(0, 1, ref)
        self.register_buffer('ref_grid', torch.stack(torch.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T,
                             persistent=False)

    def forward(self, x):  # B C D H W
        b, c, d, h, w = x.shape
        p = self.patch
        assert d % p == 0 and h % p == 0 and w % p == 0
        td, th, tw = d // p, h // p, w // p
        tokens = x.reshape(b, c, td, p, th, p, tw, p).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, td, th, tw, -1)
        axes = [(torch.arange(t, device=x.device, dtype=x.dtype) * p + (p - 1) / 2) / s for t, s in ((td, d), (th, h), (tw, w))]
        centres = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)  # td th tw 3, on [0,1)
        pos = torch.sqrt(((centres[..., None, :] - self.ref_grid.to(x.dtype)) ** 2).sum(-1))
        pos = pos[None].expand(b, -1, -1, -1, -1)
        z = self.preprocess(torch.cat((tokens, pos), dim=-1)) + self.placeholder
        z = z.permute(0, 4, 1, 2, 3)
        for block in self.blocks:
            z = block(z)
        out = self.decoder(self.ln(z.permute(0, 2, 3, 4, 1)))  # B td th tw cout*p^3
        out = out.reshape(b, td, th, tw, self.cout, p, p, p).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return out.reshape(b, self.cout, d, h, w)


# -------------------------------------------------------------------------- DeepONet
class DeepONet3d(nn.Module):
    """Branch/trunk DeepONet (Lu et al. 2021), CNN branch as in the 2D cells and the
    project's earlier deeponet3d: `levels` levels of two 3^3 convolutions (circular
    padding) + GELU and 2^3 average pooling, adaptive average pooling to pool_bins^3,
    a GELU hidden layer and a linear read to cout*rank coefficients. Trunk: tanh MLP
    over periodic coordinate features sin/cos(2 pi f x_i), f in `trunk_frequencies`
    (Dirichlet cube: sine family), linear read to `rank` basis functions.
    Output u_c(y) = sum_k B_ck T_k(y) / sqrt(rank) + bias_c."""

    def __init__(self, cin, cout, width, rank, trunk_width, levels=3, pool_bins=4,
                 frequencies=(1., 2., 3., 4.)):
        super().__init__()
        assert trunk_width >= rank
        self.cout, self.rank, self.pool_bins = cout, rank, pool_bins
        self.frequencies = tuple(float(f) for f in frequencies)
        blocks, ci = [], cin
        for level in range(levels):
            co = width * 2 ** level
            blocks.append(nn.Sequential(nn.Conv3d(ci, co, 3, padding=1, padding_mode=PAD), nn.GELU(),
                                        nn.Conv3d(co, co, 3, padding=1, padding_mode=PAD), nn.GELU()))
            ci = co
        self.branch = nn.ModuleList(blocks)
        self.branch_hidden = nn.Linear(ci * pool_bins ** 3, trunk_width)
        self.branch_read = nn.Linear(trunk_width, rank * cout)
        features = 3 * 2 * len(self.frequencies)
        self.trunk = nn.ModuleList([nn.Linear(features, trunk_width), nn.Linear(trunk_width, trunk_width),
                                    nn.Linear(trunk_width, rank)])
        self.bias = nn.Parameter(torch.zeros(cout))
        with torch.no_grad():
            self.branch_read.weight.mul_(0.1)
            self.branch_read.bias.zero_()

    def trunk_features(self, shape, device, dtype):
        axes = [torch.arange(s, device=device, dtype=dtype) / s for s in shape]
        coords = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)
        feats = []
        for f in self.frequencies:
            feats += [torch.sin(math.pi * f * coords), torch.cos(math.pi * f * coords)]
        return torch.cat(feats, dim=-1)

    def forward(self, x):  # B C D H W
        b, _, d, h, w = x.shape
        z = x
        for block in self.branch:
            z = F.avg_pool3d(block(z), 2, ceil_mode=True)
        z = F.adaptive_avg_pool3d(z, self.pool_bins).reshape(b, -1)
        coefficients = self.branch_read(F.gelu(self.branch_hidden(z))).reshape(b, self.cout, self.rank)
        trunk = self.trunk_features((d, h, w), x.device, x.dtype)
        for layer in self.trunk[:-1]:
            trunk = torch.tanh(layer(trunk))
        trunk = self.trunk[-1](trunk)
        out = coefficients @ trunk.transpose(0, 1) / math.sqrt(self.rank)
        return out.reshape(b, self.cout, d, h, w) + self.bias[None, :, None, None, None]


# -------------------------------------------------------------------------- factory
def family_of(config):
    return config['family']


def make_model(config):
    family = family_of(config)
    if family == 'fno':
        from neuralop.models import FNO
        from spectral_conv_f64 import SpectralConvF64
        model = FNO(n_modes=(config['modes'],) * 3, in_channels=config['cin'], out_channels=config['cout'],
                    hidden_channels=config['width'], n_layers=config['layers'],
                    positional_embedding=None, fno_block_precision='full', norm=config.get('norm'),
                    conv_module=SpectralConvF64)
        return to_f64(model).cuda()
    dtype = DTYPES[config.get('dtype', 'float32')]
    if family == 'unet':
        net = UNet3d(config['cin'], config['cout'], base=config['base'], groups=config.get('groups', 8))
    elif family == 'transolver':
        net = Transolver3d(config['cin'], config['cout'], dim=config['dim'], layers=config['layers'], heads=config['heads'],
                           slices=config['slices'], mlp_ratio=config.get('mlp_ratio', 2),
                           patch=config['patch'], ref=config.get('ref', 8))
    elif family == 'deeponet':
        net = DeepONet3d(config['cin'], config['cout'], width=config['width'], rank=config['rank'], trunk_width=config['trunk_width'],
                         levels=config.get('levels', 3), pool_bins=config.get('pool_bins', 4),
                         frequencies=config.get('trunk_frequencies', (1., 2., 3., 4.)))
    else:
        raise ValueError(family)
    return Precision(net, dtype).cuda()


def check_dtypes(model):
    declared = getattr(model, 'parameter_dtype', torch.float64)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point() or value.is_complex():
            expected = torch.complex128 if value.is_complex() else declared
            if value.dtype != expected:
                raise RuntimeError(f'precision failure: {name} {value.dtype}')


def parameter_count(module):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in module.parameters())


def save_checkpoint(path, model, config, norm, extra):
    torch.save(dict(model=model.state_dict(), config=config, normalization=[float(v) for v in norm], **extra), path)


def load_checkpoint(path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    model = make_model(ck['config'])
    model.load_state_dict(ck['model'])
    model.eval()
    norm = tuple(torch.tensor(v, dtype=torch.float64, device='cuda') for v in ck['normalization'])
    return model, norm, ck
