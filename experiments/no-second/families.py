"""Second and third neural-operator families on the FNO arm's exact contract.

Both families consume the same feature tensor the FNO arm consumes
(`model.features`: normalised supplied field, normalised viscosity broadcast, x, y)
and emit the same five evolved-field channels that `model.predict` then masks to the
zero-Dirichlet boundary and prepends the supplied initial state to. Nothing about
the input/output contract, the boundary mask, the normalisation or the metric
differs between families; only the network between `features` and the mask does.

Precision. Each family carries a declared parameter dtype (`float32` or
`float64`). The wrapper `Precision` receives the float64 feature tensor, casts it
to the parameter dtype at the network entry and casts the network output back to
float64 at the exit, so everything outside the network — features, mask,
trajectory assembly, the loss and every reported error — is float64 exactly as in
the FNO arm. With `float64` declared, the network itself is float64 too and the
family is precision-identical to the FNO arm. TF32 is disabled by `model.configure`
regardless, so a float32 network uses genuine IEEE single-precision arithmetic.

U-Net. The PDEBench 2D baseline: a four-level encoder–decoder with two 3×3
convolutions per level, channel doubling per level from `base` to `16*base` at the
bottleneck, 2×2 max-pooling down, 2×2 transposed-convolution up with skip
concatenation, GroupNorm(8) and GELU. The 257×257 nodal grid is zero-padded to
272×272 (divisible by 16) at the entry and cropped back at the exit; zero padding
is the natural extension of a zero-Dirichlet field.

Transolver. Physics attention on a structured 2D mesh (Wu et al., ICML 2024): each
layer projects every token to `slices` learned slice weights, aggregates the tokens
into `slices` physics tokens per head, runs ordinary attention among those
tokens and broadcasts the result back through the same slice weights, followed by
a pre-norm MLP. Tokens here are `patch`×`patch` blocks of the padded grid
(257 → 260 for patch 4) so the token count is in the regime the paper uses;
the position of every token enters as the paper's "unified" positional
encoding (distances to an `ref`×`ref` reference grid). The per-token decoder is a
linear map to `patch*patch*5` values unpatched back onto the grid and cropped.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FAMILIES = ('unet', 'transolver')
DTYPES = {'float32': torch.float32, 'float64': torch.float64}


class Precision(nn.Module):
    """Cast f64 features to the network dtype and the output back to f64."""

    def __init__(self, network, dtype):
        super().__init__()
        self.network = network
        self.parameter_dtype = dtype
        self.network.to(dtype=dtype)

    def forward(self, x):
        assert x.dtype == torch.float64
        return self.network(x.to(self.parameter_dtype)).to(torch.float64)


def pad_to_multiple(x, multiple):
    height, width = x.shape[-2:]
    ph = (-height) % multiple
    pw = (-width) % multiple
    left, top = pw // 2, ph // 2
    padded = F.pad(x, (left, pw - left, top, ph - top))
    return padded, (top, left, height, width)


def crop(x, box):
    top, left, height, width = box
    return x[..., top:top + height, left:left + width]


# ----------------------------------------------------------------------------- U-Net


def conv_block(cin, cout, groups):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.GELU())


class UNet2d(nn.Module):
    def __init__(self, cin, cout, base, groups=8, levels=4):
        super().__init__()
        assert levels == 4, 'the PDEBench baseline is four levels deep'
        f = base
        self.enc = nn.ModuleList([conv_block(cin, f, groups), conv_block(f, 2 * f, groups),
                                  conv_block(2 * f, 4 * f, groups), conv_block(4 * f, 8 * f, groups)])
        self.bottleneck = conv_block(8 * f, 16 * f, groups)
        self.up = nn.ModuleList([nn.ConvTranspose2d(16 * f, 8 * f, 2, stride=2), nn.ConvTranspose2d(8 * f, 4 * f, 2, stride=2),
                                 nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2), nn.ConvTranspose2d(2 * f, f, 2, stride=2)])
        self.dec = nn.ModuleList([conv_block(16 * f, 8 * f, groups), conv_block(8 * f, 4 * f, groups),
                                  conv_block(4 * f, 2 * f, groups), conv_block(2 * f, f, groups)])
        self.head = nn.Conv2d(f, cout, 1)
        self.multiple = 2 ** levels

    def forward(self, x):
        x, box = pad_to_multiple(x, self.multiple)
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            x = block(torch.cat((up(x), skip), dim=1))
        return crop(self.head(x), box)


# ------------------------------------------------------------------------ Transolver


class PhysicsAttention(nn.Module):
    """Physics attention on a structured 2D token grid (Transolver, structured-mesh variant)."""

    def __init__(self, dim, heads, slices, kernel=3):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dim_head = heads, dim // heads
        self.scale = self.dim_head ** -0.5
        self.in_project_x = nn.Conv2d(dim, dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(self.dim_head, slices)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

    def heads_first(self, x):  # B C H W -> B heads N dim_head
        b, c, h, w = x.shape
        return x.reshape(b, self.heads, self.dim_head, h * w).transpose(2, 3)

    def forward(self, x):  # B C H W
        b, c, h, w = x.shape
        fx_mid = self.heads_first(self.in_project_fx(x))
        x_mid = self.heads_first(self.in_project_x(x))
        weights = torch.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, 0.01, 5), dim=-1)  # B h N G
        norm = weights.sum(2)  # B h G
        token = torch.einsum('bhnc,bhng->bhgc', fx_mid, weights) / (norm + 1e-5)[..., None]
        q, k, v = self.to_q(token), self.to_k(token), self.to_v(token)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out_token = attn @ v  # B h G d
        out = torch.einsum('bhgc,bhng->bhnc', out_token, weights)  # B h N d
        out = out.transpose(2, 3).reshape(b, c, h, w).permute(0, 2, 3, 1)  # B H W C
        return self.to_out(out).permute(0, 3, 1, 2)


class TransolverBlock(nn.Module):
    def __init__(self, dim, heads, slices, mlp_ratio):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, heads, slices)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):  # B C H W
        y = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(y)
        y = self.ln2(x.permute(0, 2, 3, 1))
        return x + self.mlp(y).permute(0, 3, 1, 2)


class Transolver2d(nn.Module):
    def __init__(self, cin, cout, dim, layers, heads, slices, mlp_ratio, patch, ref):
        super().__init__()
        # The last two input channels are the x and y coordinate channels built by
        # `model.features`; they become the unified positional encoding.
        self.patch, self.ref, self.cout = patch, ref, cout
        fun_dim = (cin - 2) * patch * patch
        self.preprocess = nn.Sequential(nn.Linear(fun_dim + ref * ref, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.placeholder = nn.Parameter(torch.rand(dim) / dim)
        self.blocks = nn.ModuleList([TransolverBlock(dim, heads, slices, mlp_ratio) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.decoder = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, cout * patch * patch))
        self.register_buffer('ref_grid', torch.stack(torch.meshgrid(
            torch.linspace(0, 1, ref), torch.linspace(0, 1, ref), indexing='ij')).reshape(2, -1).T, persistent=False)

    def forward(self, x):  # B C H W, f64/f32 as cast by Precision
        b = x.shape[0]
        fields, coords = x[:, :-2], x[:, -2:]
        fields, box = pad_to_multiple(fields, self.patch)
        p = self.patch
        hh, ww = fields.shape[-2] // p, fields.shape[-1] // p
        tokens = fields.reshape(b, -1, hh, p, ww, p).permute(0, 2, 4, 1, 3, 5).reshape(b, hh, ww, -1)
        # Token centre coordinates on the unit square, computed analytically from the
        # padding box (the coordinate channels are linspace(0, 1) on the unpadded grid,
        # so a padded position extrapolates the same spacing), then the unified
        # positional encoding: distance from the token centre to every reference point.
        top, left, height, width = box
        rows = (torch.arange(hh, device=x.device, dtype=x.dtype) * p + (p - 1) / 2 - top) / (height - 1)
        cols = (torch.arange(ww, device=x.device, dtype=x.dtype) * p + (p - 1) / 2 - left) / (width - 1)
        centres = torch.stack(torch.meshgrid(rows, cols, indexing='ij'), dim=-1)  # hh ww 2
        pos = torch.sqrt(((centres[..., None, :] - self.ref_grid.to(x.dtype)) ** 2).sum(-1))  # hh ww ref*ref
        pos = pos[None].expand(b, -1, -1, -1)
        del coords
        z = self.preprocess(torch.cat((tokens, pos), dim=-1)) + self.placeholder
        z = z.permute(0, 3, 1, 2)  # B dim hh ww
        for block in self.blocks:
            z = block(z)
        out = self.decoder(self.ln(z.permute(0, 2, 3, 1)))  # B hh ww cout*p*p
        out = out.reshape(b, hh, ww, self.cout, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, self.cout, hh * p, ww * p)
        return crop(out, box)


# -------------------------------------------------------------------------- factory


def make(family, cin, cout, config):
    dtype = DTYPES[config.get('dtype', 'float32')]
    if family == 'unet':
        network = UNet2d(cin, cout, base=config['base'], groups=config.get('groups', 8))
    elif family == 'transolver':
        network = Transolver2d(cin, cout, dim=config['dim'], layers=config['layers'], heads=config['heads'],
                               slices=config['slices'], mlp_ratio=config.get('mlp_ratio', 2),
                               patch=config['patch'], ref=config.get('ref', 8))
    else:
        raise ValueError(family)
    return Precision(network, dtype)


def parameter_count(module):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in module.parameters())


if __name__ == '__main__':
    import json
    import sys
    for family, grid in (('unet', [dict(base=b) for b in (24, 32, 48)]),
                         ('transolver', [dict(dim=d, layers=8, heads=8, slices=64, patch=4) for d in (128, 192, 256)])):
        for config in grid:
            net = make(family, 4, 5, config)
            print(json.dumps(dict(family=family, config=config, real_parameters=parameter_count(net))))
