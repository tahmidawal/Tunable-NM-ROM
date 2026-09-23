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

Note the padding is applied to the *normalised* features, so the padded ring carries
the value 0 rather than the normalised boundary value; it is cropped away at the exit
and the mask restores the exact boundary, so it is a fixed ring the networks learn
around, not a correctness issue.

DeepONet. The branch/trunk operator of Lu et al. (Nat. Mach. Intell. 2021), in the
2D analogue of the form this project's 3D lanes already use (`deeponet3d` in
`paper-b3d/operators/extra_models3d.py`): a convolutional branch over the supplied
field and parameter channels -- `levels` levels of two 3x3 convolutions with GELU
followed by 2x2 average pooling, then an exact adaptive average pool to
`pool_bins` x `pool_bins`, a GELU hidden layer and a linear read to
`cout * rank` coefficients -- and an MLP trunk over the coordinate channels with
sinusoidal features at `trunk_frequencies`, `tanh` activations and a linear read to
`rank` basis functions. The output is the rank-`rank` contraction
`u_c(y) = sum_k B_{ck} T_k(y) / sqrt(rank) + bias_c`, exactly as in 3D. The trunk is
evaluated once per forward on the coordinate channels of the first batch element:
`model.features` builds those channels from the field shape alone, so they are
bitwise identical across a batch (asserted in `smoke_second.py`).

U-Net. The PDEBench 2D baseline topology (GroupNorm and GELU in place of PDEBench's
BatchNorm and tanh): a four-level encoder–decoder with two 3×3
convolutions per level, channel doubling per level from `base` to `16*base` at the
bottleneck, 2×2 max-pooling down, 2×2 transposed-convolution up with skip
concatenation, GroupNorm(8) and GELU. The 257×257 nodal grid is zero-padded to
272×272 (divisible by 16) at the entry and cropped back at the exit; zero padding
is the natural extension of a zero-Dirichlet field.

Transolver. Physics attention (Wu et al., ICML 2024), the structured-mesh layer with
3×3 convolutional projections, applied to a *patchified* grid: each
layer projects every token to `slices` learned slice weights, aggregates the tokens
into `slices` physics tokens per head, runs ordinary attention among those
tokens and broadcasts the result back through the same slice weights, followed by
a pre-norm MLP. Tokens here are `patch`×`patch` blocks of the padded grid
(257 → 260 for patch 4) so the token count is in the regime the paper uses;
the position of every token enters as the paper's "unified" positional
encoding (distances to an `ref`×`ref` reference grid). The per-token decoder is the
upstream LayerNorm + one linear map, here to `patch*patch*5` values unpatched back
onto the grid and cropped. Upstream tokens are single grid nodes; patchifying is this
lane's change to keep the token count in the paper's regime at 257².
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FAMILIES = ('unet', 'transolver', 'deeponet')
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
        weights = torch.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, 0.1, 5), dim=-1)  # B h N G
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
        self.decoder = nn.Linear(dim, cout * patch * patch)  # upstream: LayerNorm then one Linear
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


# -------------------------------------------------------------------------- DeepONet


class DeepONet2d(nn.Module):
    """Branch/trunk DeepONet; the 2D analogue of this project's `deeponet3d`."""

    def __init__(self, cin, cout, width, rank, trunk_width, levels=3, pool_bins=4,
                 frequencies=(1., 2., 4.), trunk_layers=3, trunk='mlp', nodes=None):
        super().__init__()
        assert trunk_width >= rank or trunk == 'pod', 'a declared rank needs trunk_width >= rank'
        assert trunk_layers >= 2, 'the trunk needs an input layer and a read-out'
        self.cout, self.rank, self.pool_bins, self.trunk_kind = cout, rank, pool_bins, trunk
        self.frequencies = tuple(float(f) for f in frequencies)
        blocks, ci = [], cin - 2  # the last two channels are the x, y coordinates
        for level in range(levels):
            co = width * 2 ** level
            blocks.append(nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
                                        nn.Conv2d(co, co, 3, padding=1), nn.GELU()))
            ci = co
        self.branch = nn.ModuleList(blocks)
        self.branch_hidden = nn.Linear(ci * pool_bins * pool_bins, trunk_width)
        self.branch_read = nn.Linear(trunk_width, rank * cout)
        features = 2 * (1 + 2 * len(self.frequencies))
        if trunk == 'pod':
            # POD-DeepONet (Lu et al., CMAME 2022): the trunk is NOT learned. It is the POD
            # basis of the TRAINING output fields, held fixed, and the branch learns the
            # coefficients. `train.py` fills these buffers from the training prefix only; they
            # are part of the state dict, so a checkpoint rebuilds without the training data.
            assert nodes, 'a POD trunk needs the node count of the output field'
            self.trunk = nn.ModuleList()
            self.register_buffer('modes', torch.zeros(cout, rank, nodes))
            self.register_buffer('mean_field', torch.zeros(cout, nodes))
        else:
            # `trunk_layers` counts every linear map: input, hidden..., read-out. 3 is the
            # inherited depth and the 3D lane's, so the default changes nothing.
            self.trunk = nn.ModuleList([nn.Linear(features, trunk_width)]
                                       + [nn.Linear(trunk_width, trunk_width) for _ in range(trunk_layers - 2)]
                                       + [nn.Linear(trunk_width, rank)])
        self.bias = nn.Parameter(torch.zeros(cout))
        # The 3D lane scales the read-out down at initialisation; keep that.
        with torch.no_grad():
            self.branch_read.weight.mul_(0.1)
            self.branch_read.bias.zero_()  # 3D `_dense(scale=.1)` has a zero bias

    def forward(self, x):  # B C H W
        b, _, h, w = x.shape
        z = x[:, :-2]
        for block in self.branch:
            z = F.avg_pool2d(block(z), 2, ceil_mode=True)
        z = F.adaptive_avg_pool2d(z, self.pool_bins).reshape(b, -1)
        coefficients = self.branch_read(F.gelu(self.branch_hidden(z))).reshape(b, self.cout, self.rank)
        if self.trunk_kind == 'pod':
            # u_c(y) = sum_k B_ck phi_ck(y) + mu_c(y): orthonormal modes, so no 1/sqrt(rank)
            # rescaling, and the zero-initialised read-out starts the model at the training mean.
            out = torch.einsum('bck,ckn->bcn', coefficients, self.modes) + self.mean_field[None]
            return out.reshape(b, self.cout, h, w) + self.bias[None, :, None, None]
        # Coordinate channels are batch-invariant by construction in `model.features`.
        # `model.features` builds the coordinate channels on [0, 1]; the 3D lane's trunk
        # consumes coordinates on [-1, 1], so map them before the sinusoidal features or
        # every declared frequency would cover half its period (audit finding M1).
        coords = x[0, -2:].permute(1, 2, 0).reshape(-1, 2) * 2 - 1
        trunk = [coords]
        for frequency in self.frequencies:
            trunk += [torch.sin(math.pi * frequency * coords), torch.cos(math.pi * frequency * coords)]
        trunk = torch.cat(trunk, dim=-1)
        for layer in self.trunk[:-1]:
            trunk = torch.tanh(layer(trunk))
        trunk = self.trunk[-1](trunk)  # N rank
        out = coefficients @ trunk.transpose(0, 1) / math.sqrt(self.rank)  # B cout N
        return out.reshape(b, self.cout, h, w) + self.bias[None, :, None, None]


# -------------------------------------------------------------------------- factory


def make(family, cin, cout, config):
    dtype = DTYPES[config.get('dtype', 'float32')]
    if family == 'unet':
        network = UNet2d(cin, cout, base=config['base'], groups=config.get('groups', 8))
    elif family == 'deeponet':
        intervals = config.get('mesh_intervals')
        network = DeepONet2d(cin, cout, width=config['width'], rank=config['rank'],
                             trunk_width=config['trunk_width'], levels=config.get('levels', 3),
                             pool_bins=config.get('pool_bins', 4),
                             frequencies=config.get('trunk_frequencies', (1., 2., 4.)),
                             trunk_layers=config.get('trunk_layers', 3),
                             trunk=config.get('trunk', 'mlp'),
                             nodes=(intervals + 1) ** 2 if intervals else None)
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
                         ('transolver', [dict(dim=d, layers=8, heads=8, slices=64, patch=4) for d in (128, 192, 256)]),
                         ('deeponet', [dict(width=wd, rank=rk, trunk_width=tw)
                                       for wd, rk, tw in ((48, 256, 384), (64, 384, 512), (96, 512, 768))])):
        for config in grid:
            net = make(family, 4, 5, config)
            print(json.dumps(dict(family=family, config=config, real_parameters=parameter_count(net))))
