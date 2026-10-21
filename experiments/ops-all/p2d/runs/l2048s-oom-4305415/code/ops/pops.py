"""Poisson 2D (square and L-shape) adapter for the operator networks (FNO, U-Net, Transolver, DeepONet).

Adapted from exp/2026-09-23-heat-compare-hires @ 9a5baf9a `experiments/heat-compare-hires/ops/heatops.py`
(itself from exp/2026-09-22-ops-tune-grid @ 52d1b573 `model.py`). `families.py` and
`spectral_conv_f64.py` are byte-identical copies (COPIED-FROM.json).

Contract (steady Poisson, -lap u = f, zero Dirichlet, 5-point stencil, (n+1)^2 nodal grid):
* input features: normalised nodal source, x, y coordinate channels on [0, 1] -> cin = 3;
* output: one channel, the solution, multiplied by the domain mask (zero on the boundary and, for the
  L-shape, on the removed quadrant [1/2,1]^2) -> cout = 1. The geometry is fixed, so the mask is a
  hard output constraint and not an input;
* loss: mean squared relative L2 error per case (the Table 1 metric is the worst relative L2).

Families and data (the NM-ROM training families, same seeds):
* square: `core.source_params(0, 3072)` = ms_parametric Gaussian draws, fit split of the NM-ROM
  (`fit_validation_split(3072, 20260916, 0.15)`); nodal source a*exp(-|x-c|^2/(2w^2)) on the interior;
  exact same-grid solution by the orthonormal DST-I (identical to pbk_solve.reference), computed on
  the GPU per batch;
* L-shape: `lsh_core.cohort(0, 4608, 3072)` (in-domain draws), same fit split; solution by batched
  matrix-free CG on the masked grid (the lsh_core.make_gpu_cg operator) to a tight tolerance,
  precomputed once per job (recursive relative residual 1e-9, true residual <= 2e-9 on every case, checked).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv

from spectral_conv_f64 import SpectralConvF64, UPSTREAM_SHA256
import families

CIN, COUT = 3, 1


def configure():
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
        raise RuntimeError('JAX_DEFAULT_MATMUL_PRECISION=highest is mandatory')
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA backend is mandatory')
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


# ------------------------------------------------------------------ families (copied draws)
def square_params(seed, count):
    """ms_parametric.sample_params columns (cx, cy, w, a) == multiresolution-poisson core.source_params."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, count)
    cy = rng.uniform(0.15, 0.85, count)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), count))
    a = rng.uniform(0.5, 2.0, count)
    return np.column_stack((cx, cy, w, a))


def lshape_cohort(seed, draw_count, count):
    """lsh_core.cohort: the same draw, keeping only centres inside Omega."""
    d = square_params(seed, draw_count)
    keep = ~((d[:, 0] >= 0.5) & (d[:, 1] >= 0.5))
    acc = d[keep]
    assert len(acc) >= count
    return acc[:count]


def fit_validation_split(count, seed, validation_fraction):
    order = np.random.default_rng(seed).permutation(count)
    nval = max(1, int(round(validation_fraction * count)))
    return np.sort(order[nval:]), np.sort(order[:nval])


def training_draws(problem):
    """(fit, validation) draws of the NM-ROM of that Table 1 row."""
    d = square_params(0, 3072) if problem == 'square' else lshape_cohort(0, 4608, 3072)
    fit, val = fit_validation_split(len(d), 20260916, 0.15)
    return d[fit], d[val]


def evaluation_draws(problem):
    if problem == 'square':
        return np.concatenate((square_params(7090703, 6), square_params(7090732, 6)))
    return lshape_cohort(20260917, 64, 32)


def domain_mask(problem, n, device='cuda'):
    m = torch.zeros((n + 1, n + 1), dtype=torch.float64, device=device)
    m[1:n, 1:n] = 1.0
    if problem == 'lshape':
        m[n // 2:, n // 2:] = 0.0
    return m


def sources(problem, n, draws, device='cuda'):
    """[B, n+1, n+1] f64 nodal sources, zero off the interior of the domain."""
    x = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64, device=device)
    d = torch.as_tensor(np.asarray(draws), dtype=torch.float64, device=device)
    gx = torch.exp(-(x[None, :] - d[:, 0, None]) ** 2 / (2 * d[:, 2, None] ** 2))
    gy = torch.exp(-(x[None, :] - d[:, 1, None]) ** 2 / (2 * d[:, 2, None] ** 2))
    f = d[:, 3, None, None] * gx[:, :, None] * gy[:, None, :]
    return f * domain_mask(problem, n, device)


def dst1(u):
    """Orthonormal DST-I along the last axis (self-inverse)."""
    n = u.shape[-1] + 1
    z = torch.zeros(u.shape[:-1] + (1,), dtype=u.dtype, device=u.device)
    odd = torch.cat((z, u, z, -u.flip(-1)), -1)
    return -torch.fft.rfft(odd, dim=-1).imag[..., 1:n] / math.sqrt(2.0 * n)


def square_solve(f):
    """Exact same-grid 5-point solution of the square, [B, n+1, n+1] -> [B, n+1, n+1]."""
    n = f.shape[-1] - 1
    p = torch.arange(1, n, dtype=torch.float64, device=f.device)
    lam1 = 4.0 * n * n * torch.sin(math.pi * p / (2 * n)) ** 2
    lam = lam1[:, None] + lam1[None, :]
    c = dst1(dst1(f[:, 1:-1, 1:-1]).transpose(-1, -2)).transpose(-1, -2)
    u = dst1(dst1(c / lam).transpose(-1, -2)).transpose(-1, -2)
    return torch.nn.functional.pad(u, (1, 1, 1, 1))


def lshape_cg(f, rtol=1e-9, maxiter=400000):
    """Batched matrix-free CG on the masked grid (lsh_core.make_gpu_cg operator), per-case stopping."""
    n = f.shape[-1] - 1
    mask = domain_mask('lshape', n, f.device).bool()
    h2 = float(n * n)

    def A(u):
        lap = 4.0 * u - (torch.roll(u, 1, -2) + torch.roll(u, -1, -2) + torch.roll(u, 1, -1) + torch.roll(u, -1, -1))
        return torch.where(mask, lap * h2, 0.0)

    b = torch.where(mask, f, 0.0)
    bn = b.flatten(1).norm(dim=1)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).flatten(1).sum(1)
    it = 0
    while it < maxiter:
        done = rs.sqrt() <= rtol * bn
        if bool(done.all()):
            break
        Ap = A(p)
        alpha = torch.where(done, 0.0, rs / (p * Ap).flatten(1).sum(1))
        x = x + alpha[:, None, None] * p
        r = r - alpha[:, None, None] * Ap
        rs2 = (r * r).flatten(1).sum(1)
        beta = torch.where(done, 0.0, rs2 / rs)
        p = r + beta[:, None, None] * p
        rs = torch.where(done, rs, rs2)
        it += 1
    res = (b - A(x)).flatten(1).norm(dim=1) / bn
    return x, it, res


def solve(problem, f):
    if problem == 'square':
        return square_solve(f)
    x, it, res = lshape_cg(f)
    if not bool((res <= 2e-9).all()):   # true residual; recursive residual stops at 1e-9
        raise RuntimeError(f'L-shape CG not converged: {float(res.max())}')
    return x


# ------------------------------------------------------------------ networks
def to_f64(model):
    return model._apply(lambda x: x.to(dtype=torch.complex128 if x.is_complex() else torch.float64)
                        if x.is_floating_point() or x.is_complex() else x)


def family_of(config):
    return config.get('family', 'fno')


def make_model(config):
    if family_of(config) != 'fno':
        return families.make(family_of(config), CIN, COUT, config).cuda()
    model = FNO(n_modes=(config['modes'],) * 2, in_channels=CIN, out_channels=COUT,
                hidden_channels=config['width'], n_layers=config['layers'], positional_embedding=None,
                fno_block_precision='full', norm=config.get('norm'), factorization=config.get('factorization'),
                rank=config.get('rank', 1.0), conv_module=SpectralConvF64)
    if config.get('checkpoint_blocks'):
        # activation checkpointing of each FNO block: identical arithmetic, blocks recomputed in backward
        # (memory only; declared in DESIGN A4). Not used at inference (no_grad).
        import torch.utils.checkpoint as tuc
        blocks = model.fno_blocks
        plain = blocks.forward

        def forward(x, index=0, output_shape=None):
            if torch.is_grad_enabled():
                return tuc.checkpoint(plain, x, index, output_shape, use_reentrant=False)
            return plain(x, index, output_shape=output_shape)
        blocks.forward = forward
    return to_f64(model).cuda()


def features(field, mean, std):
    """field [B, 1, H, W] nodal f64 -> [B, 3, H, W]."""
    batch, _, height, width = field.shape
    xx = torch.linspace(0, 1, height, device=field.device, dtype=torch.float64)
    yy = torch.linspace(0, 1, width, device=field.device, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(xx, yy, indexing='ij'))[None].expand(batch, -1, -1, -1)
    return torch.cat(((field - mean) / std, grid), dim=1)


def predict(model, source, mean, std, scale, mask):
    """source [B, H, W] nodal f64 -> solution [B, H, W] nodal f64 (masked)."""
    return model(features(source[:, None], mean, std))[:, 0] * scale * mask


def relative(prediction, target):
    num = (prediction - target).square().flatten(1).sum(1).sqrt()
    return num / target.square().flatten(1).sum(1).sqrt().clamp_min(1e-300)


def check_dtypes(model):
    declared = getattr(model, 'parameter_dtype', torch.float64)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point() or value.is_complex():
            expected = torch.complex128 if value.is_complex() else declared
            if value.dtype != expected:
                raise RuntimeError(f'Parameter/buffer precision failure: {name} {value.dtype}')


def load(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    net = make_model(ck['config'])
    net.load_state_dict(ck['model'])
    check_dtypes(net)
    net.eval()
    norm = tuple(v.cuda() for v in ck['normalization'])
    return net, norm, ck
