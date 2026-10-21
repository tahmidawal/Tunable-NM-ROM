"""Heat-2D adapter for the Burgers operator panel's networks (FNO, U-Net, Transolver, DeepONet).

Copied/adapted from exp/2026-09-22-ops-tune-grid @ 52d1b573 (`model.py`, `families.py`,
`spectral_conv_f64.py`; the last two are byte-identical copies, see COPIED-FROM.json).

Contract (the same as the Burgers panel except where marked HEAT):
* grid: the (n+1)^2 nodal grid including the zero-Dirichlet boundary; the interior (n-1)^2
  block is what every other subject of this lane produces and is what errors are taken on;
* input features: normalised supplied field, then x and y coordinate channels on [0, 1];
  HEAT: no parameter channel (the diffusivity is fixed, 0.02), so cin = 3;
* output: five evolved fields (t = 0.1..0.5); the supplied state is returned exactly as the
  t = 0 output; the boundary ring is masked to zero;
* HEAT: the loss is the mean squared CURRENT-relative L2 error per output time (the heat
  metric of the paper), not Burgers' initial-relative one.

Data: the family `mr2d` (hires-heat core.family) and the exact same-grid semidiscrete flow,
evaluated separably (the initial field is amp * f(x) f(y) and the heat semigroup of the
5-point Laplacian factorises over axes), so any number of cases can be generated on the GPU
at any mesh without storing a data set.
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

TIMES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
NU = 0.02
CIN, COUT = 3, 5


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


# ------------------------------------------------------------------ family and exact flow
def family(seed, count):
    """hires-heat core.family('mr2d'): rows [cx, cy, width, amplitude]."""
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(.35, .65, (count, 2)), rng.uniform(.10, .15, count), rng.uniform(.8, 1.2, count)))


def dst1(u):
    """Orthonormal DST-I along the last axis (self-inverse); same construction as core.dst_axis."""
    n = u.shape[-1] + 1
    z = torch.zeros(u.shape[:-1] + (1,), dtype=u.dtype, device=u.device)
    odd = torch.cat((z, u, z, -u.flip(-1)), -1)
    return -torch.fft.rfft(odd, dim=-1).imag[..., 1:n] / math.sqrt(2.0 * n)


def axis_factors(n, draws, times=TIMES, nu=NU, device='cuda'):
    """[B, T, 2, n-1] per-axis factors g with u(t)[i, j] = amp * g[.., 0, i] * g[.., 1, j] (f64)."""
    a = torch.arange(1, n, dtype=torch.float64, device=device) / n
    k = torch.arange(1, n, dtype=torch.float64, device=device)
    lam = 4 * n * n * torch.sin(math.pi * k / (2 * n)) ** 2
    d = torch.as_tensor(np.asarray(draws), dtype=torch.float64, device=device)
    f = torch.stack([4 * a * (1 - a) * torch.exp(-(a - d[:, ax, None]) ** 2 / (2 * d[:, 2, None] ** 2)) for ax in range(2)], 1)
    c = dst1(f)                                                    # B 2 n-1
    t = torch.as_tensor(times, dtype=torch.float64, device=device)
    g = dst1(c[:, None] * torch.exp(-nu * t[None, :, None, None] * lam))   # B T 2 n-1
    g[:, 0] = f                                                    # the supplied field is returned exactly at t=0
    return g, d[:, 3]


def interior_fields(n, draws, device='cuda'):
    """[B, T, n-1, n-1] exact same-grid trajectories (f64)."""
    g, amp = axis_factors(n, draws, device=device)
    return amp[:, None, None, None] * g[:, :, 0, :, None] * g[:, :, 1, None, :]


def nodal(interior):
    """Zero-pad the interior (..., n-1, n-1) to the nodal (..., n+1, n+1) grid."""
    return torch.nn.functional.pad(interior, (1, 1, 1, 1))


# ------------------------------------------------------------------ networks
def to_f64(model):
    return model._apply(lambda x: x.to(dtype=torch.complex128 if x.is_complex() else torch.float64)
                        if x.is_floating_point() or x.is_complex() else x)


class F32FNO(torch.nn.Module):
    """g2d ops-all: float32/complex64 FNO; f64 features in, f64 output out (same contract as families.Precision, which
    cannot be used here because Module.to(float32) would drop the imaginary part of the spectral weights)."""

    def __init__(self, network):
        super().__init__()
        self.network = network._apply(lambda x: x.to(dtype=torch.complex64 if x.is_complex() else torch.float32)
                                      if x.is_floating_point() or x.is_complex() else x)
        self.parameter_dtype = torch.float32

    def forward(self, x):
        return self.network(x.to(torch.float32)).to(torch.float64)

    def load_state_dict(self, state, strict=True, **kw):
        # neuralop's FNO.state_dict() adds a non-tensor '_metadata' entry at the top level of the wrapper's dict
        return super().load_state_dict({k: v for k, v in state.items() if k != '_metadata'}, strict=strict, **kw)


def family_of(config):
    return config.get('family', 'fno')


def make_model(config):
    if family_of(config) != 'fno':
        return families.make(family_of(config), CIN, COUT, config).cuda()
    if config.get('dtype') == 'float32':   # g2d ops-all: float32 FNO rung (upstream SpectralConv, complex64), Precision wrapper
        net = FNO(n_modes=(config['modes'],) * 2, in_channels=CIN, out_channels=COUT, hidden_channels=config['width'],
                  n_layers=config['layers'], positional_embedding=None, fno_block_precision='full', norm=config.get('norm'))
        return F32FNO(net).cuda()
    model = FNO(n_modes=(config['modes'],) * 2, in_channels=CIN, out_channels=COUT,
                hidden_channels=config['width'], n_layers=config['layers'], positional_embedding=None,
                fno_block_precision='full', norm=config.get('norm'), factorization=config.get('factorization'),
                rank=config.get('rank', 1.0), conv_module=SpectralConvF64)
    return to_f64(model).cuda()


def features(field, mean, std):
    """field [B, 1, H, W] nodal f64 -> [B, 3, H, W]."""
    batch, _, height, width = field.shape
    xx = torch.linspace(0, 1, height, device=field.device, dtype=torch.float64)
    yy = torch.linspace(0, 1, width, device=field.device, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(xx, yy, indexing='ij'))[None].expand(batch, -1, -1, -1)
    return torch.cat(((field - mean) / std, grid), dim=1)


def predict(model, field, mean, std, scale):
    """field [B, 1, H, W] nodal -> trajectory [B, 6, H, W] nodal (t=0 is the supplied field)."""
    output = model(features(field, mean, std)) * scale
    mask = torch.ones_like(output[:, :1])
    mask[..., 0, :] = 0
    mask[..., -1, :] = 0
    mask[..., :, 0] = 0
    mask[..., :, -1] = 0
    return torch.cat((field, output * mask), dim=1)


def query_interior(model, u0, mean, std, scale):
    """The TIMED query: supplied interior field [n-1, n-1] on the GPU -> [6, n-1, n-1] on the GPU."""
    field = torch.nn.functional.pad(u0, (1, 1, 1, 1))[None, None]
    return predict(model, field, mean, std, scale)[0, :, 1:-1, 1:-1].contiguous()


def current_relative(prediction, target):
    """[B, T] current-relative L2 errors (the heat metric)."""
    num = (prediction - target).square().sum(dim=(-2, -1)).sqrt()
    return num / target.square().sum(dim=(-2, -1)).sqrt().clamp_min(1e-300)


def check_dtypes(model):
    declared = getattr(model, 'parameter_dtype', torch.float64)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point() or value.is_complex():
            expected = (torch.complex64 if declared == torch.float32 else torch.complex128) if value.is_complex() else declared
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
