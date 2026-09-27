"""Poisson 3D and Heat 3D for the operators: the Table-1 generators ported to PyTorch (float64, on the GPU),
feature construction and the per-problem losses/errors.

Generators (ported line by line; smoke_generators.py checks them against the JAX originals):
* Poisson 3D (paper-p3d common.py / poisson.py, the family of poisson-bank-knob-3d):
  draw p = [cx, cy, cz, w, a]  (C.family: rng.random((count,5)) * [.3,.3,.3,.05,.4] + [.35,.35,.35,.10,.8]),
  source f = a * 64 prod x(1-x) * exp(-|x-c|^2 / (2 w^2)) on the interior grid x_i = i/n, i = 1..n-1,
  solution u = DST(DST(f) / lambda) (orthonormal DST-I, lambda = sum_axes 4 n^2 sin^2(pi k / 2n)) — the
  exact solution of the 7-point discrete Poisson problem = the Table-1 same-grid reference.
* Heat 3D (heat-bank-knob core.py, family 'h3d'): the same draw law; u0 = a * prod 4 x(1-x) exp(...);
  u(t) = DST(exp(-nu t lambda) DST(u0)), nu = 0.02, t in {0, .1, .2, .3, .4, .5} — the Table-1 same-grid
  exact time evolution.
Operator contract: input = the query's own input on the GPU ((n-1)^3 float64 interior field: forcing for
Poisson, initial field for heat); output = float64 interior fields ((n-1)^3 for Poisson; 6 x (n-1)^3 for heat,
t = 0 returned exactly as u0). Features: the field / in_scale and the three coordinates x = i/n on the n^3
grid (index 0 = boundary face, field value 0) — the 2D cells' coordinate channels.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

HEAT_TIMES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
HEAT_NU = 0.02
POISSON = dict(train_seed=920410, train_count=512, validation_seed=920411, validation_count=16,
               final_seed=920499, final_count=64)
HEAT = dict(train_seed=921000, train_count=2048, validation_seed=921777, heldout_seed=921099, heldout_count=64)


def family(seed, count):
    a = np.random.default_rng(seed).random((count, 5))
    return a * np.array([.3, .3, .3, .05, .4]) + np.array([.35, .35, .35, .10, .8])


def axis_nodes(n, device='cuda'):
    return torch.arange(1, n, dtype=torch.float64, device=device) / n


def bump(n, draws, mask_scale):
    """(B, n-1, n-1, n-1): amp * prod_axes mask_scale * x(1-x) * exp(-(x-c)^2/(2w^2)) (separable)."""
    a = axis_nodes(n)
    d = torch.as_tensor(np.asarray(draws), dtype=torch.float64, device='cuda')
    fac = [mask_scale * a * (1 - a) * torch.exp(-(a[None] - d[:, ax:ax + 1]) ** 2 / (2 * d[:, 3:4] ** 2))
           for ax in range(3)]
    return d[:, 4, None, None, None] * fac[0][:, :, None, None] * fac[1][:, None, :, None] * fac[2][:, None, None, :]


def dst_axis(u, axis):
    """Orthonormal DST-I along `axis` (port of core.dst_axis: odd extension + rfft)."""
    u = torch.movedim(u, axis, -1)
    n = u.shape[-1] + 1
    z = torch.zeros(u.shape[:-1] + (1,), dtype=u.dtype, device=u.device)
    odd = torch.cat((z, u, z, -torch.flip(u, (-1,))), -1)
    out = -torch.fft.rfft(odd, dim=-1).imag[..., 1:n] / math.sqrt(2.0 * n)
    return torch.movedim(out, -1, axis)


def dst3(u):
    for ax in (-1, -2, -3):
        u = dst_axis(u, ax)
    return u


def eig3(n):
    k = torch.arange(1, n, dtype=torch.float64, device='cuda')
    l = 4 * n * n * torch.sin(math.pi * k / (2 * n)) ** 2
    return l[:, None, None] + l[None, :, None] + l[None, None, :]


class Problem:
    """name in {'poisson','heat'}; batch(draws) -> (input (B,n-1,n-1,n-1), target) on the GPU, float64."""

    def __init__(self, name, n):
        self.name, self.n = name, n
        self.lam = eig3(n)
        if name == 'poisson':
            self.cin, self.cout = 4, 1
        else:
            self.cin, self.cout = 4, len(HEAT_TIMES) - 1
            self.decay = torch.stack([torch.exp(-HEAT_NU * t * self.lam) for t in HEAT_TIMES[1:]])

    def batch(self, draws):
        if self.name == 'poisson':
            f = bump(self.n, draws, 4.0)       # 64 prod x(1-x) = prod (4 x(1-x))
            return f, dst3(dst3(f) / self.lam)
        u0 = bump(self.n, draws, 4.0)
        c = dst3(u0)
        later = torch.stack([dst3(c * self.decay[j]) for j in range(self.cout)], dim=1)
        return u0, later                        # target: (B, 5, ...) at t = .1 .. .5

    # ---------------------------------------------------------------- network adapters
    def features(self, x, in_scale):
        """x (B, n-1, n-1, n-1) float64 -> (B, 4, n, n, n) float64: x / in_scale on the n^3 grid (index 0 = boundary)
        and the coordinates i/n."""
        b, n = x.shape[0], self.n
        f = F.pad(x / in_scale, (1, 0, 1, 0, 1, 0))[:, None]
        g = torch.arange(n, dtype=torch.float64, device=x.device) / n
        cx = g[:, None, None].expand(n, n, n)
        cy = g[None, :, None].expand(n, n, n)
        cz = g[None, None, :].expand(n, n, n)
        return torch.cat((f, torch.stack((cx, cy, cz))[None].expand(b, 3, n, n, n)), dim=1)

    def predict(self, model, x, norm):
        """norm = (in_scale, out_scale). Poisson -> (B, n-1, n-1, n-1); heat -> (B, 6, n-1, n-1, n-1) with t = 0 = x."""
        out = model(self.features(x, norm[0]))[:, :, 1:, 1:, 1:] * norm[1]
        if self.name == 'poisson':
            return out[:, 0]
        return torch.cat((x[:, None], out), dim=1)

    def errors(self, pred, target, x):
        """Per-case relative L2 errors, float64. Poisson: (B,) ||p-u||/||u||. Heat: (B, 6) per output time
        (t = 0 included, = 0 for operators): ||p_t - u_t|| / ||u_t|| (the Table-1 same-grid metric)."""
        if self.name == 'poisson':
            return ((pred - target).square().sum((-3, -2, -1)) / target.square().sum((-3, -2, -1))).sqrt()
        full = torch.cat((x[:, None], target), dim=1)
        return ((pred - full).square().sum((-3, -2, -1)) / full.square().sum((-3, -2, -1))).sqrt()

    def case_metric(self, err):
        """Per-case scalar used for selection/reporting: Poisson err; heat max over all times."""
        return err if err.ndim == 1 else err.max(dim=1).values
