"""imp-varcoef problem family (DESIGN §2-3, A1.7): cohorts and the variable diffusion coefficient.

kappa(x; mu) = nu * exp(s * ghat(x)),  s = ln C,
g(x, y) = sum_{0<=j,k<=3, (j,k)!=(0,0)} xi_jk / (1 + j^2 + k^2) cos(j pi x) cos(k pi y),
ghat = (g - min g) / (max g - min g) - 1/2 with the extrema on the 1025^2 nodal grid.

A case is a row (cx, cy, w, a, nu, xi[15]); row j of a cohort comes from numpy default_rng([seed, j]) drawing, in order,
cx, cy ~ U[.15, .85], w ~ U[.05, .20], a ~ U[.5, 2], ln nu ~ U[ln .02, ln .1], xi ~ N(0, 1)^15.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

SEEDS = dict(dev=2026102101, val=2026102102, test=2026102103)
SIZES = dict(dev=8, val=24, test=16)
NU_RANGE = (0.02, 0.1)
JK = [(j, k) for j in range(4) for k in range(4) if (j, k) != (0, 0)]       # 15 KL terms
NORM_GRID = 1025


def draw(seed, j):
    r = np.random.default_rng([seed, j])
    cx, cy = r.uniform(.15, .85), r.uniform(.15, .85)
    w = r.uniform(.05, .20)
    a = r.uniform(.5, 2.0)
    nu = float(np.exp(r.uniform(np.log(NU_RANGE[0]), np.log(NU_RANGE[1]))))
    xi = r.standard_normal(len(JK))
    return dict(desc=np.array([cx, cy, w, a, nu]), xi=xi)


def cohort(name, unlock_test=False):
    """Return a list of case dicts (desc (5,), xi (15,), gmin, gmax, id). The test split opens only for the
    test-once step after the selection is frozen (DESIGN A2.18): unlock_test=True, used by vref/vtest test configs."""
    assert name in SEEDS, name
    if name == 'test' and not unlock_test:
        raise PermissionError('test split is reserved for the frozen test-once step (DESIGN A2.18)')
    out = []
    for j in range(SIZES[name]):
        c = draw(SEEDS[name], j)
        lo, hi = g_extrema(c['xi'], NORM_GRID)
        c.update(gmin=lo, gmax=hi, id=f'{name}|{j}')
        out.append(c)
    return out


def coef_vec(xi):
    return np.array([xi[i] / (1.0 + j * j + k * k) for i, (j, k) in enumerate(JK)])


def g_field_np(xi, x, y):
    """g on the tensor grid x (n,), y (m,) -> (n, m), numpy (preparation / checks)."""
    c = coef_vec(xi)
    out = np.zeros((len(x), len(y)))
    for i, (j, k) in enumerate(JK):
        out += c[i] * np.outer(np.cos(j * np.pi * x), np.cos(k * np.pi * y))
    return out


def g_extrema(xi, n):
    x = np.linspace(0.0, 1.0, n)
    g = g_field_np(xi, x, x)
    return float(g.min()), float(g.max())


def kappa_params(case, C):
    """Traced-friendly parameter vector for kappa: [nu, s, gmin, gmax, c_1..c_15]."""
    return np.concatenate([[case['desc'][4], np.log(C), case['gmin'], case['gmax']], coef_vec(case['xi'])])


def kappa_tensor(kp, x, y):
    """kappa on the tensor grid x (n,), y (m,) (jnp, traced kp) -> (n, m)."""
    nu, s, lo, hi, c = kp[0], kp[1], kp[2], kp[3], kp[4:]
    cx = jnp.stack([jnp.cos(j * jnp.pi * x) for j in range(4)])      # (4, n)
    cy = jnp.stack([jnp.cos(k * jnp.pi * y) for k in range(4)])      # (4, m)
    Cm = jnp.zeros((4, 4), x.dtype)
    for i, (j, k) in enumerate(JK):
        Cm = Cm.at[j, k].set(c[i])
    g = cx.T @ Cm @ cy
    gh = (g - lo) / (hi - lo) - 0.5
    return nu * jnp.exp(s * gh)


def kappa_np(kp, x, y):
    return np.asarray(kappa_tensor(jnp.asarray(kp), jnp.asarray(x), jnp.asarray(y)))


def kappa_grad_np(kp, x, y):
    """(kappa, d kappa/dx, d kappa/dy) on the tensor grid, numpy (for the manufactured-solution check)."""
    nu, s, lo, hi, c = kp[0], kp[1], kp[2], kp[3], kp[4:]
    g = np.zeros((len(x), len(y)))
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    for i, (j, k) in enumerate(JK):
        g += c[i] * np.outer(np.cos(j * np.pi * x), np.cos(k * np.pi * y))
        gx += c[i] * np.outer(-j * np.pi * np.sin(j * np.pi * x), np.cos(k * np.pi * y))
        gy += c[i] * np.outer(np.cos(j * np.pi * x), -k * np.pi * np.sin(k * np.pi * y))
    kap = nu * np.exp(s * ((g - lo) / (hi - lo) - 0.5))
    f = s / (hi - lo)
    return kap, kap * f * gx, kap * f * gy


def realised(case, C, n=4097):
    """Realised continuous-grid extrema of kappa on an n^2 nodal grid (A1.7)."""
    x = np.linspace(0.0, 1.0, n)
    k = kappa_np(kappa_params(case, C), x, x)
    return dict(kmin=float(k.min()), kmax=float(k.max()), contrast=float(k.max() / k.min()))
