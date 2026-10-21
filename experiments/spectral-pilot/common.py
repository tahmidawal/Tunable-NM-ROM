"""spectral-pilot: shared pieces for Part A (box) -- cohorts, the sine pseudo-spectral ETDRK4 FOM wrapper, lattice
synthesis, the error metric, and the reduced-model (ROM) building blocks. PILOT: small, not a result.

Numerical kernels of the FOM are the vendored E0 cross-check solver (vendor/pscore.py, byte-identical):
`nonlinear` (3/2-padded products, coefficient-space right-hand side -P[u (u_x + u_y)]), `etdrk4_coefs`
(Kassam-Trefethen contour means), `sine_synth_last` / `cos_synth_last` / `along` / `kvec` (transforms),
`composite_gl` / `sine_integrals` (initial-condition quadrature). This module only adds an output schedule, batching
over cases (vmap), lattice synthesis by FFT, and the ROM.

State convention: u(x, y) = sum_{k1,k2=1}^{N-1} b[k1-1, k2-1] sin(k1 pi x) sin(k2 pi y).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import pscore as P  # noqa: E402

RUNS = HERE / 'runs'
T_END = 0.25
OUT_TIMES = np.arange(6) * 0.05
LAT = 256                       # scoring lattice x_i = i / 256 (257^2 with walls)
UMAX = 2.0                      # amplitude bound of the IC family (a <= 2)
PILOT = 'PILOT: small, not a result.'

# ------------------------------------------------------------------------------------------------- cohorts ----

LEVELS = {
    # name: (nu_lo, N_ref, n_train); nu = nu_lo * 2**u_nu (DESIGN A1.2: paired rows, factor 3 between levels)
    'A1': (0.06, 128, 96),
    'A2': (0.02, 192, 96),
    'A3': (0.02 / 3, 384, 96),
    'A4': (0.02 / 9, 768, 48),
}
TRAIN_SEED, TEST_SEED = 1_000_001, 2_000_001
N_TEST = 12
# IC family (DESIGN A1.2, compatible IC): a ~ U[1, 2], c ~ U[0.3, 0.7]^2, w ~ U[0.035, 0.05]
FAMILY = dict(a=(1.0, 2.0), c=(0.3, 0.7), w=(0.035, 0.05))
CAMPAIGN_SEEDS = {7090702, 911702, 20260927, 20260916, 923651, 923801, 923901, 0, 5550001, 5550002}


def draw_rows(seed, n, fam=FAMILY):
    """(n, 5) rows (cx, cy, w, a, u_nu); row j from default_rng([seed, j]) (prefix-stable)."""
    assert seed not in CAMPAIGN_SEEDS
    rows = []
    for j in range(n):
        r = np.random.default_rng([int(seed), int(j)])
        cx, cy = r.uniform(*fam['c']), r.uniform(*fam['c'])
        w = r.uniform(*fam['w'])
        a = r.uniform(*fam['a'])
        rows.append((cx, cy, w, a, r.uniform()))
    return np.asarray(rows, float)


def with_nu(rows, nu_lo):
    D = rows.copy()
    D[:, 4] = nu_lo * 2.0 ** rows[:, 4]
    return D


def level_cohorts(name):
    nu_lo, Nref, ntr = LEVELS[name]
    tr = with_nu(draw_rows(TRAIN_SEED, ntr), nu_lo)
    te = with_nu(draw_rows(TEST_SEED, N_TEST), nu_lo)
    for t in te:                                          # G6: disjoint descriptors
        assert not np.any(np.all(np.isclose(tr, t), axis=1))
    return tr, te


def steep(D):
    return D[:, 3] / D[:, 4]


# --------------------------------------------------------------------------------------- initial condition ----

IC_PANELS, IC_PTS = 64, 16       # composite Gauss-Legendre (1024 points) -- checked against pscore's 256 x 32 (G7)
# code audit 1: the 1024-point rule fails G7 above N ~ 384 (8.9e-11 at 512, 8.2e-6 at 768); N > 384 uses pscore's rule


def ic_panels(N):
    return (IC_PANELS, IC_PTS) if N <= 384 else (256, 32)


class ICRule:
    """Precomputed (case-independent) quadrature for the separable IC: b0 = a gx gy^T with
    g_k = 2 int_0^1 exp(-(x-c)^2 / (2 w^2)) sin(k pi x) dx. Constant tables are allowed offline for every method."""

    def __init__(self, N, panels=None, pts=None):
        if panels is None:
            panels, pts = ic_panels(N)
        X, W = P.composite_gl(panels, pts)
        self.X = jnp.asarray(X)
        self.SW = jnp.asarray(2.0 * W[:, None] * np.sin(np.pi * np.outer(X, np.arange(1, N))))   # (Q, N-1)

    def coefs(self, d):
        cx, cy, w, a = d[0], d[1], d[2], d[3]
        gx = jnp.exp(-(self.X - cx) ** 2 / (2 * w * w)) @ self.SW
        gy = jnp.exp(-(self.X - cy) ** 2 / (2 * w * w)) @ self.SW
        return gx, gy, a


# ------------------------------------------------------------------------------------------------- lattice ----

def synth_lattice(b, N):
    """(N-1, N-1) coefficients -> (257, 257) values at x_i = i/256 (walls zero), by DST via the FFT at the smallest
    multiple Mo of 256 with Mo >= N, then subsampling."""
    Mo = LAT * max(1, math.ceil(N / LAT))
    s = Mo // LAT
    bp = jnp.pad(b, ((0, Mo - N), (0, Mo - N)))
    v = P.along(P.sine_synth_last, P.along(P.sine_synth_last, bp, 1, Mo), 0, Mo)    # (Mo-1, Mo-1), x_j = j/Mo
    v = v[s - 1::s, s - 1::s]
    return jnp.pad(v, 1)


def err_runs(F, R):
    """F, R: (..., 6, 257, 257). e = max_{k>=1} ||F_k - R_k|| / ||R_0||."""
    num = jnp.sqrt(jnp.sum((F - R) ** 2, axis=(-1, -2)))
    den = jnp.sqrt(jnp.sum(R[..., 0, :, :] ** 2, axis=(-1, -2)))
    e = jnp.max(num[..., 1:], axis=-1) / den
    return jnp.where(jnp.all(jnp.isfinite(F), axis=(-1, -2, -3)), e, jnp.inf)


def pad_to(B, Nref):
    """(..., N-1, N-1) coefficients -> (..., Nref-1, Nref-1), zero-padded (or truncated if N > Nref)."""
    n = B.shape[-1]
    m = Nref - 1
    if n >= m:
        return B[..., :m, :m]
    pw = [(0, 0)] * (B.ndim - 2) + [(0, m - n), (0, m - n)]
    return jnp.pad(B, pw)


def err_coef(B, Bref):
    """Continuous L2 error (DESIGN A1.4), exact in coefficient space: B, Bref (..., 6, n, n) on the same grid of modes.
    e = max_{k>=1} ||B_k - Bref_k|| / ||Bref_0||; non-finite runs -> inf."""
    num = jnp.sqrt(jnp.sum((B - Bref) ** 2, axis=(-1, -2)))
    den = jnp.sqrt(jnp.sum(Bref[..., 0, :, :] ** 2, axis=(-1, -2)))
    e = jnp.max(num[..., 1:], axis=-1) / den
    return jnp.where(jnp.all(jnp.isfinite(B), axis=(-1, -2, -3)), e, jnp.inf)


# ------------------------------------------------------------------------------------------------ ETDRK4 -------

def etd_step(v, co, NL):
    """One ETDRK4 (Cox-Matthews) step, the arithmetic of pscore.make_ps.step."""
    Nv = NL(v)
    a = co['E2'] * v + co['Q'] * Nv
    Na = NL(a)
    b = co['E2'] * v + co['Q'] * Na
    Nb = NL(b)
    c = co['E2'] * a + co['Q'] * (2 * Nb - Nv)
    Nc = NL(c)
    return co['E'] * v + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3']


def rollout(v0, Lk, NL, n, n_out):
    """n ETDRK4 steps of length T_END/n; returns (n_out+1, ...) states at the n_out+1 equispaced output times."""
    assert n % n_out == 0, (n, n_out)
    dt = T_END / n
    co = P.etdrk4_coefs(Lk, dt)
    per = n // n_out

    def block(v, _):
        v = jax.lax.fori_loop(0, per, lambda i, x: etd_step(x, co, NL), v)
        return v, v
    _, outs = jax.lax.scan(block, v0, None, length=n_out)
    return jnp.concatenate([v0[None], outs])


def k2grid(N):
    k = jnp.arange(1, N, dtype=jnp.float64) * jnp.pi
    return k[:, None] ** 2 + k[None, :] ** 2


# --------------------------------------------------------------------------------------------- spectral FOM ----

def fom_nl(N, M=None):
    """pscore.nonlinear (M = 3N/2); with M given, the same arithmetic on an M-point padded grid (padding gate, A3)."""
    if M is None:
        return lambda b: P.nonlinear(b, N, 2)[0]
    return lambda b: nonlinear_M(b, N, M)


def nonlinear_M(b, N, M):
    """Copy of pscore.nonlinear (d = 2) with the padded grid size M as a parameter (vendor file untouched)."""
    bp = jnp.pad(b, [(0, M - N)] * 2)
    n = M - 1
    S = lambda x, ax: P.along(P.sine_synth_last, x, ax, M)
    Cc = lambda x, ax: P.along(P.cos_synth_last, x, ax, M)
    Ty = S(bp, 1)
    u = S(Ty, 0)
    g = Cc(P.kvec(n, 2, 0) * Ty, 0) + S(Cc(P.kvec(n, 2, 1) * bp, 1), 0)
    f = P.analysis(u * g, M)
    return -f[:N - 1, :N - 1]


def make_fom(N, n, n_out=5, lattice=True, M=None):
    """Batched FOM: (B, 5) descriptors -> (B, n_out+1, 257, 257) lattice fields (or coefficients if lattice=False).
    Includes the IC coefficients, the rollout and the output synthesis."""
    rule = ICRule(N)
    K2 = k2grid(N)
    NL = fom_nl(N, M)

    def one(d):
        gx, gy, a = rule.coefs(d)
        b0 = a * jnp.outer(gx, gy)
        B = rollout(b0, -d[4] * K2, NL, n, n_out)
        if not lattice:
            return B
        return jax.vmap(lambda b: synth_lattice(b, N))(B)
    return jax.jit(jax.vmap(one))


def ref_cfl_steps(N, factor=1.0):
    """Advective step estimate for ETDRK4: dt ~ 0.6 / (u_max N) (DESIGN 4.1) -> steps n = 5 * 2^k >= T/dt * factor."""
    need = T_END * UMAX * N / 0.6 * factor
    n = 5
    while n < need:
        n *= 2
    return n


# ------------------------------------------------------------------------------------------------ utilities ---

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def env_record():
    return dict(backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
                precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), xla_flags=os.environ.get('XLA_FLAGS'),
                jax=jax.__version__, commit=git_commit(), host=os.uname().nodename,
                time=time.strftime('%Y-%m-%dT%H:%M:%S'), label=PILOT)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def chunks(x, k):
    return [x[i:i + k] for i in range(0, len(x), k)]
