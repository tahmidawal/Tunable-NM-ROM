"""imp-varcoef spectral competitors (DESIGN §6, A1.8, A1.10, A1.12, A1.14): PSI (sine-Galerkin IMEX + warm-started
constant-coefficient-preconditioned CG) and ETD (ETDRK4 with a constant-coefficient split).

Coefficients b (N-1, N-1): u = sum_{k,l} b_kl sin(k pi x) sin(l pi y) (the pscore convention; projection = 4 * integral).
Variable diffusion, Galerkin with the tensor midpoint rule (Q points per axis, x_q = (q + 1/2)/Q, w_q = 1/Q):
    (D b)_kl = -4 [ Cd^T (W kappa o (Cd b S^T)) S + S^T (W kappa o (S b Cd^T)) Cd ]_kl,
S_qk = sin(k pi x_q), Cd_qk = k pi cos(k pi x_q). D is symmetric negative definite (A1.8).
Advection: pscore.nonlinear (3/2-oversampled collocation, not exact dealiasing).

One jitted program per (family, N, dt, scheme/kbar mode, tol, maxiter, output); traced inputs: b0 (prepared IC
coefficients, outside the timer) and kp (the 19 kappa parameters, vcoef.kappa_params). Everything else, including kappa
evaluation at the quadrature points, preconditioner, ETD coefficients and output synthesis, is inside the program.
Returns (fields (6, ...), stats dict).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
FOM2D = HERE.parent / 'jcp-fom2d'
if str(FOM2D) not in sys.path:
    sys.path.insert(0, str(FOM2D))
import pscore as PS            # noqa: E402  (jcp-fom2d copy of E0 pscore, unchanged)
import psjax2d                 # noqa: E402
import vcoef as VC             # noqa: E402

UMAX_GUARD = 10.0


def galerkin_mats(N, Q):
    xq = (jnp.arange(Q, dtype=jnp.float64) + 0.5) / Q
    k = jnp.arange(1, N, dtype=jnp.float64) * jnp.pi
    S = jnp.sin(xq[:, None] * k[None, :])
    Cd = k[None, :] * jnp.cos(xq[:, None] * k[None, :])
    return xq, S, Cd


def kappa_q(N, Q, kp):
    """kappa at the Q^2 midpoint quadrature points (traced kp)."""
    xq = (jnp.arange(Q, dtype=jnp.float64) + 0.5) / Q
    return VC.kappa_tensor(kp, xq, xq)


def D_of(N, Q, Kpts):
    """D(b) for kappa sampled at the quadrature points (Kpts, unweighted)."""
    _, S, Cd = galerkin_mats(N, Q)
    Kq = Kpts / (Q * Q)

    def D(b):
        ux = Cd @ b @ S.T
        uy = S @ b @ Cd.T
        return -4.0 * (Cd.T @ (Kq * ux) @ S + S.T @ (Kq * uy) @ Cd)
    return D


def make_D(N, Q, kp):
    """Return (D, kappa at the quadrature points) for the traced kappa parameters."""
    Kpts = kappa_q(N, Q, kp)
    return D_of(N, Q, Kpts), Kpts


class Program:
    """setup(kp) -> S; core(x0, S) -> (fields, stats); query(x0, kp) = core(x0, setup(kp)) as ONE jitted program.
    kernel = jit(core), setup_only = jit(setup) (DESIGN A3.14)."""

    def __init__(self, setup, core, name):
        self.name = name
        self.setup_fn, self.core_fn = setup, core
        self.query = jax.jit(lambda x0, kp: core(x0, setup(kp)))
        self.kernel = jax.jit(core)
        self.setup_only = jax.jit(setup)

    def __call__(self, x0, kp):
        return self.query(x0, kp)


def lam_ps(N):
    k = jnp.arange(1, N, dtype=jnp.float64)
    return jnp.pi ** 2 * (k[:, None] ** 2 + k[None, :] ** 2)


def pcg(Aop, rhs, x0, Pinv, tol, maxiter):
    """Preconditioned CG (SPD A, SPD diagonal-or-linear Pinv); stops at |r| <= tol |rhs| or maxiter.
    Returns x, iterations, independently recomputed true relative residual."""
    nb = jnp.maximum(jnp.linalg.norm(rhs), 1e-300)
    r0 = rhs - Aop(x0)
    z0 = Pinv(r0)
    c0 = (x0, r0, z0, jnp.vdot(r0, z0), jnp.int32(0))

    def cond(c):
        return (jnp.linalg.norm(c[1]) > tol * nb) & (c[4] < maxiter)

    def body(c):
        x, r, p, rz, it = c
        Ap = Aop(p)
        a = rz / jnp.vdot(p, Ap)
        x = x + a * p
        r = r - a * Ap
        z = Pinv(r)
        rz2 = jnp.vdot(r, z)
        p = z + (rz2 / rz) * p
        return x, r, p, rz2, it + 1
    # carry p in slot 2 (initial p = z0)
    x, r, _, _, it = jax.lax.while_loop(cond, body, c0)
    tr = jnp.linalg.norm(rhs - Aop(x)) / nb
    return x, it, tr


def _synth(out, N):
    if out == 'lattice':
        S_lat = PS.lattice_S(N, 256)
        return lambda b: PS.synth_lattice(b, S_lat, 2)
    kind, M = out
    assert kind == 'mesh' and M >= N
    return lambda b: psjax2d.synth_mesh(b, M)


def make_psi(N, dt, scheme, tol, out='lattice', maxiter=200, Q=None):
    """PSI: SBDF2 (first step SBDF1) or CNAB2 (first step CN + forward Euler); returns a Program (x0 = IC coefficients)."""
    assert scheme in ('SBDF2', 'CNAB2'), scheme
    keep = int(round(0.05 / dt))
    assert abs(keep * dt - 0.05) < 1e-12, dt
    Q = 2 * N if Q is None else Q
    synth = _synth(out, N)
    nst = 5 * keep
    theta = 1.0 if scheme == 'SBDF2' else 0.5
    lam = lam_ps(N)

    def setup(kp):
        nu = kp[0]
        den1 = 1.0 + theta * dt * nu * lam
        den2 = (1.5 if scheme == 'SBDF2' else 1.0) + theta * dt * nu * lam
        return dict(K=kappa_q(N, Q, kp), den1=den1, den2=den2)

    def core(b0, S_):
        D = D_of(N, Q, S_['K'])

        def step(k, c):
            b, bp, Np, um, st = c
            Nb, m = PS.nonlinear(b, N, 2)
            first = k == 0
            if scheme == 'SBDF2':
                alpha = jnp.where(first, 1.0, 1.5)
                rhs = jnp.where(first, b + dt * Nb, 2 * b - 0.5 * bp + dt * (2 * Nb - Np))
            else:
                alpha = 1.0
                rhs = b + 0.5 * dt * D(b) + dt * jnp.where(first, Nb, 1.5 * Nb - 0.5 * Np)
            A = lambda x: alpha * x - theta * dt * D(x)
            den = jnp.where(first, S_['den1'], S_['den2'])
            x0 = jnp.where(first, b, 2 * b - bp)
            x, it, tr = pcg(A, rhs, x0, lambda r: r / den, tol, maxiter)
            st = dict(it_sum=st['it_sum'] + it, it_max=jnp.maximum(st['it_max'], it),
                      res_max=jnp.maximum(st['res_max'], tr),
                      it_stage=st['it_stage'].at[k].set(it), res_stage=st['res_stage'].at[k].set(tr),
                      nbad=st['nbad'] + jnp.where((tr <= 10 * tol) & jnp.isfinite(tr), 0, 1),
                      nst=st['nst'] + 1)
            return x, b, Nb, jnp.maximum(um, m), st

        def interval(c, i):
            c = jax.lax.fori_loop(i * keep, (i + 1) * keep, step, c)
            mt = PS.nonlinear(c[0], N, 2)[1]                    # accepted state at the interval end, native padded grid
            c = (c[0], c[1], c[2], jnp.maximum(c[3], mt), c[4])
            return c, (synth(c[0]), c[0])
        st0 = dict(it_sum=jnp.int32(0), it_max=jnp.int32(0), res_max=jnp.float64(0.0), nbad=jnp.int32(0),
                   nst=jnp.int32(0), it_stage=jnp.full(5 * keep, -1, jnp.int32), res_stage=jnp.full(5 * keep, jnp.nan))
        c0 = (b0, b0, jnp.zeros_like(b0), jnp.float64(0.0), st0)
        c, (outs, coefs) = jax.lax.scan(interval, c0, jnp.arange(5))
        fields = jnp.concatenate((synth(b0)[None], outs))
        st = dict(c[4], umax=c[3], nst_expected=jnp.int32(nst), coefs=jnp.concatenate((b0[None], coefs)))
        return fields, st
    return Program(setup, core, f'psi|N{N}|{dt:g}|{scheme}|{tol:g}|Q{Q}')


def make_etd(N, dt, kbar_mode, out='lattice', Q=None, Mc=32):
    """ETD: ETDRK4 (pscore Kassam-Trefethen coefficients) with linear part -kbar pi^2 |k|^2 and the remainder
    D - L plus advection explicit. kbar_mode: 'max' (kappa_max) or 'mid' ((kappa_max + kappa_min) / 2) over the
    quadrature points; 'nu' (= nu) for the C = 1 identity check only. Returns a Program."""
    assert kbar_mode in ('max', 'mid', 'nu'), kbar_mode
    keep = int(round(0.05 / dt))
    assert abs(keep * dt - 0.05) < 1e-12, dt
    Q = 2 * N if Q is None else Q
    synth = _synth(out, N)
    lam = lam_ps(N)

    def setup(kp):
        K = kappa_q(N, Q, kp)
        kb = {'max': jnp.max(K), 'mid': 0.5 * (jnp.max(K) + jnp.min(K)), 'nu': kp[0]}[kbar_mode]
        Lk = -kb * lam
        return dict(K=K, Lk=Lk, co=PS.etdrk4_coefs(Lk, dt, Mc))

    def core(b0, S_):
        D = D_of(N, Q, S_['K'])
        Lk, co = S_['Lk'], S_['co']

        def Nf(v):
            a, m = PS.nonlinear(v, N, 2)
            return a + D(v) - Lk * v, m

        def step(_, carry):
            v, um = carry
            Nv, m1 = Nf(v)
            a = co['E2'] * v + co['Q'] * Nv
            Na, m2 = Nf(a)
            b = co['E2'] * v + co['Q'] * Na
            Nb, m3 = Nf(b)
            c = co['E2'] * a + co['Q'] * (2 * Nb - Nv)
            Nc, m4 = Nf(c)
            um = jnp.maximum(um, jnp.maximum(jnp.maximum(m1, m2), jnp.maximum(m3, m4)))
            return co['E'] * v + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3'], um

        def interval(c, _):
            c = jax.lax.fori_loop(0, keep, step, c)
            c = (c[0], jnp.maximum(c[1], PS.nonlinear(c[0], N, 2)[1]))   # terminal accepted state, native grid
            return c, (synth(c[0]), c[0])
        (v, um), (outs, coefs) = jax.lax.scan(interval, (b0, jnp.float64(0.0)), None, length=5)
        fields = jnp.concatenate((synth(b0)[None], outs))
        z = jnp.int32(0)
        return fields, dict(it_sum=z, it_max=z, res_max=jnp.float64(0.0), nbad=z, nst=jnp.int32(5 * keep),
                            nst_expected=jnp.int32(5 * keep), umax=um, coefs=jnp.concatenate((b0[None], coefs)))
    return Program(setup, core, f'etd|N{N}|{dt:g}|{kbar_mode}|Q{Q}')


def ic(desc, N):
    return psjax2d.ic(desc, N)
