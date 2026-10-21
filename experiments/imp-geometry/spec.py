"""imp-geometry C-SPEC (box control only): sine pseudo-spectral ETDRK4 on (0,2)^2 (DESIGN 5.1, A1).

By the scaling x = s xi, t = s tau (s = SIDE) the side-s problem (nu, T, IC (cx, cy, w, a)) is the unit-box problem with
nu' = nu / s, T' = T / s, IC (cx/s, cy/s, w/s, a); the numerical kernels are E0's vendor/pscore.py (byte-identical;
3/2-padded products, exact linear part, Kassam-Trefethen ETDRK4). The query computes the IC sine coefficients by a discrete
sine transform of the (non-separable, A1.2) initial condition sampled on the n_init grid (knob), the nu-dependent ETDRK4 coefficients, all steps,
and the synthesis to the scoring lattice (x = i/64 on side s = xi = i/(64 s) on the unit box).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))
import pscore as PS       # noqa: E402
import geom as G          # noqa: E402


def make_spec(N, nsteps, n_init=256, n_out=G.N_OUT):
    s = G.SIDE
    m_lat = int(round(s * G.LAT))
    dt = G.T_END / s / nsteps                         # unit-box time step
    keep = nsteps // n_out
    assert keep * n_out == nsteps
    Mi = n_init
    xi = np.arange(1, Mi) / Mi
    k = np.arange(1, N)
    Si = jnp.asarray((2.0 / Mi) * np.sin(np.pi * np.outer(k, xi)))           # (N-1, Mi-1)
    Xg, Yg = (jnp.asarray(a_) for a_ in np.meshgrid(xi, xi, indexing='ij'))
    S_lat = PS.lattice_S(N, m_lat)
    kk = jnp.arange(1, N, dtype=jnp.float64)
    lam0 = -jnp.pi ** 2 * (kk[:, None] ** 2 + kk[None, :] ** 2)

    def synth(v):
        return (S_lat @ v @ S_lat.T).reshape(-1)

    def q(d):
        nu = d[4]
        u0 = G.ic('B', d, Xg * s, Yg * s, jnp)                      # samples on the M_init grid (physical coords)
        b0 = Si @ u0 @ Si.T                                          # discrete sine transform (trapezoid, exact
        #                                                              for modes < M_init)
        co = PS.etdrk4_coefs(nu / s * lam0, dt)

        def step(i, v):
            Nv, _ = PS.nonlinear(v, N, 2)
            a_ = co['E2'] * v + co['Q'] * Nv
            Na, _ = PS.nonlinear(a_, N, 2)
            b_ = co['E2'] * v + co['Q'] * Na
            Nb, _ = PS.nonlinear(b_, N, 2)
            c_ = co['E2'] * a_ + co['Q'] * (2 * Nb - Nv)
            Nc, _ = PS.nonlinear(c_, N, 2)
            return co['E'] * v + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3']

        def block(v, _):
            v = jax.lax.fori_loop(0, keep, step, v)
            return v, synth(v)
        _, outs = jax.lax.scan(block, b0, None, length=n_out)
        return jnp.concatenate([synth(b0)[None], outs]), jnp.zeros(3, jnp.int32)
    return jax.jit(q)
