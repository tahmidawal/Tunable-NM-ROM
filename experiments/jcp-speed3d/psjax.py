"""jcp-speed3d F-PS timed entry point (DESIGN A1.9): E0's sine pseudo-spectral ETDRK4 solver (`pscore.py`, verbatim copy)
as ONE jitted function of (b0, nu): ETDRK4 coefficients from traced eigenvalues, the 5 x keep steps and the synthesis of
the six fields on the 63^3 scoring lattice. Gate G7 (`gate`): equal to pscore.make_ps(N, 3).run lattice output to 1e-12
relative; control: half the step must differ by > 1e-9.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import pscore as PS


def lin_eig_j(N, nu):
    k = jnp.arange(1, N, dtype=jnp.float64)
    return -nu * jnp.pi ** 2 * (k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)


def make_ps_query(N, dt, Mc=32):
    keep = int(round(0.05 / dt))
    assert abs(keep * dt - 0.05) < 1e-12, dt
    S_lat = PS.lattice_S(N, 64)

    def step(v, co):
        Nv, _ = PS.nonlinear(v, N, 3)
        a = co['E2'] * v + co['Q'] * Nv
        Na, _ = PS.nonlinear(a, N, 3)
        b = co['E2'] * v + co['Q'] * Na
        Nb, _ = PS.nonlinear(b, N, 3)
        c = co['E2'] * a + co['Q'] * (2 * Nb - Nv)
        Nc, _ = PS.nonlinear(c, N, 3)
        return co['E'] * v + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3']

    def query(b0, nu, S):
        co = PS.etdrk4_coefs(lin_eig_j(N, nu), dt, Mc)

        def interval(v, _):
            v = jax.lax.fori_loop(0, keep, lambda i, x: step(x, co), v)
            return v, PS.synth_lattice(v, S, 3)
        v, lats = jax.lax.scan(interval, b0, None, length=5)
        return jnp.concatenate((PS.synth_lattice(b0, S, 3)[None], lats))     # (6, 63^3)
    q = jax.jit(query)
    return lambda b0, nu: q(b0, nu, S_lat)


def ic(tab, j, N):
    return jnp.asarray(PS.ic_coefs_3d(tab, j, N))


def gate(tab, cases=(0, 1), N=32, dt=0.01):
    out = {}
    q = make_ps_query(N, dt)
    q2 = make_ps_query(N, dt / 2)
    ref = PS.make_ps(N, 3)
    for j in cases:
        b0, nu = ic(tab, j, N), float(tab['nu'][j])
        a = np.asarray(q(b0, nu))
        r = ref(np.asarray(b0), nu, dt, PS.lattice_S(N, 64))['lat']
        h = np.asarray(q2(b0, nu))
        out[j] = dict(rel=float(np.abs(a - r).max() / np.abs(r).max()), half_dt_diff=float(np.abs(h - a).max() / np.abs(r).max()))
    out['pass'] = all(v['rel'] <= 1e-12 for k, v in out.items() if k != 'pass')
    out['control_fires'] = all(v['half_dt_diff'] > 1e-9 for k, v in out.items() if k not in ('pass',))
    return out
