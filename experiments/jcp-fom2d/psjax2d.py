"""jcp-fom2d F-PS timed entry point (DESIGN §2): E0's sine pseudo-spectral ETDRK4 solver (`pscore.py`, byte-identical
copy) as ONE jitted function of (b0, nu): ETDRK4 coefficients from traced eigenvalues (Kassam-Trefethen contour mean,
32 points, pscore.etdrk4_coefs), the 5 x keep steps (pscore's step arithmetic, as speed3d psjax.py for d = 2) and the
synthesis of the six output fields:

  out='lattice'   the 257^2 scoring lattice with zero walls (pscore.synth_lattice with pscore.lattice_S(N, 256));
  out=('mesh', M) the (M+1)^2 nodes of mesh M with zero walls, by pscore.sine_synth_last on the coefficients zero-padded
                  to (M-1)^2 (FFT synthesis; equal to the matrix synthesis at M = 256, gate G-INT).

The query returns (fields, running max |u| on the padded grid over every stage evaluation, as pscore.make_ps).
Gate G-PS1 (`gate`): lattice output equal to pscore.make_ps(N, 2).run to 1e-12 relative and the same running maximum;
control: half the step must differ by > 1e-9.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import pscore as PS


def lin_eig_j(N, nu):
    k = jnp.arange(1, N, dtype=jnp.float64)
    return -nu * jnp.pi ** 2 * (k[:, None] ** 2 + k[None, :] ** 2)


def synth_mesh(b, M):
    """Continuous sine coefficients b (n, n), n <= M - 1 -> values on the (M+1)^2 nodes of mesh M (zero walls)."""
    n = b.shape[-1]
    bp = jnp.pad(b, [(0, 0)] * (b.ndim - 2) + [(0, M - 1 - n), (0, M - 1 - n)])
    v = PS.along(PS.sine_synth_last, PS.along(PS.sine_synth_last, bp, bp.ndim - 1, M), bp.ndim - 2, M)
    return jnp.pad(v, [(0, 0)] * (b.ndim - 2) + [(1, 1), (1, 1)])


def make_ps_query(N, dt, out='lattice', Mc=32):
    keep = int(round(0.05 / dt))
    assert abs(keep * dt - 0.05) < 1e-12, dt
    if out == 'lattice':
        S_lat = PS.lattice_S(N, 256)
        synth = lambda v, S: PS.synth_lattice(v, S, 2)
    else:
        kind, M = out
        assert kind == 'mesh' and M >= N, out
        S_lat = jnp.zeros((1,))
        synth = lambda v, S: synth_mesh(v, M)

    def step(carry, co):
        v, um = carry
        Nv, m1 = PS.nonlinear(v, N, 2)
        a = co['E2'] * v + co['Q'] * Nv
        Na, m2 = PS.nonlinear(a, N, 2)
        b = co['E2'] * v + co['Q'] * Na
        Nb, m3 = PS.nonlinear(b, N, 2)
        c = co['E2'] * a + co['Q'] * (2 * Nb - Nv)
        Nc, m4 = PS.nonlinear(c, N, 2)
        um = jnp.maximum(um, jnp.maximum(jnp.maximum(m1, m2), jnp.maximum(m3, m4)))     # pscore.make_ps running max
        return co['E'] * v + Nv * co['f1'] + 2 * (Na + Nb) * co['f2'] + Nc * co['f3'], um

    def query(b0, nu, S):
        co = PS.etdrk4_coefs(lin_eig_j(N, nu), dt, Mc)

        def interval(carry, _):
            carry = jax.lax.fori_loop(0, keep, lambda i, x: step(x, co), carry)
            return carry, synth(carry[0], S)
        (v, um), outs = jax.lax.scan(interval, (b0, jnp.float64(0.0)), None, length=5)
        return jnp.concatenate((synth(b0, S)[None], outs)), um
    q = jax.jit(query)
    cache = {}

    def run(b0, nu):
        """Ahead-of-time compiled on first use (compile seconds in run.compile_seconds, 0 afterwards)."""
        args = (jnp.asarray(b0, jnp.float64), jnp.float64(nu), S_lat)
        run.compile_seconds = 0.0
        if 'exe' not in cache:
            import time as _t
            t = _t.perf_counter()
            cache['exe'] = q.lower(*args).compile()
            run.compile_seconds = _t.perf_counter() - t
        return cache['exe'](*args)
    run.compile_seconds = 0.0
    return run


def ic(desc, N, panels=256):
    """Sine coefficients of the continuous IC (pscore.ic_coefs_2d, composite Gauss-Legendre, E0's default panels)."""
    return jnp.asarray(PS.ic_coefs_2d(desc, N, panels))


def gate(descs, nus, N=32, dt=0.01):
    """G-PS1: equal to pscore.make_ps(N, 2).run lattice output (1e-12 relative); half-step control (> 1e-9)."""
    assert len(descs) == 2 and len(nus) == 2, 'G-PS1 fixture: exactly the two registered cases (A1.5)'
    out = {}
    q, q2 = make_ps_query(N, dt), make_ps_query(N, dt / 2)
    ref = PS.make_ps(N, 2)
    for j, (desc, nu) in enumerate(zip(descs, nus)):
        b0 = ic(desc, N)
        a, um = q(b0, nu)
        a = np.asarray(a)
        rr = ref(np.asarray(b0), nu, dt, PS.lattice_S(N, 256))
        r = rr['lat']
        h = np.asarray(q2(b0, nu)[0])
        out[j] = dict(rel=float(np.abs(a - r).max() / np.abs(r).max()), half_dt_diff=float(np.abs(h - a).max() / np.abs(r).max()),
                      umax=float(um), umax_ref=rr['umax'], umax_equal=bool(abs(float(um) - rr['umax']) <= 1e-12 * rr['umax']))
    vals = list(out.values())
    return dict(cases=out, passed=all(v['rel'] <= 1e-12 and v['umax_equal'] for v in vals),
                control_fires=all(v['half_dt_diff'] > 1e-9 for v in vals))
