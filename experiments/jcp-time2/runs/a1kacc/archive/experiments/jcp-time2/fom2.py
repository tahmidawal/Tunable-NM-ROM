"""jcp-time2 full-order model with the same two-step LMM time stepping (DESIGN section 9, A1.10, A2.21, A3.5).

The project FOM (experiments/mr-burgers2d/engines.make_fom: sign-upwind advection, 5-point Laplacian, Newton with
BiCGStab and the FFT-DST Helmholtz preconditioner) with its backward-Euler residual generalised to

    a0 u + a1 u_n + a2 u_{n-1} + dt (b0 F(u) + b1 F(u_n)),   F(u) = a_h(u) - nu Lap_h u,

preconditioner (a0 + dt b0 nu (lam_x + lam_y))^{-1} in the DST basis (BE: exactly engines.make_fom's). Startup steps
by BE as in t2core.SCHEMES. dt, steps, output stride, coefficients and tolerances are traced. Per step it records the
final nonlinear relative residual and Newton count, and per Newton iteration the worst BiCGStab relative residual.
A step fails iff its final nonlinear relative residual exceeds ntol or is nonfinite. Per-step arrays (final nonlinear
relative residual, Newton count, worst linear relative residual of that step's Newton iterations) are returned for the
first NMAX steps (entries beyond `steps` are NaN / -1).
"""
from __future__ import annotations

import t2core as T2   # noqa: F401  (sys.path set-up)
import jax
import jax.numpy as jnp
import engines as e

BE_CO = T2.BE_CO


NMAX = 800          # per-step diagnostic buffer (the longest schedule is dt0/8: 400 steps)


def make_fom_lmm(L, max_newton=20):
    k1 = jnp.arange(1, L, dtype=jnp.float64)
    lam1 = 4 * L ** 2 * jnp.sin(jnp.pi * k1 / (2 * L)) ** 2
    lam2 = lam1[:, None] + lam1[None, :]

    def F(u, nu):
        adv, lap = e.spatial(u, L)
        return adv - nu * lap

    def query(u0f, nu, sch, ntol, ltol):
        dt = sch['dt']
        be = jnp.asarray(BE_CO, jnp.float64)
        u0 = u0f[1:-1, 1:-1].reshape(-1)
        n = u0.shape[0]

        def body(c):
            k, un, um, out, st = c
            co = jnp.where(k < sch['nstart'], be, sch['co'])
            a0, a1, a2, b0, b1 = co[0], co[1], co[2], co[3], co[4]
            Fn = jax.lax.cond(b1 != 0., lambda: F(un, nu), lambda: jnp.zeros(n, jnp.float64))
            const = a1 * un + a2 * um + dt * b1 * Fn
            res = lambda u: a0 * u + const + dt * b0 * F(u, nu)
            den = a0 + dt * b0 * nu * lam2

            def pre(v):
                return e.dst2(e.dst2(v.reshape(L - 1, L - 1)) / den).reshape(-1)
            scale = jnp.maximum(jnp.linalg.norm(un), 1e-300)

            def nbody(s):
                u, it, rn, lr = s
                r = res(u)
                jv = lambda v: jax.jvp(res, (u,), (v,))[1]
                delta, _ = jax.scipy.sparse.linalg.bicgstab(jv, -r, tol=ltol, maxiter=200, M=pre)
                lres = jnp.linalg.norm(jv(delta) + r) / jnp.maximum(jnp.linalg.norm(r), 1e-300)
                cand = u + delta
                return cand, it + 1, jnp.linalg.norm(res(cand)), jnp.maximum(lr, lres)
            # predictor: the previous state (as engines.make_fom)
            u, it, rn, lr = jax.lax.while_loop(lambda s: (s[2] > ntol * scale) & (s[1] < max_newton) & jnp.isfinite(s[2]),
                                               nbody, (un, jnp.int32(0), jnp.linalg.norm(res(un)), jnp.float64(0.)))
            rel = rn / scale
            fail = ~(jnp.isfinite(rel) & (rel <= ntol))
            ks = jnp.minimum(k, NMAX - 1)
            st = dict(it_sum=st['it_sum'] + it, it_max=jnp.maximum(st['it_max'], it), nfail=st['nfail'] + fail.astype(jnp.int32),
                      first_fail=jnp.where((st['first_fail'] < 0) & fail, k, st['first_fail']),
                      worst_rel=jnp.maximum(st['worst_rel'], rel), worst_lres=jnp.maximum(st['worst_lres'], lr),
                      step_rel=st['step_rel'].at[ks].set(rel), step_newton=st['step_newton'].at[ks].set(it),
                      step_lres=st['step_lres'].at[ks].set(lr))
            kk = k + 1
            store = (kk % sch['keep']) == 0
            out = jnp.where(store, out.at[jnp.clip(kk // sch['keep'], 0, 5)].set(u), out)
            return kk, u, un, out, st

        out0 = jnp.zeros((6, n), jnp.float64).at[0].set(u0)
        st0 = dict(it_sum=jnp.int32(0), it_max=jnp.int32(0), nfail=jnp.int32(0), first_fail=jnp.int32(-1),
                   worst_rel=jnp.float64(0.), worst_lres=jnp.float64(0.), step_rel=jnp.full(NMAX, jnp.nan),
                   step_newton=jnp.full(NMAX, -1, jnp.int32), step_lres=jnp.full(NMAX, jnp.nan))
        k, _, _, out, st = jax.lax.while_loop(lambda c: c[0] < sch['steps'], body, (jnp.int32(0), u0, u0, out0, st0))
        fields = jnp.pad(out.reshape(6, L - 1, L - 1), ((0, 0), (1, 1), (1, 1)))
        return dict(fields=fields, stats=dict(st, steps=k))

    return jax.jit(query)
