"""burgers3d-retry: solver additions on top of burgers3d-span/common.py (imported unchanged).

make_query_fsc — the A2/A3 fixed-sweep span path with ONE tensor contraction per swept state.
    The tensor rule's advection Jacobian is Ju(w) = Tsym . w (M x R'), LINEAR in w. The predictor candidates are the
    linear combinations we = 2 wv - wp and wq = 3 wv - 3 wp + wp2 of the last three accepted states, so their Ju are
    the same combinations of the cached Ju of those states (exact up to roundoff): the predictor needs no tensor read.
    Every sweep reads the tensor once at the trial state (as before). Everything else — scaled LSPG residual, LM step
    (Cholesky of J^T J + lam diag(J^T J), clip, accept-if-decrease, damping carried), stationarity report, adaptive
    first steps (A3) — is identical to make_query_fs. Optional tensor32: the table stored in float32 and contracted in
    float32 (state rounded to float32 for the contraction only); every other quantity stays float64. This changes the
    rule (it is certified as deployed) and is reported as a separate arm.

make_query_fsh — the same fixed-sweep path for the head (unknowns z in R^K, c = h(z)[:R']), no caching (c is
    nonlinear in z); predictor = the smallest residual among z, 2z - z_prev, 3z - 3z_prev + z_prev2; tensor slice
    M = 4K tests x R' x R'. Initial state: the lane-1 head initial fit (nearest library code + LM).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'burgers3d-span'))
import common as C  # noqa: E402


def contract(data, w, t32):
    if t32:
        return jnp.einsum('mij,j->mi', data['Ts32'], w.astype(jnp.float32),
                          preferred_element_type=jnp.float32).astype(jnp.float64)
    return jnp.einsum('mij,j->mi', data['Ts'], w)


def make_query_fsc(n, Rp, M, dt=C.DT, gtol=1e-3, trust=jnp.inf, adaptive_first=3, t32=False, unroll=True):
    steps = int(round(0.25 / dt))
    keep = int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12

    def query(u0, nu, data, hp):
        A, lam_ = data['A'], data['lam']
        S = 1.0 / (1.0 + dt * nu * lam_)
        t = jnp.concatenate([b.T @ u0 for b in data['Qb']])
        w0 = jax.scipy.linalg.solve_triangular(data['Rq'], t, lower=False)
        ic = jnp.stack([0., 4., 0., jnp.linalg.norm(u0) ** 2 - jnp.linalg.norm(t) ** 2])

        def rj(Ju, c, p):
            Ac = A @ c
            r = (Ac - p + dt * (0.5 * Ju @ c + nu * lam_ * Ac)) * S
            return r, A + (dt * S)[:, None] * Ju

        def sweep(w, Ju, r, J, rn, lam, p):
            Hm = J.T @ J
            g = J.T @ r
            d = jnp.diag(Hm) + 1e-30
            cf = jax.scipy.linalg.cho_factor(Hm + jnp.diag(lam * d), lower=True)
            dw = jax.scipy.linalg.cho_solve(cf, -g)
            nz = jnp.linalg.norm(dw)
            dw = dw * jnp.where(nz > trust, trust / (nz + 1e-300), 1.)
            ok = jnp.all(jnp.isfinite(dw))
            wn = w + jnp.where(ok, dw, 0.)
            Jun = contract(data, wn, t32)
            r2, J2 = rj(Jun, wn, p)
            rn2 = jnp.linalg.norm(r2)
            acc = ok & jnp.isfinite(rn2) & (rn2 < rn)
            lam2 = jnp.where(acc, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            sel = lambda a, b: jnp.where(acc, a, b)
            return (sel(wn, w), sel(Jun, Ju), sel(r2, r), sel(J2, J), sel(rn2, rn), lam2, (~acc).astype(jnp.int32))

        def predict(k, wv, wp, wp2, Jv, Jp, Jp2, p):
            we, Je = 2. * wv - wp, 2. * Jv - Jp
            wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
            Jq = jnp.where(k >= 2, 3. * Jv - 3. * Jp + Jp2, Je)
            cand, Ju3 = jnp.stack((wv, we, wq)), jnp.stack((Jv, Je, Jq))
            r3 = jax.vmap(lambda Ju, c: rj(Ju, c, p)[0])(Ju3, cand)
            rs = jnp.linalg.norm(r3, axis=1)
            i = jnp.argmin(jnp.where(jnp.isfinite(rs), rs, jnp.inf))
            return cand[i], Ju3[i]

        def make_step(nsweep):
            def step(carry, k):
                wv, wp, wp2, Jv, Jp, Jp2, lam = carry
                p = A @ wv
                w, Ju = predict(k, wv, wp, wp2, Jv, Jp, Jp2, p)
                r, J = rj(Ju, w, p)
                rn = jnp.linalg.norm(r)
                rej = jnp.int32(0)
                for _ in range(nsweep):
                    w, Ju, r, J, rn, lam, rj_ = sweep(w, Ju, r, J, rn, lam, p)
                    rej = rej + rj_
                gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * rn + 1e-300)
                reason = jnp.where(jnp.isfinite(rn), jnp.where(gn <= gtol, 4, 0), 3).astype(jnp.int32)
                return (w, wv, wp, Ju, Jv, Jp, lam), (w, rn, jnp.int32(nsweep), reason, gn, rej)
            return step

        J0 = contract(data, w0, t32)
        carry = (w0, w0, w0, J0, J0, J0, jnp.asarray(1e-6, dtype=jnp.float64))
        if adaptive_first:
            def evalJ(w, p_):
                return rj(contract(data, w, t32), w, p_)
            lmA = C.make_fused_lm(evalJ, Rp, 50, trust, gtol, clip=True)

            def astep(carry, k):
                wv, wp, wp2, Jv, Jp, Jp2, lam = carry
                p = A @ wv
                wi, _ = predict(k, wv, wp, wp2, Jv, Jp, Jp2, p)
                w2, rn, it, reason, gn, rej, lam2 = lmA(wi, p, 1e-12 * jnp.linalg.norm(t), lam)
                return (w2, wv, wp, contract(data, w2, t32), Jv, Jp, lam2), (w2, rn, it, reason, gn, rej)
            carry, o1 = jax.lax.scan(astep, carry, jnp.arange(adaptive_first))
            _, o2 = jax.lax.scan(make_step(1), carry, jnp.arange(adaptive_first, steps), unroll=unroll)
        else:
            carry, o1 = jax.lax.scan(make_step(3), carry, jnp.arange(2), unroll=True)
            _, o2 = jax.lax.scan(make_step(1), carry, jnp.arange(2, steps), unroll=unroll)
        ws, rn, it, reason, gn, rej = (jnp.concatenate((x, y)) for x, y in zip(o1, o2))
        internal = jnp.concatenate((w0[None], ws))
        X = data['Rq'] @ internal[::keep].T
        out, off = 0., 0
        for b in data['Qb']:
            wdt = b.shape[1]
            out = out + b @ X[off:off + wdt]
            off += wdt
        return out.T, internal, it, reason, gn, rn, rej, ic
    return jax.jit(query), jax.jit(lambda w, hp: w)


def make_query_fsh(n, Rp, M, K, dt=C.DT, gtol=1e-3, trust=jnp.inf, adaptive_first=3, fit_budget=200, unroll=True):
    steps = int(round(0.25 / dt))
    keep = int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12
    hfit = C.make_head_fit(fit_budget)
    coef = lambda z, hp: C.head(hp, z)[:Rp]

    def query(u0, nu, data, hp):
        A, lam_, Ts = data['A'], data['lam'], data['Ts']
        S = 1.0 / (1.0 + dt * nu * lam_)
        t = jnp.concatenate([b.T @ u0 for b in data['Qb']])
        i0 = jnp.argmin(jnp.sum((data['RH'] - t[None, :]) ** 2, axis=1))
        z0, rn0, it0, reason0, gn0, _, _ = hfit(hp, data['Z'][i0], data['Rq'], t, Rp)
        ic = jnp.stack([it0.astype(jnp.float64), reason0.astype(jnp.float64), gn0, rn0])

        def res(z, p):
            c = coef(z, hp)
            Ac = A @ c
            return (Ac - p + dt * (0.5 * jnp.einsum('mij,i,j->m', Ts, c, c) + nu * lam_ * Ac)) * S

        def evalJ(z, p):
            c, Jh = coef(z, hp), jax.jacfwd(lambda zz: coef(zz, hp))(z)
            Ju = jnp.einsum('mij,j->mi', Ts, c)
            Ac = A @ c
            r = (Ac - p + dt * (0.5 * Ju @ c + nu * lam_ * Ac)) * S
            return r, (A + (dt * S)[:, None] * Ju) @ Jh

        def sweep(z, r, J, rn, lam, p):
            Hm = J.T @ J
            g = J.T @ r
            d = jnp.diag(Hm) + 1e-30
            cf = jax.scipy.linalg.cho_factor(Hm + jnp.diag(lam * d), lower=True)
            dz = jax.scipy.linalg.cho_solve(cf, -g)
            nz = jnp.linalg.norm(dz)
            dz = dz * jnp.where(nz > trust, trust / (nz + 1e-300), 1.)
            ok = jnp.all(jnp.isfinite(dz))
            zn = z + jnp.where(ok, dz, 0.)
            r2, J2 = evalJ(zn, p)
            rn2 = jnp.linalg.norm(r2)
            acc = ok & jnp.isfinite(rn2) & (rn2 < rn)
            lam2 = jnp.where(acc, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            sel = lambda a, b: jnp.where(acc, a, b)
            return sel(zn, z), sel(r2, r), sel(J2, J), sel(rn2, rn), lam2, (~acc).astype(jnp.int32)

        def predict(k, zv, zp, zp2, p):
            ze = 2. * zv - zp
            zq = jnp.where(k >= 2, 3. * zv - 3. * zp + zp2, ze)
            cand = jnp.stack((zv, ze, zq))
            rs = jax.vmap(lambda z: jnp.linalg.norm(res(z, p)))(cand)
            return cand[jnp.argmin(jnp.where(jnp.isfinite(rs), rs, jnp.inf))]

        def make_step(nsweep):
            def step(carry, k):
                zv, zp, zp2, lam = carry
                p = A @ coef(zv, hp)
                z = predict(k, zv, zp, zp2, p)
                r, J = evalJ(z, p)
                rn = jnp.linalg.norm(r)
                rej = jnp.int32(0)
                for _ in range(nsweep):
                    z, r, J, rn, lam, rj_ = sweep(z, r, J, rn, lam, p)
                    rej = rej + rj_
                gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * rn + 1e-300)
                reason = jnp.where(jnp.isfinite(rn), jnp.where(gn <= gtol, 4, 0), 3).astype(jnp.int32)
                return (z, zv, zp, lam), (z, rn, jnp.int32(nsweep), reason, gn, rej)
            return step

        carry = (z0, z0, z0, jnp.asarray(1e-6, dtype=jnp.float64))
        if adaptive_first:
            lmA = C.make_fused_lm(evalJ, K, 50, trust, gtol, clip=True)

            def astep(carry, k):
                zv, zp, zp2, lam = carry
                p = A @ coef(zv, hp)
                zi = predict(k, zv, zp, zp2, p)
                z2, rn, it, reason, gn, rej, lam2 = lmA(zi, p, 1e-12 * jnp.linalg.norm(t), lam)
                return (z2, zv, zp, lam2), (z2, rn, it, reason, gn, rej)
            carry, o1 = jax.lax.scan(astep, carry, jnp.arange(adaptive_first))
            _, o2 = jax.lax.scan(make_step(1), carry, jnp.arange(adaptive_first, steps), unroll=unroll)
        else:
            carry, o1 = jax.lax.scan(make_step(3), carry, jnp.arange(2), unroll=True)
            _, o2 = jax.lax.scan(make_step(1), carry, jnp.arange(2, steps), unroll=unroll)
        zs, rn, it, reason, gn, rej = (jnp.concatenate((x, y)) for x, y in zip(o1, o2))
        internal = jnp.concatenate((z0[None], zs))
        Cc = jax.vmap(lambda z: coef(z, hp))(internal[::keep])
        X = data['Rq'] @ Cc.T
        out, off = 0., 0
        for b in data['Qb']:
            wdt = b.shape[1]
            out = out + b @ X[off:off + wdt]
            off += wdt
        return out.T, internal, it, reason, gn, rn, rej, ic
    return jax.jit(query), jax.jit(coef)
