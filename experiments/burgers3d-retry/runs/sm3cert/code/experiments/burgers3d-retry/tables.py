"""burgers3d-retry: lean per-mesh tables and the two query paths used by panel3.py.

Differences from burgers3d-span/common.build_mesh (same mathematics, less memory, so 257-node meshes fit one H200):
  * the ordered bank is stored TRANSPOSED as row blocks at the ladder edges, GT = G_hat^T (R x N); Q is never formed.
    The least-squares projection uses the Cholesky factor L of the Gram matrix G_hat^T G_hat (nested: the leading
    R' x R' block of L is the factor of the leading Gram block), so
        c0 = argmin ||G_hat[:, :R'] c - u0|| = L'^{-T} L'^{-1} (GT[:R'] u0),  fields = C GT[:R'].
    With the ordered bank's Gram condition number ~1 (lane 1: 1.002 at 129 nodes) this equals the QR projection.
  * Tsym (M_max x R x R) is built column-chunk by column-chunk with D^- applied to a 16-row block at a time.
Queries (common2's fsc and fsh, re-expressed on this layout):
  fsc : span, tensor rule, cached predictor (one tensor contraction per swept state), adaptive first 3 steps,
        optional float32 table (t32).
  fsh : head, tensor rule slice M = 4K, fixed sweeps, adaptive first 3 steps, nearest-code + LM initial fit.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'burgers3d-span'))
import common as C  # noqa: E402

DT = C.DT


def feature_rows(bank, T, x, chunk=1 << 16):
    """G_hat(x)^T as a device array (R, P), chunked over points."""
    f = jax.jit(lambda p, xx, T: (C.features(p, xx) @ T).T)
    Tj = jnp.asarray(T)
    return jnp.concatenate([f(bank, jnp.asarray(x[s:s + chunk]), Tj) for s in range(0, len(x), chunk)], 1)


def build_tables(n, bank, T, ladder, M_of, log=print, tensor=True):
    t0 = time.perf_counter()
    R = int(np.asarray(T).shape[1])
    x = C.interior_coords(n)
    edges = sorted(set([0] + list(ladder) + [R]))
    Tn = np.asarray(T)
    GTb = tuple(jax.block_until_ready(feature_rows(bank, Tn[:, a:b], x)) for a, b in zip(edges[:-1], edges[1:]))
    del x
    log(f'[mesh {n}] bank rows {[g.shape for g in GTb]} {time.perf_counter() - t0:.1f}s')
    gram = jax.jit(lambda a, b: a @ b.T)
    Gm = jnp.block([[gram(a, b) for b in GTb] for a in GTb])
    assert bool(jnp.all(jnp.isfinite(Gm)))
    L = jnp.linalg.cholesky(Gm)
    ev = jnp.linalg.eigvalsh(Gm)
    M_max = M_of(R)
    kxyz, lam = C.modes(n, M_max)
    idx = tuple(jnp.asarray(kxyz[:, a]) for a in range(3))
    phi = jax.jit(lambda rows: C.phiT(rows, n, idx))
    A = jnp.concatenate([phi(g[s:s + 16]) for g in GTb for s in range(0, g.shape[0], 16)], 0).T      # (M, R)
    log(f'[mesh {n}] Gram cond {float(jnp.sqrt(ev[-1] / ev[0])):.4f}, A {A.shape} {time.perf_counter() - t0:.1f}s')
    out = dict(n=n, R=R, edges=edges, GTb=GTb, L=L, A=A, lam=jnp.asarray(lam), kxyz=kxyz, M_max=M_max,
               gram_cond=float(jnp.sqrt(ev[-1] / ev[0])))
    if tensor:
        def rows(a, b):          # G_hat^T rows a..b (b - a <= 16), copied
            parts, off = [], 0
            for g in GTb:
                lo, hi = max(a - off, 0), min(b - off, g.shape[0])
                if lo < hi:
                    parts.append(g[lo:hi])
                off += g.shape[0]
            return jnp.concatenate(parts, 0)
        dmin = jax.jit(lambda r: C.dminus(r, n))
        prod = jax.jit(lambda gi, DGc: C.phiT(gi[None, :] * DGc, n, idx))                      # (c, M)
        cols = []
        for j0 in range(0, R, 16):
            DGc = dmin(rows(j0, min(j0 + 16, R)))                                                # (16, N)
            blk = []
            for i in range(R):
                blk.append(prod(rows(i, i + 1)[0], DGc))                                         # (16, M)
            cols.append(jnp.stack(blk, 0))                                                       # (R_i, 16, M)
            del DGc
        Tm = jnp.transpose(jnp.concatenate(cols, 1), (2, 0, 1))                                  # (M, R_i, R_j)
        del cols
        out['Tsym'] = jax.block_until_ready(Tm + jnp.transpose(Tm, (0, 2, 1)))
        del Tm
        log(f'[mesh {n}] tensor {out["Tsym"].shape} {time.perf_counter() - t0:.1f}s')
    return out


def rows_prefix(GTb, Rp):
    out, off = [], 0
    for g in GTb:
        if off >= Rp:
            break
        out.append(g if off + g.shape[0] <= Rp else g[:Rp - off])
        off += g.shape[0]
    assert off >= Rp
    return tuple(out)


def arm_data(tb, Rp, M, t32=False, head=None):
    assert Rp in tb['edges'], (Rp, tb['edges'])
    d = dict(GTb=rows_prefix(tb['GTb'], Rp), L=tb['L'][:Rp, :Rp], A=tb['A'][:M, :Rp], lam=tb['lam'][:M])
    full = M == tb['Tsym'].shape[0] and Rp == tb['Tsym'].shape[1]
    Ts = tb['Tsym'] if full else jnp.asarray(tb['Tsym'][:M, :Rp, :Rp])
    if t32:
        d['Ts32'] = Ts.astype(jnp.float32)
        if not full:
            del Ts
    else:
        d['Ts'] = Ts
    if head is not None:                       # library: training codes and their ordered-bank coefficients
        d['Z'] = jnp.asarray(head['Z'])
        H = jnp.asarray(head['H'])[:, :Rp]
        d['LH'] = H @ tb['L'][:Rp, :Rp]         # field-metric coordinates: ||L^T c - L^T c'|| = field distance
    return jax.tree_util.tree_map(lambda a: jax.block_until_ready(jnp.asarray(a)), d)


def project(u0, data):
    """(w0 = least-squares coefficients, s = L^{-1} G^T u0 whose norm is the projected energy)."""
    t = jnp.concatenate([g @ u0 for g in data['GTb']])
    s = jax.scipy.linalg.solve_triangular(data['L'], t, lower=True)
    return jax.scipy.linalg.solve_triangular(data['L'].T, s, lower=False), s


def decode(Cm, data):
    out, off = 0., 0
    for g in data['GTb']:
        w = g.shape[0]
        out = out + Cm[:, off:off + w] @ g
        off += w
    return out


def contract(data, w):
    if 'Ts32' in data:
        return jnp.einsum('mij,j->mi', data['Ts32'], w.astype(jnp.float32),
                          preferred_element_type=jnp.float32).astype(jnp.float64)
    return jnp.einsum('mij,j->mi', data['Ts'], w)


def _lm_step_parts(A, S, lam_, dt, nu):
    def rj(Ju, c, p):
        Ac = A @ c
        r = (Ac - p + dt * (0.5 * Ju @ c + nu * lam_ * Ac)) * S
        return r, A + (dt * S)[:, None] * Ju
    return rj


def make_fsc(n, Rp, M, dt=DT, gtol=1e-3, trust=jnp.inf, adaptive_first=3, unroll=True):
    steps, keep = int(round(0.25 / dt)), int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12

    def query(u0, nu, data, hp):
        A, lam_ = data['A'], data['lam']
        S = 1.0 / (1.0 + dt * nu * lam_)
        w0, s0 = project(u0, data)
        ic = jnp.stack([0., 4., 0., jnp.linalg.norm(u0) ** 2 - jnp.linalg.norm(s0) ** 2])
        rj = _lm_step_parts(A, S, lam_, dt, nu)

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
            Jun = contract(data, wn)
            r2, J2 = rj(Jun, wn, p)
            rn2 = jnp.linalg.norm(r2)
            acc = ok & jnp.isfinite(rn2) & (rn2 < rn)
            lam2 = jnp.where(acc, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            sel = lambda a, b: jnp.where(acc, a, b)
            return sel(wn, w), sel(Jun, Ju), sel(r2, r), sel(J2, J), sel(rn2, rn), lam2, (~acc).astype(jnp.int32)

        def predict(k, wv, wp, wp2, Jv, Jp, Jp2, p):
            we, Je = 2. * wv - wp, 2. * Jv - Jp
            wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
            Jq = jnp.where(k >= 2, 3. * Jv - 3. * Jp + Jp2, Je)
            cand, Ju3 = jnp.stack((wv, we, wq)), jnp.stack((Jv, Je, Jq))
            rs = jnp.linalg.norm(jax.vmap(lambda Ju, c: rj(Ju, c, p)[0])(Ju3, cand), axis=1)
            i = jnp.argmin(jnp.where(jnp.isfinite(rs), rs, jnp.inf))
            return cand[i], Ju3[i]

        def step(carry, k):
            wv, wp, wp2, Jv, Jp, Jp2, lam = carry
            p = A @ wv
            w, Ju = predict(k, wv, wp, wp2, Jv, Jp, Jp2, p)
            r, J = rj(Ju, w, p)
            w, Ju, r, J, rn, lam, rej = sweep(w, Ju, r, J, jnp.linalg.norm(r), lam, p)
            gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * rn + 1e-300)
            reason = jnp.where(jnp.isfinite(rn), jnp.where(gn <= gtol, 4, 0), 3).astype(jnp.int32)
            return (w, wv, wp, Ju, Jv, Jp, lam), (w, rn, jnp.int32(1), reason, gn, rej)

        lmA = C.make_fused_lm(lambda w, p_: rj(contract(data, w), w, p_), Rp, 50, trust, gtol, clip=True)

        def astep(carry, k):
            wv, wp, wp2, Jv, Jp, Jp2, lam = carry
            p = A @ wv
            wi, _ = predict(k, wv, wp, wp2, Jv, Jp, Jp2, p)
            w2, rn, it, reason, gn, rej, lam2 = lmA(wi, p, 1e-12 * jnp.linalg.norm(s0), lam)
            return (w2, wv, wp, contract(data, w2), Jv, Jp, lam2), (w2, rn, it, reason, gn, rej)

        J0 = contract(data, w0)
        carry = (w0, w0, w0, J0, J0, J0, jnp.asarray(1e-6, dtype=jnp.float64))
        carry, o1 = jax.lax.scan(astep, carry, jnp.arange(adaptive_first))
        _, o2 = jax.lax.scan(step, carry, jnp.arange(adaptive_first, steps), unroll=unroll)
        ws, rn, it, reason, gn, rej = (jnp.concatenate((x, y)) for x, y in zip(o1, o2))
        internal = jnp.concatenate((w0[None], ws))
        return decode(internal[::keep], data), internal, it, reason, gn, rn, rej, ic
    return jax.jit(query), jax.jit(lambda w, hp: w)


def make_fsh(n, Rp, M, K, dt=DT, gtol=1e-3, trust=jnp.inf, adaptive_first=3, fit_budget=200, unroll=True):
    steps, keep = int(round(0.25 / dt)), int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12
    coef = lambda z, hp: C.head(hp, z)[:Rp]

    def query(u0, nu, data, hp):
        A, lam_ = data['A'], data['lam']
        S = 1.0 / (1.0 + dt * nu * lam_)
        _, s0 = project(u0, data)                         # target in field-metric coordinates: L^T c ~ s0
        Lt = data['L'].T
        i0 = jnp.argmin(jnp.sum((data['LH'] - s0[None, :]) ** 2, axis=1))

        def fitJ(z, _):
            f = lambda zz: Lt @ coef(zz, hp) - s0
            return f(z), jax.jacfwd(f)(z)
        fit = C.make_fused_lm(fitJ, K, fit_budget, jnp.inf, 1e-6, clip=False)
        z0, rn0, it0, reason0, gn0, _, _ = fit(data['Z'][i0], None, 0., 1e-6)
        ic = jnp.stack([it0.astype(jnp.float64), reason0.astype(jnp.float64), gn0, rn0])
        rj = _lm_step_parts(A, S, lam_, dt, nu)

        def evalJ(z, p):
            c, Jh = coef(z, hp), jax.jacfwd(lambda zz: coef(zz, hp))(z)
            r, Jc = rj(contract(data, c), c, p)
            return r, Jc @ Jh

        def res(z, p):
            c = coef(z, hp)
            return rj(contract(data, c), c, p)[0]

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

        def step(carry, k):
            zv, zp, zp2, lam = carry
            p = A @ coef(zv, hp)
            z = predict(k, zv, zp, zp2, p)
            r, J = evalJ(z, p)
            z, r, J, rn, lam, rej = sweep(z, r, J, jnp.linalg.norm(r), lam, p)
            gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * rn + 1e-300)
            reason = jnp.where(jnp.isfinite(rn), jnp.where(gn <= gtol, 4, 0), 3).astype(jnp.int32)
            return (z, zv, zp, lam), (z, rn, jnp.int32(1), reason, gn, rej)

        lmA = C.make_fused_lm(evalJ, K, 50, trust, gtol, clip=True)

        def astep(carry, k):
            zv, zp, zp2, lam = carry
            p = A @ coef(zv, hp)
            zi = predict(k, zv, zp, zp2, p)
            z2, rn, it, reason, gn, rej, lam2 = lmA(zi, p, 1e-12 * jnp.linalg.norm(s0), lam)
            return (z2, zv, zp, lam2), (z2, rn, it, reason, gn, rej)

        carry = (z0, z0, z0, jnp.asarray(1e-6, dtype=jnp.float64))
        carry, o1 = jax.lax.scan(astep, carry, jnp.arange(adaptive_first))
        _, o2 = jax.lax.scan(step, carry, jnp.arange(adaptive_first, steps), unroll=unroll)
        zs, rn, it, reason, gn, rej = (jnp.concatenate((x, y)) for x, y in zip(o1, o2))
        internal = jnp.concatenate((z0[None], zs))
        Cm = jax.vmap(lambda z: coef(z, hp))(internal[::keep])
        return decode(Cm, data), internal, it, reason, gn, rn, rej, ic
    return jax.jit(query), jax.jit(coef)


def make_rho(n, M, kxyz):
    """rho(u) = ||rule(u) - Phi^T a_upwind(u)|| / ||Phi^T a_upwind(u)|| over the arm's M tests, u = c GT[:R'] on every
    interior node (exact side: the FOM's sign-upwind stencil and a 3D DST); rule as deployed (float32 table if the
    arm has one). Also the minimum decoded value."""
    idx = tuple(jnp.asarray(np.asarray(kxyz)[:M, a]) for a in range(3))

    def one(c, data):
        u = decode(c[None], data)[0]
        ex = C.phiT(C.upwind(u, n)[None], n, idx)[0]
        ru = 0.5 * contract(data, c) @ c
        return jnp.linalg.norm(ru - ex) / jnp.maximum(jnp.linalg.norm(ex), 1e-300), jnp.min(u)

    @jax.jit
    def batch(Cs, data):
        return jax.lax.map(lambda c: one(c, data), Cs)
    return batch
