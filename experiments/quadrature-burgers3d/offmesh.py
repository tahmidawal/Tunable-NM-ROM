"""quadrature-burgers3d: off-mesh advection for the frozen Burgers 3D span model, and the solver that uses it.

The reduced model is unchanged from burgers3d-retry (vendor/burgers3d-retry/tables.py, make_fsc): span u = G_hat[:, :R'] c,
LSPG on the M lowest discrete sine tests Phi (mesh-orthonormal), scaled residual
    r = S * ( A c - A c_prev + dt ( adv(c) + nu lam * A c ) ),  A = Phi^T G_hat (exact, by DST),  S = 1/(1 + dt nu lam),
cached-predictor fixed-sweep Levenberg-Marquardt (adaptive LM on the first 3 steps, then one sweep per step).
ONLY adv(c) and its Jacobian Ju(c) = d adv / dc change. Every rule here is a symmetric quadratic form in c, so
adv(c) = 0.5 Ju(c) c and Ju is linear in c (the cached predictor stays exact):

  tensor   adv_m = 0.5 c^T Tsym_m c,  Tsym_m = Phi_m^T (G_i * D^- G_j) + (i <-> j)     (the incumbent; burgers3d-retry)
  offmesh  adv_m = sum_q P_qm (B c)_q (D c)_q,  B = G_hat(x_q), D = (d_x + d_y + d_z) G_hat(x_q) (analytic, one JVP),
           P_qm = (n-1)^{3/2} w_q psi_m(x_q),  psi_m = 2^{3/2} sin(pi kx x) sin(pi ky y) sin(pi kz z)
           Ju(c) = P^T ( diag(D c) B + diag(B c) D )                                   (Hari's "point" form, hybrid)
  dense    adv = Phi^T a_upwind(G_hat c) on every interior node, the FOM's sign-upwind stencil (Ju by JVP per column);
           piecewise quadratic, so Ju is NOT linear in c: the predictor re-evaluates Ju at each candidate.

The off-mesh term is a quadrature of the continuum advection u (u_x + u_y + u_z); it differs from the mesh stencil by the
stencil's O(h) consistency gap, which is the point of measuring both a continuum and a mesh target.
All large arrays are explicit jit arguments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'burgers3d-span'))
sys.path.insert(0, str(HERE / 'vendor' / 'burgers3d-retry'))
import common as C  # noqa: E402
import tables as TB  # noqa: E402

DT = C.DT


# ------------------------------------------------------------------ point blocks
def point_blocks(bank, T, X, chunk=1 << 15):
    """(B, D) at points X (m, 3): B = G_hat(X) (m, R), D = sum of the three coordinate derivatives (m, R).
    One forward-mode JVP with tangent (1,1,1) per point (points are independent rows of features)."""
    Tj = jnp.asarray(T)

    @jax.jit
    def f(p, x, T):
        g = lambda xx: C.features(p, xx) @ T
        return jax.jvp(g, (x,), (jnp.ones_like(x),))
    Bs, Ds = [], []
    for s in range(0, len(X), chunk):
        b, d = f(bank, jnp.asarray(X[s:s + chunk]), Tj)
        Bs.append(b)
        Ds.append(d)
    return jnp.concatenate(Bs, 0), jnp.concatenate(Ds, 0)


def psi(X, kxyz):
    """psi_m(x_q) (m, M), L2-orthonormal sines on the unit cube."""
    X = jnp.asarray(X)
    k = jnp.asarray(np.asarray(kxyz), dtype=jnp.float64)
    out = 2.0 ** 1.5 * jnp.ones((X.shape[0], k.shape[0]))
    for a in range(3):
        out = out * jnp.sin(jnp.pi * X[:, a:a + 1] * k[None, :, a])
    return out


def test_block(n, X, w, kxyz):
    """P (m, M) = (n-1)^{3/2} w_q psi_m(x_q): the off-mesh rule reproducing the mesh-tested term."""
    return (n - 1) ** 1.5 * jnp.asarray(w)[:, None] * psi(X, kxyz)


def offmesh_tables(n, bank, T, X, w, kxyz, chunk=1 << 15):
    B, D = point_blocks(bank, T, X, chunk)
    P = jax.block_until_ready(test_block(n, X, w, kxyz))
    return dict(B=B, D=D, P=P)


# ------------------------------------------------------------------ advection rules: contract(data, w) -> Ju (M, R')
def contract_tensor(data, w):
    return jnp.einsum('mij,j->mi', data['Ts'], w)


def contract_offmesh(data, w):
    B, D, P = data['B'], data['D'], data['P']
    Bw, Dw = B @ w, D @ w
    return P.T @ (Dw[:, None] * B + Bw[:, None] * D)


def adv_offmesh_batch(data, Cs):
    """adv for a batch of coefficient rows Cs (S, R') -> (S, M)."""
    return ((Cs @ data['B'].T) * (Cs @ data['D'].T)) @ data['P']


def make_contract_dense(n, M, kxyz, chunk=16):
    """Exact sign-upwind advection on every interior node (the FOM's stencil), Jacobian by JVP column by column."""
    idx = tuple(jnp.asarray(np.asarray(kxyz)[:M, a]) for a in range(3))

    def contract(data, w):
        u = TB.decode(w[None], data)[0]
        rows = jnp.concatenate(data['GTb'], 0)                                   # (R', N)
        Rp = rows.shape[0]
        pad = (-Rp) % chunk
        rows_p = jnp.concatenate((rows, jnp.zeros((pad, rows.shape[1]))), 0).reshape(-1, chunk, rows.shape[1])

        def blk(G):
            cols = jax.vmap(lambda g: jax.jvp(lambda v: C.upwind(v, n), (u,), (g,))[1])(G)
            return C.phiT(cols, n, idx)                                         # (chunk, M)
        J = jax.lax.map(blk, rows_p).reshape(-1, M)[:Rp]
        return J.T
    return contract


# ------------------------------------------------------------------ solver (tables.make_fsc with the rule injected)
def make_fsc_rule(n, Rp, M, contract, dt=DT, gtol=1e-3, trust=jnp.inf, adaptive_first=3, linear_jac=True,
                  unroll=True):
    """Identical to vendor tables.make_fsc except that contract(data, w) is a parameter, and, when the rule's Jacobian
    is not linear in c (linear_jac=False, dense upwind), the predictor evaluates Ju at each candidate."""
    steps, keep = int(round(0.25 / dt)), int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12

    def query(u0, nu, data, hp):
        A, lam_ = data['A'], data['lam']
        S = 1.0 / (1.0 + dt * nu * lam_)
        w0, s0 = TB.project(u0, data)
        ic = jnp.stack([0., 4., 0., jnp.linalg.norm(u0) ** 2 - jnp.linalg.norm(s0) ** 2])
        rj = TB._lm_step_parts(A, S, lam_, dt, nu)

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
            we = 2. * wv - wp
            wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
            cand = jnp.stack((wv, we, wq))
            if linear_jac:
                Je = 2. * Jv - Jp
                Jq = jnp.where(k >= 2, 3. * Jv - 3. * Jp + Jp2, Je)
                Ju3 = jnp.stack((Jv, Je, Jq))
            else:
                Ju3 = jax.lax.map(lambda c: contract(data, c), cand)
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
        return TB.decode(internal[::keep], data), internal, it, reason, gn, rn, rej, ic
    return jax.jit(query)


# ------------------------------------------------------------------ targets for rho
def make_mesh_target(n, M, kxyz):
    """Phi^T a_upwind(G_hat c) over every interior node (the FOM's stencil), batch over coefficient rows."""
    idx = tuple(jnp.asarray(np.asarray(kxyz)[:M, a]) for a in range(3))

    @jax.jit
    def batch(Cs, data):
        def one(c):
            u = TB.decode(c[None], data)[0]
            return C.phiT(C.upwind(u, n)[None], n, idx)[0], jnp.min(u)
        return jax.lax.map(one, Cs)
    return batch


def continuum_adv(n, bank, T, X, w, kxyz, Cs, chunk=1 << 16):
    """adv of a fine rule (X, w) for coefficient rows Cs (S, R'), accumulated over point chunks (no m x R storage).
    Uses the first R' = Cs.shape[1] columns of the ordered bank and the first len(kxyz) tests."""
    Rp = Cs.shape[1]
    Tj = jnp.asarray(np.asarray(T)[:, :Rp])
    Cj = jnp.asarray(Cs)

    @jax.jit
    def part(p, x, ww, T, Cs):
        g = lambda xx: C.features(p, xx) @ T
        Bq, Dq = jax.jvp(g, (x,), (jnp.ones_like(x),))
        P = (n - 1) ** 1.5 * ww[:, None] * psi(x, kxyz)
        return ((Cs @ Bq.T) * (Cs @ Dq.T)) @ P
    out = 0.
    for s in range(0, len(X), chunk):
        out = out + part(bank, jnp.asarray(X[s:s + chunk]), jnp.asarray(w[s:s + chunk]), Tj, Cj)
    return out
