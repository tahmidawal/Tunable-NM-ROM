"""burgers-bank-knob: the nested bank truncation R' of the frozen R = 512 Burgers model, without retraining.

Three pieces, all deployment-time:

1. The ROTATED bank G' = G T (T from make_rotation.py, training data only) is held on the device as row blocks x
   nested COLUMN blocks whose edges are the R' ladder, so a model truncated at R' reads exactly the first R'
   columns with no slicing copy. `bank_apply` below replaces `hops.bank_apply` (it is patched into the module, so
   hfast / xfast / hops.dense_targets use it unchanged): a flat tuple of row blocks is the parent's bank, a tuple
   of tuples is the nested rotated bank.

2. Head + corrections at R' (arms q = 0 and q > 0): the model's coefficient vector c = h(z) + C y (length R) becomes
   a = L[:R'] c. Because the head's last layer and its linear skip are linear, L[:R'] is FOLDED into them
   (`fold_head`), and the directions become C' = L[:R'] C. hfast / xfast then run unchanged with R' in place of R.
   Off-grid bank values (stencil, Gauss points) come from `RotBank`, i.e. G(x) T[:, :R'].

3. The linear rung (q = R', head dropped): u = G'[:, :R'] a with a in R^{R'} the unknowns. Advection is still
   nonlinear, so every time step is the same fused Levenberg-Marquardt solve (hfast.make_fused_lm with all R'
   unknowns damped and clipped), the same predictor / damping carry-over / Cholesky / stopping rule, the same weak
   residual with the empirical-quadrature advection (or the exact one for the first `exact_steps` steps). The
   initial fit is the closed-form least-squares projection on the same Gauss-point state fit the head arms use
   (for a linear map the LM initial fit converges to exactly this). The Jacobian is analytic:
       J = A + dt S Pq^T (D . G5),   D = d a / d us  (pointwise upwind-flux derivative).
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import hops as H
import hfast as HF


# ------------------------------------------------------------ nested bank ----

def bank_apply(Gb, X):
    """G @ X. Gb: tuple of row blocks (parent) or tuple of tuples of column blocks (rotated, nested prefix)."""
    rows = []
    for g in Gb:
        if isinstance(g, (tuple, list)):
            acc, off = None, 0
            for cb in g:
                w = cb.shape[1]
                part = cb @ X[off:off + w]
                acc = part if acc is None else acc + part
                off += w
            assert off == X.shape[0], (off, X.shape)
            rows.append(acc)
        else:
            rows.append(g @ X)
    return jnp.concatenate(rows, axis=0)


H.bank_apply = bank_apply            # hfast, xfast and hops.dense_targets look it up through the module


def prefix(Grot, edges, Rp):
    """The first R' columns of the nested rotated bank, as the nested tuple bank_apply reads."""
    nb = list(edges).index(Rp)
    assert nb > 0, (edges, Rp)
    return tuple(tuple(rb[:nb]) for rb in Grot)


def project_nested(Grot, edges, Rp, sx, sy, L):
    """A = Phi^T G'[:, :R'] by the separable projection, one column block at a time."""
    nb = list(edges).index(Rp)
    return jnp.concatenate([H.project_bank(tuple(rb[j] for rb in Grot), sx, sy, L) for j in range(nb)], axis=1)


class RotBank:
    """Off-grid values of the rotated, truncated bank: G(x) T[:, :R']."""

    kind = 'coord'

    def __init__(self, base, T, Rp):
        self.base, self.T, self.dim = base, jnp.asarray(np.asarray(T)[:, :Rp]), int(Rp)

    def at(self, xy, chunk=8192):
        return self.base.at(xy, chunk=chunk) @ self.T

    def stencil(self, ij, L):
        g5 = self.base.stencil(ij, L)
        return jnp.einsum('msr,rp->msp', g5, self.T)


def fold_head(params, Lm, Rp):
    """Head parameters whose output is L[:R'] h(z): the last MLP layer and the linear skip are linear in their
    output, so the (R, R') map folds into them exactly. The bank-side parameters are untouched (the bank is passed
    explicitly everywhere)."""
    P = jnp.asarray(np.asarray(Lm)[:Rp].T)                 # (R, R')
    h = list(params['h'])
    w, b = h[-1]
    h[-1] = (w @ P, b @ P)
    return dict(params, h=h, h_lin=params['h_lin'] @ P)


# ------------------------------------------------------------ linear rung ----

def make_linear_query(Rp, L, dt, trust, exact_steps=0, step_budget=600, gtol=1e-6, ridge=1e-10, solver='chol',
                      tangent_chunks=1, clip=False, lam_carry=False, predictor='lin'):
    """Supplied dense field on GPU -> six dense GPU fields, for u = G'[:, :R'] a. Output tuple layout identical to
    hfast.make_query (slot 8/12/14 the LM exit gradient, slot 15 the rejected-step count per step). `data` carries
    A (M, R'), lam, G (nested prefix), G5 (m, 5, R'), Pq (m, M), and sx, sy when exact_steps > 0."""
    steps = int(round(.25 / dt))
    keep = int(round(.05 / dt))
    dadv = jax.vmap(jax.grad(lambda u: H.advect_points(u, L)))
    assert Rp % tangent_chunks == 0

    def res_q(w, prev, nu, data, tab):
        us = jnp.einsum('msr,r->ms', data['G5'], w)
        ah = data['A'] @ w
        lam = data['lam']
        return (ah - prev + dt * (data['Pq'].T @ H.advect_points(us, L) + nu * lam * ah)) / (1 + dt * nu * lam)

    def evalJ_q(w, args):
        prev, nu, data, tab = args
        us = jnp.einsum('msr,r->ms', data['G5'], w)
        ah = data['A'] @ w
        lam = data['lam']
        S = 1. / (1 + dt * nu * lam)
        r = (ah - prev + dt * (data['Pq'].T @ H.advect_points(us, L) + nu * lam * ah)) * S
        dA = jnp.einsum('ms,msr->mr', dadv(us), data['G5'])
        return r, data['A'] + (dt * S)[:, None] * (data['Pq'].T @ dA)

    def res_d(w, prev, nu, data, tab):
        adv = e.spatial(H.bank_apply(data['G'], w), L)[0].reshape(L - 1, L - 1)
        ah = data['A'] @ w
        lam = data['lam']
        return (ah - prev + dt * (H.sep_project(adv, data['sx'], data['sy'], L) + nu * lam * ah)) / (1 + dt * nu * lam)

    def evalJ_d(w, args):
        prev, nu, data, tab = args
        r, lin = jax.linearize(lambda ww: res_d(ww, prev, nu, data, tab), w)
        basis = jnp.eye(Rp).reshape(tangent_chunks, Rp // tangent_chunks, Rp)
        J = jax.lax.map(lambda B: jax.vmap(lin)(B), basis).reshape(Rp, -1).T
        return r, J

    lm_q = HF.make_fused_lm(evalJ_q, Rp, 0, step_budget, trust, gtol, ridge, solver, clip)
    lm_d = HF.make_fused_lm(evalJ_d, Rp, 0, step_budget, trust, gtol, ridge, solver, clip)

    def initialize(u0, data, cold, tab):
        xy, wq, Q, R, _, _, _ = cold
        ui = e.sample_field(u0, xy, L) * wq
        yv = Q.T @ ui
        w0 = jax.scipy.linalg.solve_triangular(R, yv, lower=False)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(wq))
        icrn = jnp.linalg.norm(R @ w0 - yv)
        z = jnp.zeros(())
        return w0, scale, jnp.int32(0), jnp.int32(1), z, icrn, jnp.linalg.norm(ui), z

    def make_step(res, lm, nu, scale, data, tab):
        def step(carry, k):
            wv, wprev, wprev2, lam0 = carry
            p = data['A'] @ wv
            we = wv + (wv - wprev)
            if predictor == 'quad':
                wq = jnp.where(k >= 2, 3. * wv - 3. * wprev + wprev2, we)
                cand = jnp.stack((wv, we, wq))
                rs = jax.vmap(lambda ww: jnp.linalg.norm(res(ww, p, nu, data, tab)))(cand)
                rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
                wi = cand[jnp.argmin(rs)]
            else:
                r0 = jnp.linalg.norm(res(wv, p, nu, data, tab))
                re = jnp.linalg.norm(res(we, p, nu, data, tab))
                wi = jnp.where(jnp.isfinite(re) & (re < r0), we, wv)
            w2, rn, it, reason, gn, rej, lam = lm(wi, (p, nu, data, tab), 1e-9 * scale, lam0)
            return (w2, wv, wprev, lam if lam_carry else lam0), (w2, rn, it, reason, gn, rej)
        return step

    def evolve(w0, nu, scale, data, tab):
        carry = (w0, w0, w0, jnp.asarray(1e-6, dtype=jnp.float64))
        if exact_steps:
            carry, o1 = jax.lax.scan(make_step(res_d, lm_d, nu, scale, data, tab), carry, jnp.arange(exact_steps))
            _, o2 = jax.lax.scan(make_step(res_q, lm_q, nu, scale, data, tab), carry, jnp.arange(exact_steps, steps))
            return tuple(jnp.concatenate((a, b)) for a, b in zip(o1, o2))
        _, out = jax.lax.scan(make_step(res_q, lm_q, nu, scale, data, tab), carry, jnp.arange(steps))
        return out

    def decode_fields(W, data, tab):
        U = H.bank_apply(data['G'], W.T).T.reshape(len(W), L - 1, L - 1)
        return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))

    def query(u0, nu, data, cold, tab):
        w0, scale, icit, icreason, icgn, icrn, uin, icgj = initialize(u0, data, cold, tab)
        ws, rn, it, reason, gn, rej = evolve(w0, nu, scale, data, tab)
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::keep]
        fields = decode_fields(W, data, tab)
        return (fields, it, rn, reason, W, icit, icreason, internal, gn, icgn, icrn, uin, gn, icgj, gn, rej)

    return jax.jit(query)
