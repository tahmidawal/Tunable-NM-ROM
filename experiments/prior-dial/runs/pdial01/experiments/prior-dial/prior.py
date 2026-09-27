"""The prior dial: one frozen checkpoint, one knob, lambda.

Arm (a) of the head ablation pins the bank coefficients to the neural head and
solves the weak residual for z alone. This cell keeps EVERY weight, the bank G,
K = 16, the initializer policy, dt, the stopping rule and the output contract,
and instead solves for the full bank coefficient vector c in R^R under a penalty
pulling it towards the head:

    min_{z, c}  || r_w(c) ||^2  +  lambda * || R_G (c - h_theta(z)) ||^2 ,

with G = Q_G R_G the thin QR already used by arms.py, so || R_G d || = || G d ||
and the penalty is the squared FIELD-norm distance from the head's own state.

Writing y = R_G (c - h_theta(z)) gives u = G c = G h_theta(z) + Q_G y and turns
the penalty into lambda ||y||^2. Everything online is formed in that split form,
so R_G^{-1} is never applied to a solved vector. The solver sees

    F(z, y) = [ r_w( h_theta(z) + R_G^{-1} y ) ;  sqrt(lambda) y ]  in R^{M+R}.

At lambda = infinity the y block has ZERO WIDTH, every y term is dropped at trace
time by a Python branch on Ry, and F is literally arm (a)'s residual in this same
function. That is what makes the lambda = infinity gate a bitwise statement.

Scaling of lambda: lambda = lambda_rel * sigma^2 with sigma = ||A R_G^{-1}||_2 =
||Phi^T Q_G||_2, the exact linear part of d r_w / d y. It is state independent
because the 1/(1 + dt nu lam_j) row scaling of the weak residual cancels the
diffusion term exactly; on Burgers both Phi and Q_G have orthonormal columns, so
sigma is the largest principal cosine between the test-mode span and the bank
span. lambda_rel = 1 balances the strongest linear residual response against the
prior. See DESIGN.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'head-ablation'))

import engines as e          # noqa: E402
import sep_common as sc      # noqa: E402
import arms as A             # noqa: E402


# ------------------------------------------------------------- operators ----

def build_prior_operators(bank, L, M, quadrature, Ry, Qg, Rg, Zcoef=None, m=None,
                          params=None, eq_seed=20259, candidate_cap=8192, fit_states=64):
    """Exact linear weak terms, the y-space companions, and either the retained
    decoder-output NNLS rule or the exact dense grid sum.

    The empirical-quadrature block is `arms.build_operators`' block verbatim, with
    the neural head and the checkpoint's own code table, so at M = 4K it is arm
    (a)'s own rule and not a refit of it.
    """
    t0 = time.perf_counter()
    D = bank.dim
    G = bank.on_grid(L)
    jax.block_until_ready(G)
    Phi, lam, mode_ids = e.modes(L, M)
    P = jnp.asarray(Phi)
    Adata = P.T @ G
    info = dict(intervals=L, interior_unknowns=(L - 1) ** 2, dim=D, M=int(M),
                quadrature=quadrature, mode_ids_count=int(len(mode_ids)),
                correction_dimension=int(Ry))
    data = dict(A=Adata, lam=jnp.asarray(lam), G=G)
    # y-space companions. Ay = A R_G^{-1} = Phi^T Q_G is formed directly from the
    # orthonormal factor rather than by a triangular solve with R_G, so nothing
    # depends on cond(R_G).
    if Ry > 0:
        assert Ry == D, (Ry, D)
        data['Qg'] = Qg
        data['Ay'] = P.T @ Qg
        sigma = float(jnp.linalg.norm(data['Ay'], 2))
    else:
        sigma = float(jnp.linalg.norm(P.T @ Qg, 2))
    info['sigma'] = sigma
    info['sigma_definition'] = 'largest singular value of A R_G^{-1} = Phi^T Q_G'

    if quadrature == 'dense':
        data['Phi'] = P
        info.update(m=None, eq_relative_fit=None)
    else:
        assert m is not None and Zcoef is not None and params is not None
        head = lambda z: sc.head(params, z)
        rng = np.random.default_rng(eq_seed)
        cp = np.sort(rng.choice((L - 1) ** 2, min(candidate_cap, (L - 1) ** 2), replace=False))
        fit = np.sort(rng.choice(len(Zcoef), min(fit_states, len(Zcoef)), replace=False))
        adv = jax.jit(lambda g, z: e.spatial(g @ head(z), L)[0])
        rows, targets = [], []
        for i in fit:
            n = adv(G, jnp.asarray(Zcoef[i]))
            targets.append(np.asarray(P.T @ n))
            rows.append(Phi[cp].T * np.asarray(n)[cp])
        design = np.concatenate(rows)
        b = np.concatenate(targets)
        scale = np.linalg.norm(design, axis=1) + 1e-300
        design /= scale[:, None]
        b /= scale
        supp, w = e.nnls_capped(design, b, m)
        if len(supp) < m:
            rest = np.setdiff1d(np.arange(len(cp)), supp)
            supp = np.concatenate((supp, rest[np.argsort(-np.abs(design[:, rest]).mean(0))[:m - len(supp)]]))
        w, _ = scipy.optimize.nnls(design[:, supp], b, maxiter=10 * len(supp))
        pos = cp[supp]
        ij = np.stack(np.unravel_index(pos, (L - 1, L - 1)), 1) + 1
        G5 = bank.stencil(ij, L)
        data['G5'] = G5
        data['Pq'] = jnp.asarray(Phi[pos] * w[:, None])
        if Ry > 0:
            # G5 R_G^{-1}: a small triangular solve, (m*5) x R.
            flat = G5.reshape(-1, D)
            data['G5y'] = jax.scipy.linalg.solve_triangular(
                Rg.T, flat.T, lower=True).T.reshape(G5.shape)
        info.update(m=int(m), eq_seed=int(eq_seed), eq_actual_candidates=int(len(cp)),
                    eq_fit_rows=fit.tolist(), eq_indices=pos.tolist(), eq_weights=w.tolist(),
                    eq_relative_fit=float(np.linalg.norm(design[:, supp] @ w - b) / np.linalg.norm(b)))
    info['setup_seconds'] = time.perf_counter() - t0
    info['array_bytes'] = int(sum(x.nbytes for x in jax.tree_util.tree_leaves(data)))
    return data, info


# ------------------------------------------------------- weak residual ------

def make_weak(params, K, Ry, L, dt, quadrature):
    """The penalized weak residual. Ry = 0 drops every y term at trace time."""
    if quadrature == 'eq':
        def state(h, y, data):
            us = jnp.einsum('msr,r->ms', data['G5'], h)
            if Ry:
                us = us + jnp.einsum('msr,r->ms', data['G5y'], y)
            c, xp, xm, yp, ym = [us[:, i] for i in range(5)]
            return c * L * (jnp.where(c > 0, c - xm, xp - c) + jnp.where(c > 0, c - ym, yp - c))

        def project(adv, data):
            return data['Pq'].T @ adv
    else:
        def state(h, y, data):
            u = data['G'] @ h
            if Ry:
                u = u + data['Qg'] @ y
            return e.spatial(u, L)[0]

        def project(adv, data):
            return data['Phi'].T @ adv

    def coefficients(w, data):
        h = sc.head(params, w[:K])
        ah = data['A'] @ h
        if Ry:
            ah = ah + data['Ay'] @ w[K:]
        return h, ah

    def prev_projection(w, data):
        return coefficients(w, data)[1]

    def weak(w, prev, nu, data, sl):
        h, ah = coefficients(w, data)
        y = w[K:]
        adv = state(h, y, data)
        lam = data['lam']
        r = (ah - prev + dt * (project(adv, data) + nu * lam * ah)) / (1 + dt * nu * lam)
        if Ry:
            return jnp.concatenate((r, sl * y))
        return r

    return weak, prev_projection


def decode(params, K, Ry, L):
    def field(w, data):
        u = data['G'] @ sc.head(params, w[:K])
        if Ry:
            u = u + data['Qg'] @ w[K:]
        return u
    return field


# ------------------------------------------------------------- solver ------

def make_block_lm(fun, budget, K, gtol=1e-6, linear='gj'):
    """`arms.make_stationary_lm` with a BLOCK trust region and a traced radius.

    The trust radius of arm (a) is 1% of the radius of the training code cloud: a
    latent-space quantity. The correction y lives in field-norm units, where a
    departure of order one is what a percent-level correction needs, so the radius
    is applied to the latent block and the correction block separately. With
    Ry = 0 the second block is empty, its norm is exactly zero, and the rule is
    arm (a)'s rule unchanged.
    """
    solve = e.gj_solve if linear == 'gj' else jnp.linalg.solve

    def lm(w0, args, tol, trust_z, trust_y):
        def evaluate(w):
            r = fun(w, *args)
            J = jax.jacfwd(fun)(w, *args)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        r, J, rn = evaluate(w0)
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(grad(r, J) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            w, r, J, rn, lam, it, reason = s
            H = J.T @ J
            g = J.T @ r
            dw = solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = (jnp.all(jnp.isfinite(dw))
                  & (jnp.linalg.norm(dw[:K]) <= trust_z)
                  & (jnp.linalg.norm(dw[K:]) <= trust_y))
            wn = w + jnp.where(ok, dw, 0.)
            rn2 = jnp.linalg.norm(fun(wn, *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: evaluate(wn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dw) <= 1e-14 * (1 + jnp.linalg.norm(w)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn2 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, wn, w), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        w, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body,
            (w0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return w, rn, it, reason, grad(r, J)
    return lm


# -------------------------------------------------------------- query ------

def make_prior_query(params, K, Ry, L, dt, quadrature, ic_budget=400, step_budget=180,
                     gtol=1e-6, linear='gj'):
    """The complete matched query: supplied dense field -> six dense fields.

    The initializer is arm (a)'s unchanged: nearest candidate among the
    checkpoint's own K-dimensional codes, then a K-dimensional Levenberg-Marquardt
    state fit against the fixed 48x48 Gauss rule. The correction starts at y = 0
    at every lambda, so lambda = infinity and every finite lambda enter the first
    time step from the same state.
    """
    wk, prev_projection = make_weak(params, K, Ry, L, dt, quadrature)
    field = decode(params, K, Ry, L)
    head0 = lambda z: sc.head(params, z)
    ic = A.make_stationary_lm(lambda z, y, R: R @ head0(z) - y, ic_budget, gtol=gtol, linear='gj')
    lm = make_block_lm(wk, step_budget, K, gtol, linear)
    nsteps = int(round(.25 / dt))
    stride = int(round(.05 / dt))

    def query(u0, nu, sl, trust_z, trust_y, data, cold):
        xy, wq, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * wq
        y = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ y)
        z, icrn, icit, icreason, icgn = ic(Zcand[idx], (y, R), 0.)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(wq))
        w0 = jnp.concatenate((z, jnp.zeros(Ry))) if Ry else z

        def step(carry, _):
            w, wprev = carry
            p = prev_projection(w, data)
            we = w + (w - wprev)
            r0 = jnp.linalg.norm(wk(w, p, nu, data, sl))
            re = jnp.linalg.norm(wk(we, p, nu, data, sl))
            wi = jnp.where(jnp.isfinite(re) & (re < r0), we, w)
            w2, rn, it, reason, gn = lm(wi, (p, nu, data, sl), 1e-9 * scale, trust_z, trust_y)
            return (w2, w), (w2, rn, it, reason, gn)

        _, (ws, rn, it, reason, gn) = jax.lax.scan(step, (w0, w0), None, length=nsteps)
        internal = jnp.concatenate((w0[None], ws))
        W = internal[::stride]
        us = jax.vmap(lambda w: field(w, data))(W)
        fields = jax.vmap(lambda u: e.output_field(u, L, L))(us)
        ynorm = (jnp.linalg.norm(internal[:, K:], axis=1) if Ry
                 else jnp.zeros(internal.shape[0]))
        return (fields, it, rn, reason, W[:, :K], icit, icreason, internal[:, :K], gn, icgn,
                icrn, jnp.linalg.norm(ui), ynorm, jnp.linalg.norm(us, axis=1))
    return jax.jit(query)
