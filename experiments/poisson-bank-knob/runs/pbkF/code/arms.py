"""Matched coefficient-map arms in one frozen spatial bank, plus classical POD-LSPG.

Every arm writes the reduced state as

    u(z) = B h(z),      B in R^{n x D},   h : R^K -> R^D,

and solves the SAME overdetermined weak residual for z. Only the pair (B, h)
changes between arms:

  a  neural    B = frozen separable bank G (D=R=512), h = retained MLP head
  b  linear    B = G,  h(z) = c + W z                      (POD inside the bank)
  c  quadratic B = G,  h(z) = c + W z + Q vech(z z^T)      (quadratic manifold in the bank)
  d  free      B = G,  h = identity on R^R                 (bank projection ceiling)
  e  pod       B = V_k' (classical POD of truth snapshots), h = identity on R^{k'}

The weak residual, test modes, time discretization, initializer policy, stopping
rule and output contract are shared. The only necessary deviations are recorded
in the driver: arm (d) needs M > R tests, and arms above the matched rank use a
dense (exact) advection quadrature because a nonnegative-least-squares rule with
m = 4M points is not constructible inside the job budget.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.optimize
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import sep_common as sc


# ----------------------------------------------------------------- banks ----

class CoordBank:
    """The frozen coordinate-network bank. Continuous in x; boundary-vanishing."""

    kind = 'coord'

    def __init__(self, params, K, R):
        self.params, self.K, self.R = params, int(K), int(R)
        self.dim = int(R)
        self._grid = {}

    def at(self, xy, chunk=8192):
        return sc.SeparableDecoder(self.params, self.K, self.R).feat_at(np.asarray(xy), chunk=chunk)

    def on_grid(self, L):
        # Memoized: every arm sharing this bank must share the identical array,
        # both for provenance and to avoid holding several copies on the device.
        if L not in self._grid:
            self._grid[L] = self.at(e.coords(L))
        return self._grid[L]

    def stencil(self, ij, L):
        off = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
        pts = ((ij[:, None, :] + off[None, :, :]) / L).reshape(-1, 2)
        return self.at(pts).reshape(len(ij), 5, self.dim)


class GridBank:
    """A grid-resident bank (classical POD modes). Off-grid values are the same
    aligned bilinear interpolation that the supplied input field already gets."""

    kind = 'grid'

    def __init__(self, V, L):
        # V: (n, D) interior modes on the L-grid.
        self.L = int(L)
        self.V = jnp.asarray(V)
        self.dim = int(V.shape[1])
        self.pad = jnp.pad(self.V.T.reshape(self.dim, L - 1, L - 1), ((0, 0), (1, 1), (1, 1)))

    def at(self, xy, chunk=8192):
        xy = jnp.asarray(np.asarray(xy))
        return jax.vmap(lambda f: e.sample_field(f, xy, self.L), out_axes=1)(self.pad)

    def on_grid(self, L):
        assert L == self.L
        return self.V

    def stencil(self, ij, L):
        assert L == self.L
        off = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
        idx = ij[:, None, :] + off[None, :, :]
        return self.pad[:, idx[..., 0], idx[..., 1]].transpose(1, 2, 0)


# ----------------------------------------------------------------- heads ----

def neural_head(params):
    return lambda z: sc.head(params, z)


def linear_head(hp):
    c, W = hp['c'], hp['W']
    return lambda z: c + W @ z


def quadratic_head(hp):
    c, W, Q, iu, ju = hp['c'], hp['W'], hp['Q'], hp['iu'], hp['ju']
    return lambda z: c + W @ z + Q @ (z[iu] * z[ju])


def identity_head():
    return lambda z: z


def vech_index(K):
    iu, ju = np.triu_indices(K)
    return jnp.asarray(iu), jnp.asarray(ju)


# ------------------------------------------------------------- map fitting ---

def whiten(B):
    """Thin QR of the bank: ||B d|| == ||Rb d||, so coefficient least squares in
    the Rb metric is exactly field-metric least squares."""
    Qb, Rb = jnp.linalg.qr(jnp.asarray(B), mode='reduced')
    return Qb, Rb


def fit_linear_map(coef, Rb, k, seed=0):
    """Field-metric POD of bank coefficients: the optimal rank-k affine map."""
    coef = jnp.asarray(coef)
    centre = jnp.mean(coef, axis=0)
    tilde = (coef - centre) @ Rb.T
    _, s, Vt = jnp.linalg.svd(tilde, full_matrices=False)
    Wt = Vt[:k].T                                     # (D, k), field-orthonormal
    W = jnp.linalg.solve(Rb, Wt)
    Z = tilde @ Wt                                    # (Ns, k) latent coordinates
    total = float(jnp.sum(s ** 2))
    kept = float(jnp.sum(s[:k] ** 2))
    info = dict(singular_values=np.asarray(s[:max(k, 256)]).tolist(),
                retained_energy_fraction=kept / total if total > 0 else 1.0,
                relative_projection_rms=float(np.sqrt(max(total - kept, 0.) / max(total, 1e-300))),
                seed=int(seed))
    return dict(c=centre, W=W), np.asarray(Z), Wt, info


def fit_quadratic_map(coef, Rb, k, Z, Wt, seed=0, holdout=0.2,
                      gammas=(0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)):
    """Quadratic-manifold correction fitted to the rank-k POD residual, with the
    ridge chosen on a seeded held-out split of the same snapshots."""
    coef = jnp.asarray(coef)
    centre = jnp.mean(coef, axis=0)
    tilde = (coef - centre) @ Rb.T
    resid = tilde - jnp.asarray(Z) @ Wt.T             # (Ns, D) in whitened coords
    iu, ju = np.triu_indices(k)
    Pi = jnp.asarray(np.asarray(Z)[:, iu] * np.asarray(Z)[:, ju])   # (Ns, P)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Z))
    ntest = max(1, int(round(holdout * len(Z))))
    te, tr = order[:ntest], order[ntest:]
    A = Pi[tr]
    scale = float(jnp.trace(A.T @ A) / A.shape[1])
    best = None
    trace = []
    for g in gammas:
        Qt = jnp.linalg.solve(A.T @ A + g * scale * jnp.eye(A.shape[1]), A.T @ resid[tr])
        err = float(jnp.linalg.norm(Pi[te] @ Qt - resid[te]) / max(float(jnp.linalg.norm(tilde[te])), 1e-300))
        trace.append(dict(gamma=float(g), heldout_relative=err))
        if best is None or err < best[1]:
            best = (g, err, Qt)
    g, err, _ = best
    Qt = jnp.linalg.solve(Pi.T @ Pi + g * scale * jnp.eye(Pi.shape[1]), Pi.T @ resid)
    Q = jnp.linalg.solve(Rb, Qt.T)                    # (D, P)
    W = jnp.linalg.solve(Rb, Wt)
    full = float(jnp.linalg.norm(Pi @ Qt - resid) / max(float(jnp.linalg.norm(tilde)), 1e-300))
    info = dict(ridge=float(g), heldout_relative=err, heldout_fraction=holdout,
                ridge_trace=trace, in_sample_residual_relative=full, seed=int(seed),
                terms=int(Pi.shape[1]))
    hp = dict(c=centre, W=W, Q=Q, iu=jnp.asarray(iu), ju=jnp.asarray(ju))
    return hp, info


# ------------------------------------------------------- reduced operators ---

def build_operators(bank, L, M, quadrature, Zcoef=None, m=None, eq_seed=20259,
                    candidate_cap=8192, fit_states=64, head=None):
    """Exact linear weak terms plus either a decoder-output NNLS rule or the
    exact dense advection quadrature. Mirrors engines.build_rom row for row."""
    t0 = time.perf_counter()
    D = bank.dim
    G = bank.on_grid(L)
    jax.block_until_ready(G)
    Phi, lam, mode_ids = e.modes(L, M)
    P = jnp.asarray(Phi)
    A = P.T @ G
    info = dict(intervals=L, interior_unknowns=(L - 1) ** 2, dim=D, M=int(M),
                quadrature=quadrature, mode_ids_count=int(len(mode_ids)))
    data = dict(A=A, lam=jnp.asarray(lam), G=G)
    if quadrature == 'dense':
        data['Phi'] = P
        info.update(m=None, eq_relative_fit=None)
    else:
        assert m is not None and head is not None and Zcoef is not None
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
        data['G5'] = bank.stencil(ij, L)
        data['Pq'] = jnp.asarray(Phi[pos] * w[:, None])
        info.update(m=int(m), eq_seed=int(eq_seed), eq_actual_candidates=int(len(cp)),
                    eq_fit_rows=fit.tolist(), eq_indices=pos.tolist(), eq_weights=w.tolist(),
                    eq_relative_fit=float(np.linalg.norm(design[:, supp] @ w - b) / np.linalg.norm(b)))
    info['setup_seconds'] = time.perf_counter() - t0
    info['array_bytes'] = int(sum(x.nbytes for x in jax.tree_util.tree_leaves(data)))
    return data, info


def build_cold(bank, head, Zcand, axis_points=48):
    """The retained fixed-Gauss state-fitting initializer, for any bank/head."""
    t0 = time.perf_counter()
    axis, weights = np.polynomial.legendre.leggauss(axis_points)
    axis = (axis + 1) / 2
    weights = weights / 2
    xy = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    w = np.sqrt(np.outer(weights, weights).ravel())
    Gi = bank.at(jnp.asarray(xy)) * jnp.asarray(w)[:, None]
    Q, R = jnp.linalg.qr(Gi, mode='reduced')
    H = jax.jit(jax.vmap(head))(jnp.asarray(Zcand))
    Hrot = H @ R.T
    cold = jax.block_until_ready((jnp.asarray(xy), jnp.asarray(w), Q, R, Hrot,
                                  jnp.sum(Hrot * Hrot, 1), jnp.asarray(Zcand)))
    info = dict(rule='fixed_gauss', axis_points=axis_points, points=len(xy),
                candidates=int(len(Zcand)), setup_seconds=time.perf_counter() - t0,
                array_bytes=int(sum(x.nbytes for x in cold)))
    return cold, info


# ------------------------------------------------------------ weak residual --

def weak_eq(z, prev, nu, data, head, L, dt):
    h = head(z)
    us = jnp.einsum('msr,r->ms', data['G5'], h)
    c, xp, xm, yp, ym = [us[:, i] for i in range(5)]
    adv = c * L * (jnp.where(c > 0, c - xm, xp - c) + jnp.where(c > 0, c - ym, yp - c))
    ah = data['A'] @ h
    lam = data['lam']
    return (ah - prev + dt * (data['Pq'].T @ adv + nu * lam * ah)) / (1 + dt * nu * lam)


def weak_dense(z, prev, nu, data, head, L, dt):
    h = head(z)
    adv, _ = e.spatial(data['G'] @ h, L)
    ah = data['A'] @ h
    lam = data['lam']
    return (ah - prev + dt * (data['Phi'].T @ adv + nu * lam * ah)) / (1 + dt * nu * lam)


def weak_fn(quadrature):
    return weak_eq if quadrature == 'eq' else weak_dense


# ------------------------------------------------------------------ solver ---

def make_stationary_lm(fun, budget, trust=np.inf, gtol=1e-6, linear='gj'):
    """accuracy_paths.make_stationary_lm with the small linear solve selectable.

    `gj` is the incumbent unpivoted Gauss-Jordan used by every retained Burgers
    result; it unrolls one graph level per unknown, so arms above 64 unknowns use
    a pivoted dense solve instead. That is a more, not less, accurate step."""
    solve = e.gj_solve if linear == 'gj' else jnp.linalg.solve

    def lm(z0, args, tol):
        def evaluate(z):
            r = fun(z, *args)
            J = jax.jacfwd(fun)(z, *args)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        r, J, rn = evaluate(z0)
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(grad(r, J) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, r, J, rn, lam, it, reason = s
            H = J.T @ J
            g = J.T @ r
            dz = solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz)) & (jnp.linalg.norm(dz) <= trust)
            zn = z + jnp.where(ok, dz, 0.)
            rn2 = jnp.linalg.norm(fun(zn, *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: evaluate(zn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn2 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, zn, z), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)),
                    it + 1, reason)

        z, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body,
            (z0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return z, rn, it, reason, grad(r, J)
    return lm


def make_query(head, K, L, dt, trust, quadrature, ic_budget=400, step_budget=180,
               gtol=1e-6, linear='gj'):
    """The complete matched query: supplied dense field -> six dense fields."""
    wk = weak_fn(quadrature)
    ic = make_stationary_lm(lambda z, y, R: R @ head(z) - y, ic_budget, gtol=gtol, linear=linear)
    lm = make_stationary_lm(lambda z, p, nu, data: wk(z, p, nu, data, head, L, dt),
                            step_budget, trust, gtol, linear)

    def query(u0, nu, data, cold):
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        y = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ y)
        z, icrn, icit, icreason, icgn = ic(Zcand[idx], (y, R), 0.)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(w))

        def step(carry, _):
            z, zprev = carry
            p = data['A'] @ head(z)
            ze = z + (z - zprev)
            r0 = jnp.linalg.norm(wk(z, p, nu, data, head, L, dt))
            re = jnp.linalg.norm(wk(ze, p, nu, data, head, L, dt))
            zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, z)
            z2, rn, it, reason, gn = lm(zi, (p, nu, data), 1e-9 * scale)
            return (z2, z), (z2, rn, it, reason, gn)

        _, (zs, rn, it, reason, gn) = jax.lax.scan(step, (z, z), None, length=int(round(.25 / dt)))
        internal = jnp.concatenate((z[None], zs))
        Z = internal[::int(round(.05 / dt))]
        fields = jax.vmap(lambda z: e.output_field(data['G'] @ head(z), L, L))(Z)
        # icrn and ||ui|| are returned because the normalized-gradient stationarity
        # test is uninformative for arms whose reduced fit is attainable: there the
        # residual goes to round-off while ||J^T r||/(||J|| ||r||) stays O(1).
        return (fields, it, rn, reason, Z, icit, icreason, internal, gn, icgn,
                icrn, jnp.linalg.norm(ui))
    return jax.jit(query)


def make_reconstruction(head, K, L, budget=400, gtol=1e-6, linear='gj'):
    """Best-found reconstruction of a supplied field on the arm's own manifold.
    Offline diagnostic; never inside a timed query."""
    lm = make_stationary_lm(lambda z, target, G: G @ head(z) - target, budget, gtol=gtol, linear=linear)

    def fit(starts, target, G):
        out = jax.vmap(lambda z0: lm(z0, (target, G), 0.))(starts)
        best = jnp.argmin(out[1])
        return out[0][best], out[1][best], out[2][best], out[3][best]
    return jax.jit(fit)
