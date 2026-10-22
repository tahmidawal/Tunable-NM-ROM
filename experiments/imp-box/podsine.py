"""imp-box POD arms on sine-coefficient data: R-PODG, R-TRANS (static translation approximation) and R-ECSW
(DESIGN §4.4-4.6, A1.3, A2.3, A3.4).

POD by the method of snapshots in the L2 inner product, exact by Parseval: <u, v> = (1/4) sum b b' (f64 Gram); rank =
eigenvalues >= 1e-13 lambda_1, at most 512. Modes are sine series, so values and exact derivatives are evaluated at any
point by separable synthesis: on a tensor rule with 1-D points x (m1,), V = Sx B Sx^T, d/dx = (Cx kpi) B Sx^T,
d/dy = Sx B (Cx kpi)^T. The rom.Space holds P (values), S = d/dx + d/dy, w (rule weights), M, K (on the rule), IC points =
the rule, lattice rows Pl.

ECSW: `ecsw_fit` is a copy of imp-geometry pod.ecsw_fit @620942e86 (pod.py imports the FD module, which this lane does
not use), with the candidate rule of A3.4; greedy NNLS = vendor/nnls.py (byte-identical).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor'))
import geom as G   # noqa: E402
import rom as RM   # noqa: E402

R_MAX = 512
RANK_REL = 1e-13
ECSW_SEED = 7_199_001
N_CAND = 32768
ECSW_KAPPA = (2, 4, 8)


def pod(Bs, R=R_MAX):
    """Bs (S, n, n) sine coefficients -> Phi (R_pod, n, n) L2-orthonormal modes, eigenvalues (S,)."""
    S = Bs.shape[0]
    X = jnp.asarray(Bs.reshape(S, -1))
    Gm = np.asarray(jnp.matmul(X, X.T, precision='highest')) / 4.0
    lam, V = np.linalg.eigh(Gm)
    lam, V = lam[::-1], V[:, ::-1]
    keep = int(min(R, np.sum(lam >= RANK_REL * lam[0])))
    A = jnp.asarray(V[:, :keep] / np.sqrt(lam[:keep]))
    Phi = np.asarray(jnp.matmul(A.T, X, precision='highest'))
    # one re-orthonormalisation pass (Gram-Schmidt via QR in the 1/4-scaled metric)
    Qm, _ = np.linalg.qr(Phi.T / 2.0)
    Phi = (Qm.T * 2.0) * np.sign(np.sum(Qm.T * Phi / 2.0, axis=1))[:, None]
    n = Bs.shape[1]
    return Phi.reshape(keep, n, n), lam


def project(Bs, Phi):
    """Coefficients z = <u, phi_i> = (1/4) sum b phi_i."""
    S = Bs.shape[0]
    return np.asarray(jnp.matmul(jnp.asarray(Bs.reshape(S, -1)), jnp.asarray(Phi.reshape(len(Phi), -1)).T,
                                 precision='highest')) / 4.0


def _tables(x, n):
    k = np.arange(1, n + 1)
    Sx = np.sin(np.pi * np.outer(x, k))
    Dx = np.cos(np.pi * np.outer(x, k)) * (np.pi * k)[None, :]
    return jnp.asarray(Sx), jnp.asarray(Dx)


def eval_modes(Phi, x, deriv=True, chunk=32):
    """Values and (d/dx + d/dy) of every mode on the tensor grid x (m1,) x x: (m1^2, R) each."""
    R, n, _ = Phi.shape
    Sx, Dx = _tables(x, n)

    @jax.jit
    def f(B):
        V = jnp.einsum('ia,rab,jb->rij', Sx, B, Sx, precision='highest')
        if not deriv:
            return V, V
        Gx = jnp.einsum('ia,rab,jb->rij', Dx, B, Sx, precision='highest')
        Gy = jnp.einsum('ia,rab,jb->rij', Sx, B, Dx, precision='highest')
        return V, Gx + Gy
    vals, ders = [], []
    for s in range(0, R, chunk):
        B = jnp.asarray(Phi[s:s + chunk])
        if B.shape[0] < chunk:
            B = jnp.pad(B, ((0, chunk - B.shape[0]), (0, 0), (0, 0)))
        V, D = f(B)
        m = min(chunk, R - s)
        vals.append(np.asarray(V[:m]).reshape(m, -1).T)
        ders.append(np.asarray(D[:m]).reshape(m, -1).T)
    return np.concatenate(vals, 1), np.concatenate(ders, 1)


def gx_gy(Phi, x, chunk=32):
    """Separate d/dx and d/dy (for the exact stiffness check)."""
    R, n, _ = Phi.shape
    Sx, Dx = _tables(x, n)
    gx, gy = [], []
    for s in range(0, R, chunk):
        B = jnp.asarray(Phi[s:s + chunk])
        gx.append(np.asarray(jnp.einsum('ia,rab,jb->rij', Dx, B, Sx, precision='highest')).reshape(len(B), -1).T)
        gy.append(np.asarray(jnp.einsum('ia,rab,jb->rij', Sx, B, Dx, precision='highest')).reshape(len(B), -1).T)
    return np.concatenate(gx, 1), np.concatenate(gy, 1)


def pod_space(Phi, k_q, q, name='pod'):
    """rom.Space of the POD span on the composite (k_q, q) rule; M, K on that rule; IC on the rule; lattice rows."""
    x1, w1 = G.rule_1d(k_q, q)
    Xq, wq = G.composite_rule('B', k_q, q)
    Pm, Sm = eval_modes(Phi, x1)
    gx, gy = gx_gy(Phi, x1)
    wj = jnp.asarray(wq)[:, None]
    mm = lambda A, B: np.asarray(jnp.matmul((jnp.asarray(A) * wj).T, jnp.asarray(B), precision='highest'))
    Mm = mm(Pm, Pm)
    Km = mm(gx, gx) + mm(gy, gy)
    xl = np.arange(0, G.LAT + 1) / G.LAT                         # full lattice incl. walls (zero rows)
    Pl, _ = eval_modes(Phi, xl, deriv=False)
    return RM.Space(Pm, Sm, wq, Mm, Km, Pm, wq, Pl, name=f'{name}-k{k_q}-q{q}',
                    meta=dict(dom='B', ic_points=Xq, k_q=k_q, q=q, M_dev=float(np.linalg.norm(Mm - np.eye(len(Mm))))))


def exact_K(Phi):
    """(pi^2 / 4) Phi diag(k^2 + l^2) Phi^T (exact stiffness of L2-orthonormal sine-series modes)."""
    R, n, _ = Phi.shape
    k = np.arange(1, n + 1, dtype=np.float64)
    lam = (np.pi ** 2) * (k[:, None] ** 2 + k[None, :] ** 2)
    F = Phi.reshape(R, -1)
    return (F * lam.reshape(1, -1)) @ F.T / 4.0


def parseval_check(Bs, k_q, q):
    """max relative difference of rule-integrated squared norms vs Parseval over snapshots Bs (S, n, n)."""
    x1, w1 = G.rule_1d(k_q, q)
    n = Bs.shape[1]
    Sx, _ = _tables(x1, n)
    V = np.asarray(jnp.einsum('ia,sab,jb->sij', Sx, jnp.asarray(Bs), Sx, precision='highest'))
    rule = np.einsum('sij,i,j->s', V ** 2, w1, w1)
    exact = np.sum(Bs.reshape(len(Bs), -1) ** 2, axis=1) / 4.0
    return float(np.max(np.abs(rule - exact) / exact))


def gram_check(Phi, k_q):
    """relative Frobenius difference of M on (k_q, 12) vs (k_q, 16)."""
    out = []
    for q in (12, 16):
        x1, w1 = G.rule_1d(k_q, q)
        V, _ = eval_modes(Phi, x1, deriv=False)
        w = np.outer(w1, w1).ravel()
        out.append((V * w[:, None]).T @ V)
    return float(np.linalg.norm(out[0] - out[1]) / np.linalg.norm(out[1]))


def choose_kq(Bs64, Phi, ks=(4, 8, 16, 32, 64), tol=1e-8):
    """A2.3: smallest k_q with Parseval (q = 12) <= tol on 64 snapshots and the Gram criterion <= tol."""
    rec = []
    for k in ks:
        p = parseval_check(Bs64, k, 12)
        g = gram_check(Phi, k) if p <= tol else None
        rec.append(dict(k_q=k, parseval=p, gram=g))
        if p <= tol and g is not None and g <= tol:
            return k, rec
    return None, rec


# ------------------------------------------------------------------------------- R-TRANS shifted decoder -------

def make_shift_decoder(Phi_r, lat=G.LAT, sign=-1.0):
    """z (r,) and shift delta (2,) -> lattice values (P,) of u(x) = sum B sin k pi (x - delta_x) sin l pi (y - delta_y),
    B = sum z_i Phi_i. sign=-1 is the registered decoder (x - delta); sign=+1 is the MF-TRANS-sign fault."""
    R, n, _ = Phi_r.shape
    F = jnp.asarray(Phi_r.reshape(R, -1))
    x = jnp.asarray(np.arange(0, lat + 1) / lat)                  # full lattice incl. walls (scored)
    kp = jnp.asarray(np.pi * np.arange(1, n + 1))

    def dec(z, delta):
        B = (z @ F).reshape(n, n)
        Sx = jnp.sin(jnp.outer(x + sign * delta[0], kp))
        Sy = jnp.sin(jnp.outer(x + sign * delta[1], kp))
        return (Sx @ B @ Sy.T).reshape(-1)
    return dec


def make_trans_query(sp, Phi, r, nsteps, form='tensor', tol='prod', cert=False, sign=-1.0, zero_shift=False, lat=G.LAT,
                     **kw):
    """R-TRANS query: centred descriptor -> Galerkin ETDRK4 z(t_k) (rom.make_query, out='z') -> shifted decode."""
    inner = RM.make_query(sp, r, 'gal_etdrk4', nsteps, form=form, tol=tol, cert=cert, out='z', **kw)
    dec = make_shift_decoder(Phi[:r], lat=lat, sign=sign)

    def q(d):
        dc = d.at[0].set(G.C_BAR).at[1].set(G.C_BAR)
        Z, st = inner(dc)
        delta = jnp.stack([d[0] - G.C_BAR, d[1] - G.C_BAR])
        if zero_shift:
            delta = jnp.zeros(2)
        return jax.vmap(lambda z: dec(z, delta))(Z), st
    q_ = jax.jit(q)
    q_.info = dict(inner=getattr(inner, 'info', {}))
    return q_


def boundary_defect(Phi_r, Z, delta, R0max, n_wall=256):
    """max over outputs of |decoded trace| on the four walls / max|R_0| (diagnostic, A1.3)."""
    R, n, _ = Phi_r.shape
    s = np.arange(n_wall + 1) / n_wall
    kp = np.pi * np.arange(1, n + 1)
    worst = 0.0
    for z in Z:
        B = np.tensordot(z, Phi_r, axes=1)
        for xw in (0.0, 1.0):
            sx = np.sin(kp * (xw - delta[0]))
            sy = np.sin(np.outer(s - delta[1], kp))
            worst = max(worst, float(np.max(np.abs(sy @ B.T @ sx))))
            sy0 = np.sin(kp * (xw - delta[1]))
            sx0 = np.sin(np.outer(s - delta[0], kp))
            worst = max(worst, float(np.max(np.abs(sx0 @ B @ sy0))))
    return worst / R0max


# ------------------------------------------------------------------------------------------------- ECSW --------

def ecsw_fit(sp, r, Mt, Z, kappas=ECSW_KAPPA, seed=ECSW_SEED, n_cand=N_CAND, max_seconds=1800):
    """Copy of imp-geometry pod.ecsw_fit (candidate seed = seed + r, A3.4)."""
    import time as _t
    import nnls as NN
    t_start = _t.time()                        # the fit budget includes the matrix construction (code audit 2)
    P = sp.P
    n = P.shape[0]
    Zr = Z[:, :r]
    U = Zr @ P[:, :r].T
    Sv = Zr @ sp.S[:, :r].T
    contrib_norm = np.sqrt(np.sum((P[:, :Mt] ** 2), axis=1)) * np.mean(np.abs(U * Sv), axis=0)
    rng = np.random.default_rng(seed + r)
    if n <= n_cand:
        cand = np.arange(n)
    else:
        half = n_cand // 2
        uni = rng.choice(n, half, replace=False)
        rest = np.setdiff1d(np.arange(n), uni)
        p = contrib_norm[rest] + 1e-300
        imp = rng.choice(rest, n_cand - half, replace=False, p=p / p.sum())
        cand = np.sort(np.concatenate([uni, imp]))
    w = sp.w
    ns = Zr.shape[0]
    Gm = np.empty((ns * Mt, len(cand)))        # preallocated (no row-list + concatenate copy; code audit 2)
    bv = np.empty(ns * Mt)
    Pc = P[cand, :Mt]
    for zi in range(ns):
        if _t.time() - t_start > max_seconds:     # hard budget during construction (job audit 1): incomplete fit
            return {k: None for k in kappas}, dict(reason='budget-construction', n_cand=len(cand), rows=int(zi * Mt),
                                                  support_reached=0, relres=float('nan'))
        full = P[:, :Mt].T @ (w * U[zi] * Sv[zi])
        sc = 1.0 / (np.linalg.norm(full) + 1e-300)
        Gm[zi * Mt:(zi + 1) * Mt] = (Pc * (U[zi, cand] * Sv[zi, cand] * sc)[:, None]).T
        bv[zi * Mt:(zi + 1) * Mt] = full * sc
    targets = sorted({min(int(k * r), len(cand)) for k in kappas})
    left = max(1.0, max_seconds - (_t.time() - t_start))
    o = NN.greedy_nnls(Gm, bv, supports=targets, cap=max(targets), max_seconds=left)
    out = {}
    for k in kappas:
        t = min(int(k * r), len(cand))
        x = o['records'].get(('s', t))
        if x is None and o['reason'] == 'kkt' and len(o['support']):
            pos = o['weights'] > 0
            x = dict(support=o['support'][pos], weights=o['weights'][pos], relres=o['relres'], size=int(pos.sum()))
        out[k] = None if x is None else dict(nodes=cand[x['support']], weights=np.asarray(x['weights']),
                                             relres=float(x['relres']), size=int(x['size']))
    return out, dict(reason=o['reason'], n_cand=len(cand), rows=Gm.shape[0], support_reached=int(len(o['support'])),
                     relres=float(o['relres']))
