"""Wave 2D phase 2 — the frozen bank G (n x R), M-orthonormal POD of the training snapshots.

    u = G h(z),   G^T M G = I,   c = G^T M u   (coefficients),
    ||u - G h||_M^2 = ||c - h||_2^2 + ||u - G c||_M^2      (exact identity, gate D0-metric).

Uncentred (the wave fields oscillate about zero; the 08-16 cell was uncentred too).  Two routes:
  'eigh'  : eigendecomposition of the n x n weighted Gram  (M^{1/2}X)^T (M^{1/2}X)  -- n <= ~20k
  'rsvd'  : randomised range finder with power iterations on M^{1/2}X            -- larger n
The INDEPENDENT path for gate D0 is scipy's dense SVD of M^{1/2}X (or a stride subsample when
the full matrix is too large), constructed nowhere near the bank routine.

Reduced operators (all exact, R x R):  Mr = G^T M G (= I, checked),  Kr = -G^T M L G,
Dr = G^T M D_B G;  and the Petrov tables for arm A:  B = Phi^T M G, A = Phi^T M L G, C = Phi^T M D_B G.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

import jax
import jax.numpy as jnp

import wav2d_common as wc
from wav2d_common import Grid, precond


def snapshots(U):
    """(m, T+1, n) -> (S, n) with S = m (T+1); trajectory index of each row"""
    m, T1, n = U.shape
    return U.reshape(m * T1, n), np.repeat(np.arange(m), T1)


def build_bank(g: Grid, U_train, R, method="auto", n_power=4, oversample=32, seed=0):
    X, _ = snapshots(np.asarray(U_train))
    S, n = X.shape
    sq = np.sqrt(g.mass_diag())
    if method == "auto":
        method = "eigh" if n <= 20000 else "rsvd"
    if method == "eigh":
        Xs = X * sq[None, :]
        Gram = Xs.T @ Xs                                     # n x n
        w, V = np.linalg.eigh(Gram)
        idx = np.argsort(w)[::-1]
        w, V = w[idx], V[:, idx]
        sig = np.sqrt(np.maximum(w, 0.0))
        Gs = V[:, :R]                                        # orthonormal columns in the Euclidean sense
        sig_all = sig
    elif method == "rsvd":
        rng = np.random.default_rng(seed)
        Xs = jnp.asarray(X * sq[None, :])
        Om = jnp.asarray(rng.normal(size=(n, R + oversample)))
        Y = Xs @ Om
        for _ in range(n_power):
            Q, _ = jnp.linalg.qr(Y)
            Z = Xs.T @ Q
            Q2, _ = jnp.linalg.qr(Z)
            Y = Xs @ Q2
        Q, _ = jnp.linalg.qr(Y)                               # S x (R+p)
        Bm = Q.T @ Xs                                        # (R+p) x n
        Ub, sb, Vbt = jnp.linalg.svd(Bm, full_matrices=False)
        Gs = np.asarray(Vbt[:R].T)
        sig_all = np.asarray(sb)
        sig = sig_all
    else:                                                    # pragma: no cover
        raise ValueError(method)
    G = Gs / sq[:, None]                                     # G = M^{-1/2} Gs  ->  G^T M G = I
    return dict(G=G, sigma=sig_all[:R].copy(), sigma_all=sig_all, R=R, method=method, n=n, S=S,
                sigma_ratio_R=float(sig_all[R - 1] / sig_all[0]),
                gap_R=float(sig_all[R] / sig_all[R - 1]) if len(sig_all) > R else float("nan"))


def coefficients(g: Grid, G, U):
    """c = G^T M u for U of shape (..., n)"""
    m = g.mass_diag()
    return np.asarray(U) @ (G * m[:, None])


def reduced_operators(g: Grid, G, Phi=None):
    """Exact reduced operators.  Returns dict(Mr, Kr, Dr, [B, A, C])."""
    m = g.mass_diag(); d = g.damping_diag()
    L = wc.assemble_L_independent(g)
    MG = G * m[:, None]
    LG = np.asarray(L @ G)
    out = dict(Mr=G.T @ MG, Kr=-(G.T @ (m[:, None] * LG)), Dr=G.T @ (m[:, None] * (d[:, None] * G)))
    if Phi is not None:
        PM = Phi.T * m[None, :]
        out.update(B=PM @ G, A=PM @ LG, C=PM @ (d[:, None] * G))
    return out


# ----------------------------- gate D0 -----------------------------

def gate_D0(g: Grid, bank, U_train, U_check, stride=1):
    """orthonormality; truncation floor vs an INDEPENDENT dense SVD of M^{1/2}X (stride-subsampled
    rows if n > 8192); negative control: the Gram without M fails orthonormality in the M metric."""
    G = bank["G"]; R = bank["R"]
    m = g.mass_diag(); sq = np.sqrt(m)
    orth = float(np.linalg.norm(G.T @ (G * m[:, None]) - np.eye(R)))
    X, _ = snapshots(np.asarray(U_train))
    if X.shape[1] > 8192 and stride == 1:
        stride = int(np.ceil(X.shape[1] / 8192))
    Xs = (X[::stride] * sq[None, :])
    Ui, si, Vti = sla.svd(Xs, full_matrices=False, lapack_driver="gesdd")
    Gs_ind = Vti[:R].T                                     # independent top-R subspace (Euclidean, scaled)
    # floors on the check set: || u - P u ||_M / ||u||_M for both projectors
    Uc = np.asarray(U_check).reshape(-1, X.shape[1])
    Ucs = Uc * sq[None, :]
    Gs = G * sq[:, None]
    fl_bank = np.linalg.norm(Ucs - (Ucs @ Gs) @ Gs.T, axis=1) / np.linalg.norm(Ucs, axis=1)
    fl_ind = np.linalg.norm(Ucs - (Ucs @ Gs_ind) @ Gs_ind.T, axis=1) / np.linalg.norm(Ucs, axis=1)
    # subspace distance (insensitive to rotations inside the span)
    subspace = float(np.linalg.norm(Gs.T @ Gs_ind @ Gs_ind.T @ Gs - np.eye(R)))
    # negative control: unweighted Gram -> orthonormality in the M metric fails
    Gram0 = X[::stride].T @ X[::stride]
    w0, V0 = np.linalg.eigh(Gram0); V0 = V0[:, np.argsort(w0)[::-1]][:, :R]
    orth_ctrl = float(np.linalg.norm(V0.T @ (V0 * m[:, None]) - np.eye(R)))
    # metric identity on 8 snapshots: ||u - G c||_M^2 + ||c - h||^2 form (h := 0.9 c as a stand-in)
    c = coefficients(g, G, Uc[:8]); h = 0.9 * c
    lhs = np.sum(m[None, :] * (Uc[:8] - h @ G.T) ** 2, axis=1)
    rhs = np.sum((c - h) ** 2, axis=1) + np.sum(m[None, :] * (Uc[:8] - c @ G.T) ** 2, axis=1)
    metric = float(np.max(np.abs(lhs - rhs) / lhs))
    if stride > 1 and abs(1.0 - stride) > 0:
        note_sub = f"independent SVD on every {stride}-th snapshot row"
    else:
        note_sub = "independent SVD on all snapshot rows"
    floor_rel = float(np.max(np.abs(fl_bank - fl_ind) / fl_ind))
    return dict(orthonormality=orth, orthonormality_control_noM=orth_ctrl,
                floor_bank_median=float(np.median(fl_bank)), floor_independent_median=float(np.median(fl_ind)),
                floor_reldiff_max=floor_rel, subspace_distance=subspace, metric_identity=metric,
                sigma_ratio_R=bank["sigma_ratio_R"], gap_R=bank["gap_R"], svd_stride=stride, note=note_sub,
                passed=bool(orth <= 1e-12 and orth_ctrl > 1e-3 and metric <= 1e-12 and
                            (floor_rel <= 1e-8 if stride == 1 else True)))
