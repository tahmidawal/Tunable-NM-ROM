"""POD and quadratic-manifold bases of the heat-2D TRAINING snapshots at a target mesh, exactly,
through the separable structure of the snapshots (offline; NumPy/SciPy f64).

Every training snapshot is u = amp * g_x (x) g_y (the initial field amp*f(x)f(y) evolved by the
5-point heat semigroup, which factorises over axes). Stack all x-factors into G_x (n-1 x Ns) and
take its SVD G_x = U_x S_x Z_x^T, truncated at 1e-14 relative (likewise y). Then every snapshot
is (U_x (x) U_y) c_j with c_j = amp_j kron(U_x^T g_x, U_y^T g_y), so the snapshot matrix is
S = (U_x (x) U_y) C with orthonormal U_x (x) U_y: every POD / least-squares quantity of S equals
the same quantity of the small core C. A field with core vector m (reshaped rho_x x rho_y) is
U_x M U_y^T on the (n-1)^2 grid (row-major flattening = hires-heat's bank ordering).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dst

TIMES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
NU = 0.02


def family(seed, count):
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(.35, .65, (count, 2)), rng.uniform(.10, .15, count), rng.uniform(.8, 1.2, count)))


def axis_factors(n, draws, times=TIMES, nu=NU):
    """g[j, t, ax, :] (f64) with u_j(t)[i, k] = amp_j g[j,t,0,i] g[j,t,1,k]; amp [j]."""
    a = np.arange(1, n) / n; k = np.arange(1, n)
    lam = 4.0 * n * n * np.sin(np.pi * k / (2 * n)) ** 2
    f = np.stack([4 * a * (1 - a) * np.exp(-(a - draws[:, ax, None]) ** 2 / (2 * draws[:, 2, None] ** 2)) for ax in range(2)], 1)
    c = dst(f, type=1, norm='ortho', axis=-1)
    t = np.asarray(times)
    g = dst(c[:, None] * np.exp(-nu * t[None, :, None, None] * lam), type=1, norm='ortho', axis=-1)
    g[:, 0] = f
    return g, draws[:, 3].copy()


def axis_basis(vectors, rtol=1e-14):
    u, s, _ = np.linalg.svd(vectors, full_matrices=False)
    keep = s > rtol * s[0]
    return u[:, keep], s


def factor_model(n, draws):
    """Returns dict(Ux, Uy, C [rho_x*rho_y, Ns], traj [Ns] trajectory id per column, ...)."""
    g, amp = axis_factors(n, draws)
    J, T = g.shape[:2]
    gx = g[:, :, 0].reshape(J * T, -1).T; gy = g[:, :, 1].reshape(J * T, -1).T
    ux, sx = axis_basis(gx); uy, sy = axis_basis(gy)
    cx = ux.T @ gx; cy = uy.T @ gy                              # rho x Ns
    ampc = np.repeat(amp, T)
    C = (cx[:, None, :] * cy[None, :, :]).reshape(ux.shape[1] * uy.shape[1], -1) * ampc
    # exactness of the factor representation (x-factors / y-factors are reproduced by the bases)
    fx = np.linalg.norm(gx - ux @ cx) / np.linalg.norm(gx); fy = np.linalg.norm(gy - uy @ cy) / np.linalg.norm(gy)
    return dict(Ux=ux, Uy=uy, C=C, traj=np.repeat(np.arange(J), T), rho=(ux.shape[1], uy.shape[1]),
                factor_residual=max(fx, fy), g=g, amp=amp, axis_singular_values=(sx, sy))


def field(fm, m):
    """Core vector(s) m [rho_x*rho_y, k] -> fields [k, n-1, n-1] (NumPy; used for gates and small meshes)."""
    rx, ry = fm['rho']; M = m.reshape(rx, ry, -1)
    return np.einsum('ia,abk,jb->kij', fm['Ux'], M, fm['Uy'], optimize=True)


def pod(C, r):
    """Uncentred POD of the core: leading r left singular vectors and all singular values."""
    u, s, _ = np.linalg.svd(C, full_matrices=False)
    return u[:, :r], s


def vech(a):
    """a [Ns, r] -> [Ns, r(r+1)/2] monomials a_i a_j, i <= j (row-major upper triangle)."""
    i, j = np.triu_indices(a.shape[1])
    return a[:, i] * a[:, j]


GAMMAS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)


def qm_fit(C, r, gamma):
    """Quadratic manifold fitted on core columns C: returns (uref, V, W, A, stats)."""
    uref = C.mean(1, keepdims=True); Cc = C - uref
    u, s, _ = np.linalg.svd(Cc, full_matrices=False); V = u[:, :r]
    A = V.T @ Cc; E = Cc - V @ A; Pi = vech(A.T).T                 # P x Ns
    G = Pi @ Pi.T; P = G.shape[0]; scale = np.trace(G) / P
    W = np.linalg.solve(G + gamma * scale * np.eye(P), Pi @ E.T).T    # (rho^2 x P); symmetric system
    stats = dict(gram_condition=float(np.linalg.cond(G)), in_sample_relative=float(np.linalg.norm(E - W @ Pi) / np.linalg.norm(Cc)),
                 linear_in_sample_relative=float(np.linalg.norm(E) / np.linalg.norm(Cc)), columns=1 + r + P)
    return uref[:, 0], V, W, A, stats


def qm_reconstruct(uref, V, W, C):
    a = V.T @ (C - uref[:, None])
    return uref[:, None] + V @ a + W @ vech(a.T).T


def qm_select(C, traj, r, seed=20260923, fraction=0.2, gammas=GAMMAS):
    """Trajectory-split holdout: whole trajectories go to one side; fit everything on the rest."""
    ids = np.unique(traj); rng = np.random.default_rng(seed)
    held = rng.choice(ids, int(round(fraction * len(ids))), replace=False)
    te = np.isin(traj, held); tr = ~te
    assert not set(traj[te]) & set(traj[tr])
    rows = []
    for g in gammas:
        uref, V, W, _, st = qm_fit(C[:, tr], r, g)
        rec = qm_reconstruct(uref, V, W, C[:, te])
        cc = C[:, te] - uref[:, None]
        per = np.linalg.norm(rec - C[:, te], axis=0) / np.linalg.norm(C[:, te], axis=0)
        rows.append(dict(gamma=g, holdout_frobenius_relative_centred=float(np.linalg.norm(rec - C[:, te]) / np.linalg.norm(cc)),
                         holdout_worst_snapshot_relative=float(per.max()), gram_condition=st['gram_condition']))
    best = min(rows, key=lambda x: x['holdout_frobenius_relative_centred'])
    return best['gamma'], dict(grid=rows, held_out_trajectories=sorted(int(h) for h in held), split_seed=seed, fraction=fraction)
