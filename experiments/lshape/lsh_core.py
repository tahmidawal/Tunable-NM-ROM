"""Shared numerics for the lshape cell: geometry, FD operator, FOM solvers, decoder
factors, banks, floors, oracles, weak operators, POD.

The L-shaped domain Omega = (0,1)^2 \\ [1/2,1)^2 is a masked square grid. The 5-point
operator is assembled two independent ways and gated (DESIGN.md section 2). The full-order
solvers are SciPy SuperLU (reference and timed direct solve), a matrix-free JAX CG on the
masked grid, and SciPy ILU-PCG. The decoder is the parent lane's separable form with the
boundary factor made a parameter (DESIGN.md section 4).

Generic bank / floor / oracle utilities are copied from `p-bank-head/pbh_core.py` so this
module depends only on `sep_common` and `arms` (and `engines` through `arms`).

Staged flat: every module sits beside this file on the cluster.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import sep_common as sc
import arms as A

LSHAPE_LAMBDA1 = 4.0 * 9.6397238440219   # Trefethen-Betcke side-2 value, scaled to side 1


# --------------------------------------------------------------- utilities ---

def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def weights_sha(params):
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()


def dump(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating,)):
            x = float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


def summarise(errors):
    e = np.asarray(errors, dtype=float)
    return dict(worst=float(e.max()), median=float(np.median(e)),
                mean=float(e.mean()), count=int(e.size), per_case=e.tolist())


def fit_validation_split(count, seed, validation_fraction):
    order = np.random.default_rng(seed).permutation(count)
    nval = max(1, int(round(validation_fraction * count)))
    return np.sort(order[nval:]), np.sort(order[:nval])


# ---------------------------------------------------------------- geometry ---

class Geometry:
    """Masked square grid. `shape='square'` reproduces the parent lane's domain and exists
    for the G-FOM-6 fidelity gate only."""

    def __init__(self, N, shape='lshape'):
        assert shape in ('lshape', 'square'), shape
        assert shape == 'square' or N % 2 == 0, 'the corner must be a node'
        self.N, self.shape, self.h = int(N), shape, 1.0 / N
        mask = np.zeros((N + 1, N + 1), dtype=bool)
        mask[1:N, 1:N] = True
        if shape == 'lshape':
            mask[N // 2:, N // 2:] = False
        self.mask = mask
        self.idx = np.flatnonzero(mask.ravel())            # lexicographic, i-major
        self.n = int(len(self.idx))
        ii, jj = np.divmod(self.idx, N + 1)
        self.ij = np.column_stack((ii, jj))
        self.coords = np.column_stack((ii / N, jj / N))
        number = -np.ones((N + 1, N + 1), dtype=np.int64)
        number[mask] = np.arange(self.n)
        self.number = number

    def gather(self, full):
        return np.asarray(full).ravel()[self.idx]

    def scatter(self, vec):
        out = np.zeros((self.N + 1) ** 2)
        out[self.idx] = np.asarray(vec)
        return out.reshape(self.N + 1, self.N + 1)

    def restrict_from(self, fine_field, Nfine):
        assert Nfine % self.N == 0
        s = Nfine // self.N
        return np.asarray(fine_field)[::s, ::s]


def assemble_stencil(geom):
    """5-point -Laplacian on the interior nodes by index arithmetic on the masked grid."""
    N, num = geom.N, geom.number
    ii, jj = geom.ij[:, 0], geom.ij[:, 1]
    rows = [np.arange(geom.n)]
    cols = [np.arange(geom.n)]
    vals = [np.full(geom.n, 4.0 * N * N)]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = num[ii + di, jj + dj]
        keep = nb >= 0
        rows.append(np.arange(geom.n)[keep])
        cols.append(nb[keep])
        vals.append(np.full(int(keep.sum()), -1.0 * N * N))
    Acoo = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(geom.n, geom.n))
    return Acoo.tocsr()


def assemble_kron(geom):
    """Independent assembly: the square's Kronecker-sum operator restricted to the
    interior-node index set (a principal submatrix)."""
    N = geom.N
    m = N - 1
    T = sp.diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1], format='csr')
    Asq = (N * N) * (sp.kron(sp.identity(m, format='csr'), T) + sp.kron(T, sp.identity(m, format='csr')))
    sel = (geom.ij[:, 0] - 1) * m + (geom.ij[:, 1] - 1)
    return Asq.tocsr()[sel][:, sel].tocsr()


# --------------------------------------------------------------------- FOM ---

class FOM:
    """Sparse direct reference + timed comparators on one geometry."""

    def __init__(self, geom, ilu_drop_tol=1e-4, ilu_fill_factor=10., build_ilu=True):
        self.geom = geom
        t0 = time.perf_counter()
        self.A = assemble_stencil(geom)
        self.assembly_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.lu = spla.splu(self.A.tocsc())
        self.factor_seconds = time.perf_counter() - t0
        self.ilu = None
        self.ilu_seconds = None
        if build_ilu:
            t0 = time.perf_counter()
            self.ilu = spla.spilu(self.A.tocsc(), drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor)
            self.ilu_seconds = time.perf_counter() - t0
            self.ilu_nnz = int(self.ilu.L.nnz + self.ilu.U.nnz)
        self.lu_nnz = int(self.lu.L.nnz + self.lu.U.nnz)

    def gates(self, tol_sym=1e-12):
        A1 = self.A
        A2 = assemble_kron(self.geom)
        diff = (A1 - A2)
        diff.eliminate_zeros()
        asym = (A1 - A1.T)
        asym.eliminate_zeros()
        return dict(nnz=int(A1.nnz), interior_unknowns=int(self.geom.n),
                    symmetry_max_abs=float(abs(asym).max()) if asym.nnz else 0.0,
                    symmetric=bool((abs(asym).max() if asym.nnz else 0.0) <= tol_sym),
                    independent_assembly_nnz_difference=int(diff.nnz),
                    independent_assembly_agrees=bool(diff.nnz == 0),
                    diagonal_constant=bool(np.allclose(A1.diagonal(), 4.0 * self.geom.N ** 2)))

    def solve(self, f_int):
        return self.lu.solve(np.asarray(f_int))

    def solve_many(self, F, block=64):
        F = np.asarray(F)
        out = np.empty_like(F)
        for s in range(0, F.shape[0], block):
            out[s:s + block] = self.lu.solve(F[s:s + block].T).T
        return out

    def reference(self, f_int, refinements=2):
        """Direct solve plus iterative refinement; returns (u, relative residual)."""
        f = np.asarray(f_int)
        u = self.lu.solve(f)
        for _ in range(refinements):
            r = f - self.A @ u
            u = u + self.lu.solve(r)
        return u, float(np.linalg.norm(self.A @ u - f) / np.linalg.norm(f))

    def modes(self, M, seed=0):
        """Lowest-M eigenpairs of A by shift-invert Lanczos on the SuperLU factor."""
        t0 = time.perf_counter()
        n = self.geom.n
        opinv = spla.LinearOperator((n, n), matvec=self.lu.solve, dtype=float)
        v0 = np.random.default_rng(seed).standard_normal(n)
        lam, phi = spla.eigsh(self.A, k=M, sigma=0.0, which='LM', OPinv=opinv, v0=v0, tol=0)
        order = np.argsort(lam)
        lam, phi = lam[order], phi[:, order]
        resid = float(np.linalg.norm(self.A @ phi - phi * lam[None, :]) / np.linalg.norm(phi * lam[None, :]))
        orth = float(np.linalg.norm(phi.T @ phi - np.eye(M)))
        return lam, phi, dict(M=int(M), seconds=time.perf_counter() - t0, eigen_residual=resid,
                              orthonormality_error=orth, lambda_min=float(lam[0]),
                              lambda_max_retained=float(lam[-1]),
                              lambda1_reference=LSHAPE_LAMBDA1 if self.geom.shape == 'lshape' else 2 * np.pi ** 2,
                              lambda1_relative_difference=float(abs(lam[0] - (LSHAPE_LAMBDA1 if self.geom.shape == 'lshape' else 2 * np.pi ** 2)) / (LSHAPE_LAMBDA1 if self.geom.shape == 'lshape' else 2 * np.pi ** 2)))

    def pcg_ilu(self, f_int, rtol, maxiter=100000):
        n = self.geom.n
        M = spla.LinearOperator((n, n), matvec=self.ilu.solve, dtype=float)
        count = [0]

        def cb(_):
            count[0] += 1
        u, info = spla.cg(self.A, np.asarray(f_int), rtol=rtol, atol=0.0, maxiter=maxiter, M=M, callback=cb)
        return u, int(count[0]), int(info)


def make_gpu_cg(geom, maxiter):
    """Matrix-free CG on the masked full grid. Input/outputs are (N+1)^2 device arrays
    that are zero off the interior; the iteration count and final residual are returned."""
    N = geom.N
    mask = jnp.asarray(geom.mask)
    h2 = float(N * N)

    def Aop(u):
        lap = 4.0 * u - (jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0) + jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1))
        return jnp.where(mask, lap * h2, 0.0)

    @jax.jit
    def cg(b_full, rtol):
        b = jnp.where(mask, b_full, 0.0)
        bn = jnp.linalg.norm(b)
        x = jnp.zeros_like(b)
        r = b
        p = r
        rs = jnp.sum(r * r)

        def cond(s):
            x, r, p, rs, it = s
            return (jnp.sqrt(rs) > rtol * bn) & (it < maxiter)

        def body(s):
            x, r, p, rs, it = s
            Ap = Aop(p)
            alpha = rs / jnp.sum(p * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs2 = jnp.sum(r * r)
            p = r + (rs2 / rs) * p
            return x, r, p, rs2, it + 1

        x, r, p, rs, it = jax.lax.while_loop(cond, body, (x, r, p, rs, jnp.int32(0)))
        return x, it, jnp.sqrt(rs) / bn
    return cg


# ---------------------------------------------------------- source family ----

def source_params(seed, count):
    """The parent lane's draw (`core.source_params`), copied so no square module is
    needed: each column at its own length."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, count)
    cy = rng.uniform(0.15, 0.85, count)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), count))
    a = rng.uniform(0.5, 2.0, count)
    return np.column_stack((cx, cy, w, a))


def in_omega(draws):
    d = np.asarray(draws)
    return ~((d[:, 0] >= 0.5) & (d[:, 1] >= 0.5))


def cohort(seed, draw_count, count, shape='lshape'):
    draws = source_params(seed, draw_count)
    keep = in_omega(draws) if shape == 'lshape' else np.ones(len(draws), bool)
    accepted = draws[keep]
    assert len(accepted) >= count, (seed, draw_count, count, len(accepted))
    return accepted[:count], dict(seed=int(seed), draw_count=int(draw_count), count=int(count),
                                  accepted_in_draw=int(keep.sum()), rejected_in_draw=int((~keep).sum()),
                                  parameters_sha256=sha_array(accepted[:count]))


def source_full(geom, param):
    """Full nodal source: Gaussian at every node of the square grid, zero outside Omega and on
    the boundary (only interior values enter the equation)."""
    cx, cy, w, a = param
    x = np.linspace(0.0, 1.0, geom.N + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    f = a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2))
    return np.where(geom.mask, f, 0.0)


def source_interior(geom, param):
    return geom.gather(source_full(geom, param))


def fields(fom, draws):
    """Same-mesh sparse-direct interior solutions, (S, n), host float64."""
    F = np.stack([source_interior(fom.geom, q) for q in draws])
    return fom.solve_many(F)


# ---------------------------------------------------- boundary factors -------

def factor_poly(xy, xp=jnp):
    x, y = xy[:, 0], xy[:, 1]
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def factor_smooth(xy, xp=jnp):
    x, y = xy[:, 0], xy[:, 1]
    a, b = 0.5 - x, 0.5 - y
    return 16.0 * x * (1.0 - x) * y * (1.0 - y) * (a + b + xp.sqrt(a * a + b * b))


def factor_sdf(xy, xp=jnp):
    x, y = xy[:, 0], xy[:, 1]
    right = xp.where(y <= 0.5, 1.0 - x, 0.5 - x)
    top = xp.where(x <= 0.5, 1.0 - y, 0.5 - y)
    corner = xp.sqrt((0.5 - x) ** 2 + (0.5 - y) ** 2)
    d = xp.minimum(xp.minimum(xp.minimum(x, y), xp.minimum(right, top)), corner)
    return 4.0 * d


FACTORS = dict(poly=factor_poly, smooth=factor_smooth, sdf=factor_sdf)


def singular_columns(xy, count, xp=jnp):
    """Corner singular functions r^{2j/3} sin(2j(phi - pi/2)/3), cut off at the outer walls."""
    x, y = xy[:, 0], xy[:, 1]
    xi1, xi2 = x - 0.5, y - 0.5
    r = xp.sqrt(xi1 * xi1 + xi2 * xi2)
    phi = xp.arctan2(xi2, xi1)
    phi = xp.where(phi < 0, phi + 2 * np.pi, phi)
    cut = 16.0 * x * (1.0 - x) * y * (1.0 - y)
    cols = [cut * r ** (2.0 * j / 3.0) * xp.sin((2.0 * j / 3.0) * (phi - np.pi / 2)) for j in range(1, count + 1)]
    return xp.stack(cols, axis=1)


# ----------------------------------------------------------------- decoder ---

def make_features(factor, n_enrich):
    """features(params, xy) -> (n_pts, R + n_enrich): bc(x) g~(x) [| s(x)/norm]."""
    bc = FACTORS[factor]

    def features(params, xy):
        ang = 2.0 * jnp.pi * (xy @ params['B'])
        ff = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        G = bc(xy)[..., None] * sc.apply_mlp(params['g'], ff)
        if n_enrich:
            S = singular_columns(xy, n_enrich) / params['enrich_norm'][None, :]
            G = jnp.concatenate([G, S], axis=1)
        return params['out_scale'] * G
    return features


def features_np(params, xy, factor, n_enrich):
    """Pure-NumPy twin of `make_features` for the audit."""
    def mlp(layers, x):
        for w, b in layers[:-1]:
            x = x @ np.asarray(w) + np.asarray(b)
            x = x / (1.0 + np.exp(-x))
        w, b = layers[-1]
        return x @ np.asarray(w) + np.asarray(b)
    xy = np.asarray(xy)
    ang = 2.0 * np.pi * (xy @ np.asarray(params['B']))
    ff = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)
    G = FACTORS[factor](xy, np)[:, None] * mlp(params['g'], ff)
    if n_enrich:
        S = singular_columns(xy, n_enrich, np) / np.asarray(params['enrich_norm'])[None, :]
        G = np.concatenate([G, S], axis=1)
    return float(params['out_scale']) * G


def init_decoder(key, K, R, factor, n_enrich, out_scale, arch, xy_norm):
    """The parent's `init_separable` with g's output sliced to R and the head widened to
    R + n_enrich; the enrichment columns are normalised to unit mean square on `xy_norm`."""
    params = sc.init_separable(key, K, R + n_enrich, out_scale=out_scale, g_hidden=R, **arch)
    w, b = params['g'][-1]
    params['g'][-1] = (w[:, :R], b[:R])
    if n_enrich:
        S = np.asarray(singular_columns(jnp.asarray(xy_norm), n_enrich))
        params['enrich_norm'] = jnp.asarray(np.sqrt(np.mean(S * S, axis=0)))
    return params


def bank_of(features, params, geom, chunk=16384):
    feat = jax.jit(features)
    xy = geom.coords
    parts = []
    for s in range(0, len(xy), chunk):
        block = feat(params, jnp.asarray(xy[s:s + chunk]))
        block.block_until_ready()
        parts.append(block)
    G = jnp.concatenate(parts, axis=0)
    G.block_until_ready()
    return G


def head_of(params):
    return lambda z: sc.head(params, z)


# ------------------------------------------------------------ bank algebra ---

def bank_r(G, retries=2):
    """Thin R factor without materialising Q. The first GPU QR of a tall f64 matrix has
    returned all-NaN on the local GB10 (parent amendment 6), so a non-finite factor is
    recomputed and finiteness is folded into `rank_valid`."""
    G = jnp.asarray(G)
    for attempt in range(retries + 1):
        R = jnp.linalg.qr(G, mode='r')
        if bool(jnp.isfinite(R).all()):
            break
        print(f'WARNING: non-finite QR factor, attempt {attempt + 1}', flush=True)
    values = np.asarray(jnp.linalg.svd(R, compute_uv=False))
    threshold = float(values[0] * max(np.asarray(G).shape) * np.finfo(float).eps)
    rank = int((values > threshold).sum())
    info = dict(rank=rank, rank_valid=bool(rank == int(R.shape[1]) and np.isfinite(values).all()),
                rank_threshold=threshold, condition_number=float(values[0] / values[-1]),
                singular_values=values.tolist(), R_sha256=sha_array(R))
    return R, info


def project_targets(G, R, U, chunk=512):
    """T = Q^T u and the perpendicular energy, from R alone: R^T T = G^T u."""
    Ts, perps, nus = [], [], []
    Gj, Rj = jnp.asarray(G), jnp.asarray(R)
    for s in range(0, int(np.asarray(U).shape[0]), chunk):
        Ub = jnp.asarray(np.asarray(U)[s:s + chunk])
        y = Ub @ Gj
        T = jax.scipy.linalg.solve_triangular(Rj.T, y.T, lower=True).T
        nu2 = jnp.sum(Ub ** 2, axis=1)
        perp2 = jnp.clip(nu2 - jnp.sum(T * T, axis=1), 0., None)
        Ts.append(T); perps.append(perp2); nus.append(nu2)
    return jnp.concatenate(Ts), jnp.concatenate(perps), jnp.concatenate(nus)


def bank_floor(G, R, U):
    _, perp2, nu2 = project_targets(G, R, U)
    return np.asarray(jnp.sqrt(perp2 / nu2))


def oracle_errors(head, Rg, Zcand, T, perp2, nu2, budget, starts=8, gtol=1e-6, linear='lu', block=16):
    """Worst/median best-found error over a cohort, multistart LM from candidate codes
    (the parent's amendment-5 oracle: pivoted step, fixed padded batch)."""
    lm = A.make_stationary_lm(lambda z, t, R: R @ head(z) - t, budget, gtol=gtol, linear=linear)

    @jax.jit
    def fit(starts_b, T_b, R):
        out = jax.vmap(lambda zs, t: jax.vmap(lambda z0: lm(z0, (t, R), 0.))(zs))(starts_b, T_b)
        best = jnp.argmin(out[1], axis=1)
        idx = jnp.arange(out[1].shape[0])
        return out[1][idx, best], out[2][idx, best], out[3][idx, best]

    H = jax.jit(jax.vmap(head))(jnp.asarray(Zcand))
    Hr = H @ jnp.asarray(Rg).T
    Hn = jnp.sum(Hr * Hr, axis=1)
    Zc = jnp.asarray(Zcand)
    Rgj = jnp.asarray(Rg)
    total = int(T.shape[0])
    block = min(block, total)
    errs, iters, reasons = [], [], []
    for s in range(0, total, block):
        take = np.arange(s, min(s + block, total))
        keep = len(take)
        if keep < block:
            take = np.concatenate((take, np.full(block - keep, take[-1])))
        Tb = jnp.asarray(np.asarray(T)[take])
        score = Hn[None, :] - 2. * (Tb @ Hr.T)
        pick = jnp.argsort(score, axis=1)[:, :starts]
        rn, it, reason = jax.device_get(fit(Zc[pick], Tb, Rgj))
        p2 = np.asarray(perp2)[take[:keep]]
        n2 = np.asarray(nu2)[take[:keep]]
        errs.append(np.sqrt(np.asarray(rn)[:keep] ** 2 + p2) / np.sqrt(n2))
        iters.append(np.asarray(it)[:keep])
        reasons.append(np.asarray(reason)[:keep])
    return (np.concatenate(errs), np.concatenate(iters).astype(int), np.concatenate(reasons).astype(int))


def stored_code_errors(head, Rg, Z, T, perp2, nu2):
    H = jax.jit(jax.vmap(head))(jnp.asarray(Z)) @ jnp.asarray(Rg).T
    res2 = jnp.sum((H - T) ** 2, axis=1)
    return np.asarray(jnp.sqrt((res2 + perp2) / nu2))


# ---------------------------------------------------------- weak operators ---

def weak_ops(fom, M, seed=0):
    """P = Lambda^{-1} Phi^T (M x n) and the operator; B = P A G for any bank."""
    lam, phi, info = fom.modes(M, seed)
    P = (phi / lam[None, :]).T
    return dict(P=jnp.asarray(P), P_np=P, lam=lam, phi=phi, info=info, M=int(M))


def reduce_bank(ops, fom, G):
    """B = P A G, with the sparse product on the host and the dense one on the device."""
    AG = fom.A @ np.asarray(G)
    B = ops['P'] @ jnp.asarray(AG)
    B.block_until_ready()
    return B


def weak_sources(ops, F_int):
    Fm = ops['P'] @ jnp.asarray(np.asarray(F_int)).T
    return Fm.T


def correction_basis(head, Z, Rg, T, perp2, nu2, count):
    """Right singular vectors of the normalised training residuals in the exact QR
    physical metric, nested prefixes (the parent's `correction_basis`)."""
    norm = np.sqrt(np.asarray(nu2))
    coeff = np.asarray(jax.jit(jax.vmap(head))(jnp.asarray(Z)))
    residual = (np.asarray(T) - coeff @ np.asarray(Rg).T) / norm[:, None]
    _, s, Vt = np.linalg.svd(residual, full_matrices=False)
    count = int(min(count, Vt.shape[0]))
    physical = Vt[:count].T
    directions = np.linalg.solve(np.asarray(Rg), physical)
    W = np.asarray(Rg) @ directions
    orth = float(np.linalg.norm(W.T @ W - np.eye(count)))
    return dict(coefficient_directions=directions, physical_metric_directions=physical,
                R=np.asarray(Rg), singular_values=s, training_latents=np.asarray(Z),
                training_norms=norm), dict(orthogonality_error=orth, count=int(count),
                                           singular_values=s[:count].tolist())


# ---------------------------------------------------------------- POD --------

def pod_from_host(U, maxrank, block=512):
    """Exact POD of a host snapshot matrix (S, n): blockwise Gram on the device."""
    t0 = time.perf_counter()
    S = int(U.shape[0])
    bounds = [(i, min(i + block, S)) for i in range(0, S, block)]
    gram = np.zeros((S, S))
    for (i0, i1) in bounds:
        Ua = jnp.asarray(U[i0:i1])
        for (j0, j1) in bounds:
            if j0 < i0:
                continue
            g = np.asarray(Ua @ jnp.asarray(U[j0:j1]).T)
            gram[i0:i1, j0:j1] = g
            gram[j0:j1, i0:i1] = g.T
        del Ua
    w, V = np.linalg.eigh(gram)
    w, V = w[::-1], V[:, ::-1]
    energy = float(np.sum(np.clip(w, 0., None)))
    Wt = V[:, :maxrank] / np.sqrt(np.clip(w[:maxrank], 1e-300, None))[None, :]
    modes = jnp.zeros((int(U.shape[1]), maxrank))
    for (i0, i1) in bounds:
        modes = modes + jnp.asarray(U[i0:i1]).T @ jnp.asarray(Wt[i0:i1])
    coords = np.concatenate([np.asarray(jnp.asarray(U[i0:i1]) @ modes) for (i0, i1) in bounds])
    modes.block_until_ready()
    orth = float(np.linalg.norm(np.asarray(modes.T @ modes) - np.eye(maxrank)))
    info = dict(sources=S, blocks=len(bounds), maxrank=int(maxrank), total_energy=energy,
                eigenvalues=np.asarray(w[:maxrank]).tolist(), gram_sha256=sha_array(gram),
                modes_sha256=sha_array(modes), orthonormality_error=orth,
                seconds=time.perf_counter() - t0)
    return modes, coords, info


# -------------------------------------------------------------- timing ------

def burn(seconds):
    a = jnp.ones((384, 384), dtype=jnp.float64) * 0.001
    fn = jax.jit(lambda x: x @ x + 0.0001)
    fn(a).block_until_ready()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn(a).block_until_ready()
