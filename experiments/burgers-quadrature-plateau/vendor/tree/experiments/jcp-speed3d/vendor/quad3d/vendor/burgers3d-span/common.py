"""burgers3d-span: shared primitives (DESIGN.md).

Scalar Burgers u_t + u (u_x + u_y + u_z) = nu Lap u on the unit cube, zero Dirichlet walls, the paper-b3d family
(vendor/b3d_common.py, imported unchanged from experiments/paper-b3d/vendor), backward Euler dt = 0.005 x 50 steps,
sign-upwind advection, 7-point Laplacian. Mesh n = nodes per axis INCLUDING the walls (33, 65, 129); ni = n - 2.

Reduced model (paper Sec. 3): bank G(x) = mu(x) g_phi(x) (coordinate network, exact Dirichlet factor), ordered once,
G_hat = G T; head h(z) trained on ordered-bank coefficients. A query solves either
    span:  u = G_hat[:, :R'] c,  c in R^{R'}
    head:  u = G_hat[:, :R'] h(z)[:R'],  z in R^K
by LSPG on the M lowest sine tests with row scaling S = 1/(1 + dt nu lambda):
    r = S * ( A c - A c_prev + dt ( adv(c) + nu lambda * A c ) ),   A = Phi^T G_hat  (exact, precomputed by DST)
adv(c) is the tested advection Phi^T a(u), approximated by a certified RULE:
    'tensor'  : the backward-difference quadratic form Phi^T (u * D^- u), precomputed per mesh (exact wherever the
                decoded state is positive; its departure from the sign-upwind stencil is certified by rho)
    'lat16'   : empirical quadrature on the uniform lattice x = k/16 with equal weights (the paper's 2D lattice EQ)
Levenberg-Marquardt with the fast path of the 2D lane: analytic Jacobian, one (r, J) per iteration, Cholesky of the
damped normal matrix, clipped step, damping carried between steps, quadratic predictor.
All large arrays are explicit jit arguments.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'paper-b3d' / 'vendor'))
import b3d_common as b3          # noqa: E402  family, upwind stencil, Laplacian (unchanged vendor file)

DT = 0.005
STEPS = 50
KEEP = 10                        # output every 10 steps: t = 0, .05, ..., .25
TIMES = np.arange(6) * 0.05


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False, default=_default) + '\n')
    tmp.replace(path)


def _default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def sha_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def block(x):
    return jax.block_until_ready(x)


# ------------------------------------------------------------------ family ---

def table(seed, count):
    """paper-b3d raw parameter table with its 257^3 peak normalisation (prefix-stable rows)."""
    return b3.build_param_table(int(seed), int(count))


def table_sha(tab):
    return tab['sha256']


def initial_interior(n, tab, j):
    """Interior initial field (ni^3, flat, x slowest) of row j at mesh n (paper-b3d blob family)."""
    ni = n - 2
    ax = np.linspace(0.0, 1.0, n)[1:-1]
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    return np.asarray(b3.blob_ic_3d(n, tab, j, coords=coords)).reshape(ni ** 3)


def interior_coords(n):
    ax = np.linspace(0.0, 1.0, n)[1:-1]
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


# ------------------------------------------------------------------- DST ---

def dst_axis(u, axis):
    """Orthonormal DST-I along `axis` (length ni = n - 2): X_k = sqrt(2/(n-1)) sum_j x_j sin(pi j k / (n-1))."""
    u = jnp.moveaxis(u, axis, -1)
    ni = u.shape[-1]
    z = jnp.zeros(u.shape[:-1] + (1,), dtype=u.dtype)
    odd = jnp.concatenate((z, u, z, -u[..., ::-1]), -1)
    out = -jnp.fft.rfft(odd, axis=-1).imag[..., 1:ni + 1] / jnp.sqrt(2.0 * (ni + 1))
    return jnp.moveaxis(out, -1, axis)


def dst3(u):
    """u (..., ni, ni, ni) -> orthonormal 3D DST-I on the last three axes (self-inverse)."""
    for ax in (-1, -2, -3):
        u = dst_axis(u, ax)
    return u


def lam1(n):
    p = np.arange(1, n - 1)
    return 4.0 * (n - 1) ** 2 * np.sin(np.pi * p / (2 * (n - 1))) ** 2


def mode_order(n):
    """All ni^3 modes sorted by discrete eigenvalue (stable), as paper-b3d test_modes_3d."""
    ni = n - 2
    l1 = lam1(n)
    lam = (l1[:, None, None] + l1[None, :, None] + l1[None, None, :]).ravel()
    order = np.argsort(lam, kind='stable')
    return order, lam


def complete_M(n, M, order=None, lam=None):
    """Extend M to the end of its degenerate eigen-shell (paper-b3d convention)."""
    if order is None:
        order, lam = mode_order(n)
    if M >= lam.size:
        return lam.size
    lm = lam[order[M - 1]]
    while M < lam.size and abs(lam[order[M]] - lm) <= 1e-9 * lm:
        M += 1
    return M


def modes(n, M):
    """First M modes (after shell completion by the caller): (kx, ky, kz) 1-based, lambda."""
    order, lam = mode_order(n)
    ni = n - 2
    kx, ky, kz = np.unravel_index(order[:M], (ni, ni, ni))
    return np.stack([kx + 1, ky + 1, kz + 1], 1), lam[order[:M]]


def phiT(cols, n, idx):
    """Phi^T X for X given as (c, ni^3) rows; idx = (kx, ky, kz) 1-based arrays. Returns (c, M)."""
    ni = n - 2
    d = dst3(cols.reshape((-1, ni, ni, ni)))
    return d[:, idx[0] - 1, idx[1] - 1, idx[2] - 1]


def phi_explicit(n, kxyz, nodes_ijk):
    """Phi values at given interior nodes (1-based node index per axis), columns = modes: (len(nodes), M). NumPy."""
    S = np.sqrt(2.0 / (n - 1))
    out = np.ones((len(nodes_ijk), len(kxyz)))
    for a in range(3):
        out *= S * np.sin(np.pi * np.outer(nodes_ijk[:, a], kxyz[:, a]) / (n - 1))
    return out


# ---------------------------------------------------------------- stencils ---

def upwind(u, n):
    return b3.upwind_adv_field_3d(u, n)


def backward_adv(u, n):
    return b3.backward_adv_field_3d(u, n)


def dminus(U, n):
    """Backward-difference divergence-like operator D^- = D^-_x + D^-_y + D^-_z (ghost zeros) on rows of U (c, ni^3)."""
    ni = n - 2
    V = U.reshape((-1, ni, ni, ni))
    P = jnp.pad(V, ((0, 0), (1, 0), (1, 0), (1, 0)))
    d = 3 * V - P[:, :-1, 1:, 1:] - P[:, 1:, :-1, 1:] - P[:, 1:, 1:, :-1]
    return (d * (n - 1)).reshape(U.shape)


def advect_points(us, dx):
    """Sign-upwind advection at stencil points us (7,): [c, x-, x+, y-, y+, z-, z+]."""
    c = us[0]
    s = 0.
    for a in range(3):
        lo, hi = us[1 + 2 * a], us[2 + 2 * a]
        s = s + jnp.where(c > 0, c - lo, hi - c)
    return c * s / dx


# ---------------------------------------------------------------------- FOM ---

MAX_NEWTON = 20
LIN_MAXITER = 2000


def make_fom(n, dt, ntol, ltol, save_steps=None):
    """Tolerance-terminated backward Euler, Newton with matrix-free BiCGStab preconditioned by the exact Helmholtz
    inverse in the discrete sine basis (paper-b3d core.make_fom, output restricted to the six output times).
    query(u0, nu) -> (fields (6, ni^3), newton iterations (steps,), final relative residuals (steps,)).
    save_steps (training data only): return the states at these step indices (0..steps) instead."""
    ni = n - 2
    steps = int(round(0.25 / dt))
    keep = int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12
    save = None if save_steps is None else jnp.asarray(np.asarray(save_steps, dtype=np.int64))
    l1 = jnp.asarray(lam1(n))
    spec = l1[:, None, None] + l1[None, :, None] + l1[None, None, :]

    def residual(u, prev, nu):
        return u - prev + dt * (b3.upwind_adv_field_3d(u, n) - nu * b3.lap_3d(u, n))

    def query(u0, nu):
        def pre(v):
            return dst3(dst3(v.reshape(ni, ni, ni)) / (1 + dt * nu * spec)).ravel()

        def step(prev, _):
            scale = jnp.maximum(jnp.linalg.norm(prev), 1e-300)

            def body(state):
                u, it, _ = state
                r = residual(u, prev, nu)
                jv = lambda v: jax.jvp(lambda w: residual(w, prev, nu), (u,), (v,))[1]
                du, _ = jax.scipy.sparse.linalg.bicgstab(jv, -r, tol=ltol, maxiter=LIN_MAXITER, M=pre)
                u = u + du
                return u, it + 1, jnp.linalg.norm(residual(u, prev, nu))
            u, it, rn = jax.lax.while_loop(lambda s: (s[2] > ntol * scale) & (s[1] < MAX_NEWTON), body,
                                          (prev, jnp.int32(0), jnp.linalg.norm(residual(prev, prev, nu))))
            return u, (u, it, rn / scale)
        _, (fields, it, rn) = jax.lax.scan(step, u0, None, length=steps)
        if save is not None:
            return jnp.concatenate((u0[None], fields))[save], it, rn
        out = jnp.concatenate((u0[None], fields[keep - 1::keep]))
        return out, it, rn
    return jax.jit(query)


def fom_residual_np(u, prev, nu, n, dt):
    """Independent NumPy backward-Euler residual (audit)."""
    ni = n - 2
    v = u.reshape(ni, ni, ni)
    p = np.pad(v, 1)
    adv = np.zeros_like(v)
    lap = np.zeros_like(v)
    for axis in range(3):
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[axis] = slice(None, -2)
        hi[axis] = slice(2, None)
        a, b = p[tuple(lo)], p[tuple(hi)]
        adv += v * np.where(v > 0, v - a, b - v) * (n - 1)
        lap += (b - 2 * v + a) * (n - 1) ** 2
    return (u - prev + dt * (adv - nu * lap).ravel())


# ------------------------------------------------------------------- bank ---

def mlp(p, x):
    for w, b in p[:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = p[-1]
    return x @ w + b


def mask(x):
    return 64.0 * jnp.prod(x * (1 - x), axis=-1)


def features(p, x):
    """G(x) = mu(x) * scale * g_phi(x): (P, R)."""
    ang = 2 * jnp.pi * (x @ p['freq'])
    return (p['scale'] * mask(x))[:, None] * mlp(p['net'], jnp.concatenate((jnp.sin(ang), jnp.cos(ang)), -1))


def features_np(p, x):
    """Independent NumPy evaluation (audit)."""
    p = jax.tree_util.tree_map(np.asarray, p)
    ang = 2 * np.pi * (x @ p['freq'])
    h = np.concatenate((np.sin(ang), np.cos(ang)), -1)
    for w, b in p['net'][:-1]:
        a = h @ w + b
        h = a / (1 + np.exp(-a))
    w, b = p['net'][-1]
    return (float(p['scale']) * 64.0 * np.prod(x * (1 - x), axis=-1))[:, None] * (h @ w + b)


def head(p, z):
    return mlp(p['net'], z) + z @ p['skip']


def head_np(p, z):
    p = jax.tree_util.tree_map(np.asarray, p)
    h = np.asarray(z)
    for w, b in p['net'][:-1]:
        a = h @ w + b
        h = a / (1 + np.exp(-a))
    w, b = p['net'][-1]
    return h @ w + b + np.asarray(z) @ p['skip']


def bank_at(bank_params, T, x, chunk=1 << 16):
    """G_hat(x) = G(x) T for points x (P, 3), chunked, device array (P, R)."""
    f = jax.jit(lambda p, xx, T: features(p, xx) @ T)
    Tj = jnp.asarray(T)
    parts = [f(bank_params, jnp.asarray(x[s:s + chunk]), Tj) for s in range(0, len(x), chunk)]
    return jnp.concatenate(parts, 0)


# ---------------------------------------------------------------- the model at a mesh ---

def ladder_blocks(ladder, R):
    edges = sorted(set([0] + list(ladder) + [R]))
    return edges


def build_mesh(n, bank_params, T, head_params, library, ladder, M_of, lattice=16, log=print,
               build_tensor=True, build_eq=True):
    """Everything a query at mesh n needs, computed offline from the frozen model (no training data).

    Returns dict with Q column blocks (orthonormal basis of span G_hat at this mesh, nested), Rq (G_hat = Q Rq),
    A (M_max x R) = Phi^T G_hat, lam, modes, Tsym (M_max x R x R) = T + T^T with T[m,i,j] = Phi_m^T(G_i * D^- G_j),
    lattice EQ tables (G7 (m,7,R), Pq (m, M_max)), library of training codes and R-coefficients."""
    t0 = time.perf_counter()
    R = int(np.asarray(T).shape[1])
    ni = n - 2
    x = interior_coords(n)
    G = bank_at(bank_params, T, x)                                  # (N, R)
    log(f'[mesh {n}] bank evaluated {G.shape} {time.perf_counter() - t0:.1f}s')
    Q, Rq = jnp.linalg.qr(G, mode='reduced')
    sgn = jnp.sign(jnp.diag(Rq))
    Q, Rq = Q * sgn[None, :], Rq * sgn[:, None]
    edges = ladder_blocks(ladder, R)
    Qb = tuple(Q[:, a:b] for a, b in zip(edges[:-1], edges[1:]))
    del Q
    M_max = M_of(R)
    kxyz, lam = modes(n, M_max)
    idx = tuple(jnp.asarray(kxyz[:, a]) for a in range(3))
    A = jnp.concatenate([phiT(G[:, s:s + 32].T, n, idx) for s in range(0, R, 32)], 0).T      # (M, R)
    log(f'[mesh {n}] A {A.shape} {time.perf_counter() - t0:.1f}s')
    out = dict(n=n, R=R, edges=edges, Qb=Qb, Rq=Rq, A=A, lam=jnp.asarray(lam), kxyz=kxyz, M_max=M_max)
    if build_tensor:
        DG = dminus(G.T, n)                                           # (R, N)
        f = jax.jit(lambda gi, DGc: phiT(gi[None, :] * DGc, n, idx))  # (c, M)
        Tt = []
        for i in range(R):
            row = jnp.concatenate([f(G[:, i], DG[s:s + 64]) for s in range(0, R, 64)], 0)   # (R_j, M)
            Tt.append(row.T)                                           # (M, R_j)
        Tm = jnp.stack(Tt, 1)                                          # (M, R_i, R_j)
        del DG, Tt
        out['Tsym'] = Tm + jnp.transpose(Tm, (0, 2, 1))
        del Tm
        log(f'[mesh {n}] tensor {out["Tsym"].shape} {time.perf_counter() - t0:.1f}s')
    if build_eq:
        s = (n - 1) // lattice
        assert s * lattice == n - 1 and s >= 2, (n, lattice)
        k = np.arange(1, lattice) * s                                  # node index along an axis (0..n-1)
        I, J, K = np.meshgrid(k, k, k, indexing='ij')
        nodes = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)          # 1..n-2, interior
        offs = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
        st = nodes[:, None, :] + offs[None]                            # (m, 7, 3), all interior since s >= 2
        flat = ((st[..., 0] - 1) * ni + (st[..., 1] - 1)) * ni + (st[..., 2] - 1)
        out['G7'] = G[jnp.asarray(flat.ravel())].reshape(flat.shape + (R,))
        out['Pq'] = jnp.asarray(s ** 3 * phi_explicit(n, kxyz, nodes))  # (m, M)
        out['lattice_nodes'] = nodes
        out['lattice_flat'] = flat
        log(f'[mesh {n}] lattice EQ m={len(nodes)} {time.perf_counter() - t0:.1f}s')
    if head_params is not None:
        out['library'] = dict(Z=jnp.asarray(library['Z']), H=jnp.asarray(library['H']))
    out['G'] = G
    return out


def exact_tested_adv(c, G, n, idx):
    """Phi^T a_upwind(G c) over every interior node (the FOM's stencil), via DST."""
    u = G @ c
    return phiT(upwind(u, n)[None], n, idx)[0]


# --------------------------------------------------------------------- LM ---

def make_fused_lm(evalJ, K, budget, trust, gtol, clip=True):
    """hires-burgers hfast.make_fused_lm restricted to q = 0 with Cholesky: every unknown damped and clipped.
    reason: 4 stationary (scale-free gradient), 1 residual tolerance, 2 tiny step, 3 damping exhausted / nonfinite,
    0 budget."""
    def ratio(g, J, rn):
        return jnp.linalg.norm(g) / (jnp.linalg.norm(J) * rn + 1e-300)

    def lm(w0, args, tol, lam0):
        r, J = evalJ(w0, args)
        rn = jnp.linalg.norm(r)
        g = J.T @ r
        reason = jnp.where(jnp.isfinite(rn),
                           jnp.where(ratio(g, J, rn) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            w, r, J, g, rn, lam, it, reason, rej = s
            Hm = J.T @ J
            d = jnp.diag(Hm) + 1e-30
            c = jax.scipy.linalg.cho_factor(Hm + jnp.diag(lam * d), lower=True)
            step = jax.scipy.linalg.cho_solve(c, -g)
            if clip:
                nz = jnp.linalg.norm(step)
                step = step * jnp.where(nz > trust, trust / (nz + 1e-300), 1.)
            ok = jnp.all(jnp.isfinite(step))
            wn = w + jnp.where(ok, step, 0.)
            r2, J2 = evalJ(wn, args)
            rn2 = jnp.linalg.norm(r2)
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            w3 = jnp.where(accept, wn, w)
            r3 = jnp.where(accept, r2, r)
            J3 = jnp.where(accept, J2, J)
            rn3 = jnp.where(accept, rn2, rn)
            g3 = J3.T @ r3
            gn = ratio(g3, J3, rn3)
            tiny = ok & (jnp.linalg.norm(step) <= 1e-14 * (1 + jnp.linalg.norm(w)))
            reason = jnp.where(gn <= gtol, 4,
                      jnp.where(rn3 <= tol, 1,
                       jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            lam2 = jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            return (w3, r3, J3, g3, rn3, lam2, it + 1, reason, rej + (~accept).astype(jnp.int32))

        w, r, J, g, rn, lam, it, reason, rej = jax.lax.while_loop(
            lambda s: (s[6] < budget) & (s[7] == 0), body,
            (w0, r, J, g, rn, jnp.asarray(lam0, dtype=jnp.float64), jnp.int32(0), reason, jnp.int32(0)))
        return w, rn, it, reason, ratio(g, J, rn), rej, lam
    return lm


def make_head_fit(budget=200, gtol=1e-6):
    """Initial head fit: min_z || Rq' h(z)[:R'] - t' || (LM on K unknowns), paper-b3d style."""
    def fit(hp, z0, Rq, t, Rp):
        def evalJ(z, args):
            f = lambda zz: Rq @ head(hp, zz)[:Rp] - t
            return f(z), jax.jacfwd(f)(z)
        lm = make_fused_lm(evalJ, z0.shape[0], budget, jnp.inf, gtol, clip=False)
        return lm(z0, None, 0., 1e-6)
    return fit


def make_query(n, kind, rule, Rp, M, K=None, gtol=1e-3, step_budget=50, trust=jnp.inf, fit_budget=200, kxyz=None):
    """Build the jitted query for one arm at mesh n.

    kind 'span' (unknowns c in R^{R'}) or 'head' (unknowns z in R^K, u = G_hat[:, :R'] h(z)[:R']);
    rule 'tensor' or 'lat16'. data: dict with Qb (nested column blocks, prefix used), Rq (R' x R'), A (M x R'),
    lam (M,), and Ts (M x R' x R') or (G7 (m,7,R'), Pq (m, M)), and for the head library Z, RH (lib x R').
    query(u0, nu, data, hp) -> (fields (6, N), internal states (51, d), iterations (50,), reasons (50,),
                                gradients (50,), residuals (50,), rejected (50,), initial-fit record (4,))."""
    dx = 1.0 / (n - 1)
    dadv = jax.vmap(jax.grad(lambda us: advect_points(us, dx)))
    adv_pts = jax.vmap(lambda us: advect_points(us, dx))

    def coef(w, hp):
        return w if kind == 'span' else head(hp, w)[:Rp]

    if rule == 'exact':
        eidx = tuple(jnp.asarray(np.asarray(kxyz)[:M, a]) for a in range(3))

    def exact_adv(c, data):
        return phiT(upwind(data['G'] @ c, n)[None], n, eidx)[0]

    def tested_adv(c, data):
        """(adv (M,), Ju (M, R')) of the rule at coefficients c."""
        if rule == 'exact':          # control: the FOM's sign-upwind advection on every node, exact Jacobian
            u = data['G'] @ c
            cols = jax.vmap(lambda g: jax.jvp(lambda v: upwind(v, n), (u,), (g,))[1])(data['G'].T)
            return exact_adv(c, data), phiT(cols, n, eidx).T
        if rule == 'tensor':
            Ju = jnp.einsum('mij,j->mi', data['Ts'], c)
            return 0.5 * Ju @ c, Ju
        us = jnp.einsum('msr,r->ms', data['G7'], c)
        a = adv_pts(us)
        da = dadv(us)
        return data['Pq'].T @ a, data['Pq'].T @ jnp.einsum('ms,msr->mr', da, data['G7'])

    def evalJ(w, args):
        prev, nu, data, hp = args
        lam = data['lam']
        S = 1.0 / (1.0 + DT * nu * lam)
        if kind == 'span':
            c = w
            Ac = data['A'] @ c
            adv, Ju = tested_adv(c, data)
            r = (Ac - prev + DT * (adv + nu * lam * Ac)) * S
            return r, data['A'] + (DT * S)[:, None] * Ju
        c, Jh = head(hp, w)[:Rp], jax.jacfwd(lambda z: head(hp, z)[:Rp])(w)
        Ac = data['A'] @ c
        adv, Ju = tested_adv(c, data)
        r = (Ac - prev + DT * (adv + nu * lam * Ac)) * S
        return r, (data['A'] + (DT * S)[:, None] * Ju) @ Jh

    def res(w, args):
        prev, nu, data, hp = args
        lam = data['lam']
        c = coef(w, hp)
        Ac = data['A'] @ c
        if rule == 'tensor':
            adv = 0.5 * jnp.einsum('mij,i,j->m', data['Ts'], c, c)
        elif rule == 'exact':
            adv = exact_adv(c, data)
        else:
            adv = data['Pq'].T @ adv_pts(jnp.einsum('msr,r->ms', data['G7'], c))
        return (Ac - prev + DT * (adv + nu * lam * Ac)) / (1.0 + DT * nu * lam)

    d = Rp if kind == 'span' else K
    lm = make_fused_lm(evalJ, d, step_budget, trust, gtol, clip=True)
    hfit = make_head_fit(fit_budget) if kind == 'head' else None

    def project(u0, data):
        return jnp.concatenate([b.T @ u0 for b in data['Qb']])        # Q_{R'}^T u0

    def decode(C, data):
        X = (data['Rq'] @ C.T)                                         # (R', 6)
        out, off = 0., 0
        for b in data['Qb']:
            w = b.shape[1]
            out = out + b @ X[off:off + w]
            off += w
        return out.T

    def query(u0, nu, data, hp):
        t = project(u0, data)
        scale = jnp.linalg.norm(t)
        if kind == 'span':
            w0 = jax.scipy.linalg.solve_triangular(data['Rq'], t, lower=False)
            ic = jnp.stack([0., 4., 0., jnp.linalg.norm(u0) ** 2 - scale ** 2])
        else:
            i0 = jnp.argmin(jnp.sum((data['RH'] - t[None, :]) ** 2, axis=1))
            w0, rn, it, reason, gn, _, _ = hfit(hp, data['Z'][i0], data['Rq'], t, Rp)
            ic = jnp.stack([it.astype(jnp.float64), reason.astype(jnp.float64), gn, rn])

        def step(carry, k):
            wv, wp, wp2, lam0 = carry
            p = data['A'] @ coef(wv, hp)
            args = (p, nu, data, hp)
            we = 2. * wv - wp
            wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
            cand = jnp.stack((wv, we, wq))
            rs = jax.vmap(lambda ww: jnp.linalg.norm(res(ww, args)))(cand)
            rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
            wi = cand[jnp.argmin(rs)]
            w2, rn, it, reason, gn, rej, lam = lm(wi, args, 1e-12 * scale, lam0)
            return (w2, wv, wp, lam), (w2, rn, it, reason, gn, rej)
        carry = (w0, w0, w0, jnp.asarray(1e-6, dtype=jnp.float64))
        _, (ws, rn, it, reason, gn, rej) = jax.lax.scan(step, carry, jnp.arange(STEPS))
        internal = jnp.concatenate((w0[None], ws))
        C = jax.vmap(lambda w: coef(w, hp))(internal[::KEEP])
        fields = decode(C, data)
        return fields, internal, it, reason, gn, rn, rej, ic
    return jax.jit(query), jax.jit(coef)


def make_query_fs(n, Rp, M, dt=DT, gtol=1e-3, trust=jnp.inf, unroll=True, first_sweeps=3, adaptive_first=0):
    """Amendment A2 fast path: span + tensor rule, the same scaled LSPG residual and the same LM step
    (Cholesky of J^T J + lam diag(J^T J), clipped, damping carried, accept only if the residual decreases), but a
    FIXED number of sweeps per time step after the predictor (3 on the first two steps, which have no
    extrapolation history, then 1) instead of a while loop, and the time loop unrolled, so the
    query is one straight-line program with no host round trips. The tensor is read twice per step: once for the
    three predictor candidates (their residuals AND the chosen candidate's Jacobian), once at the swept state.
    Stationarity is measured after the sweep and reported (reason 4 if ||J^T r|| <= gtol ||J||_F ||r||, else 0);
    it is not enforced. dt in {0.005, 0.01}: backward Euler of the FOM at that step (the FOM grid has the same knob).
    Output tuple identical to make_query."""
    steps = int(round(0.25 / dt))
    keep = int(round(0.05 / dt))
    assert abs(steps * dt - 0.25) < 1e-12 and abs(keep * dt - 0.05) < 1e-12

    def query(u0, nu, data, hp):
        A, Ts, lam_ = data['A'], data['Ts'], data['lam']
        S = 1.0 / (1.0 + dt * nu * lam_)
        t = jnp.concatenate([b.T @ u0 for b in data['Qb']])
        w0 = jax.scipy.linalg.solve_triangular(data['Rq'], t, lower=False)
        ic = jnp.stack([0., 4., 0., jnp.linalg.norm(u0) ** 2 - jnp.linalg.norm(t) ** 2])

        def rj(Ju, c, p):
            Ac = A @ c
            r = (Ac - p + dt * (0.5 * Ju @ c + nu * lam_ * Ac)) * S
            return r, A + (dt * S)[:, None] * Ju

        def sweep(w, r, J, rn, lam, p):
            Hm = J.T @ J
            g = J.T @ r
            d = jnp.diag(Hm) + 1e-30
            cf = jax.scipy.linalg.cho_factor(Hm + jnp.diag(lam * d), lower=True)
            dw = jax.scipy.linalg.cho_solve(cf, -g)
            nz = jnp.linalg.norm(dw)
            dw = dw * jnp.where(nz > trust, trust / (nz + 1e-300), 1.)
            ok = jnp.all(jnp.isfinite(dw))
            wn = w + jnp.where(ok, dw, 0.)
            r2, J2 = rj(jnp.einsum('mij,j->mi', Ts, wn), wn, p)
            rn2 = jnp.linalg.norm(r2)
            acc = ok & jnp.isfinite(rn2) & (rn2 < rn)
            lam2 = jnp.where(acc, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14))
            return (jnp.where(acc, wn, w), jnp.where(acc, r2, r), jnp.where(acc, J2, J), jnp.where(acc, rn2, rn),
                    lam2, (~acc).astype(jnp.int32))

        def make_step(nsweep):
            def step(carry, k):
                wv, wp, wp2, lam = carry
                p = A @ wv
                we = 2. * wv - wp
                wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
                cand = jnp.stack((wv, we, wq))
                Ju3 = jnp.einsum('mij,kj->kmi', Ts, cand)
                r3 = jax.vmap(lambda Ju, c: rj(Ju, c, p)[0])(Ju3, cand)
                rs = jnp.linalg.norm(r3, axis=1)
                rs = jnp.where(jnp.isfinite(rs), rs, jnp.inf)
                i = jnp.argmin(rs)
                w = cand[i]
                r, J = rj(Ju3[i], w, p)
                rn = jnp.linalg.norm(r)
                rej = jnp.int32(0)
                for _ in range(nsweep):
                    w, r, J, rn, lam, rj_ = sweep(w, r, J, rn, lam, p)
                    rej = rej + rj_
                gn = jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * rn + 1e-300)
                reason = jnp.where(jnp.isfinite(rn), jnp.where(gn <= gtol, 4, 0), 3).astype(jnp.int32)
                return (w, wv, wp, lam), (w, rn, jnp.int32(nsweep), reason, gn, rej)
            return step
        carry = (w0, w0, w0, jnp.asarray(1e-6, dtype=jnp.float64))
        if adaptive_first:
            # amendment A3: the first `adaptive_first` steps (no or short extrapolation history) use the adaptive
            # LM of make_query (while loop, same stopping test, budget 50); every later step one fixed sweep
            def evalJ(w, args):
                p_ = args
                return rj(jnp.einsum('mij,j->mi', Ts, w), w, p_)
            lmA = make_fused_lm(evalJ, Rp, 50, trust, gtol, clip=True)

            def astep(carry, k):
                wv, wp, wp2, lam = carry
                p = A @ wv
                we = 2. * wv - wp
                wq = jnp.where(k >= 2, 3. * wv - 3. * wp + wp2, we)
                cand = jnp.stack((wv, we, wq))
                Ju3 = jnp.einsum('mij,kj->kmi', Ts, cand)
                rs = jnp.linalg.norm(jax.vmap(lambda Ju, c: rj(Ju, c, p)[0])(Ju3, cand), axis=1)
                wi = cand[jnp.argmin(jnp.where(jnp.isfinite(rs), rs, jnp.inf))]
                w2, rn, it, reason, gn, rej, lam2 = lmA(wi, p, 1e-12 * jnp.linalg.norm(t), lam)
                return (w2, wv, wp, lam2), (w2, rn, it, reason, gn, rej)
            carry, o1 = jax.lax.scan(astep, carry, jnp.arange(adaptive_first))
            _, o2 = jax.lax.scan(make_step(1), carry, jnp.arange(adaptive_first, steps), unroll=unroll)
        else:
            carry, o1 = jax.lax.scan(make_step(first_sweeps), carry, jnp.arange(2), unroll=True)
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


def arm_data(mesh, kind, rule, Rp, M, lib=None):
    """Slice the mesh tables to one arm (prefix R', M). Copies are made once, offline."""
    edges = mesh['edges']
    nb = edges.index(Rp)
    d = dict(Qb=tuple(mesh['Qb'][:nb]), Rq=mesh['Rq'][:Rp, :Rp], A=mesh['A'][:M, :Rp], lam=mesh['lam'][:M])
    assert M <= mesh['A'].shape[0] and Rp <= mesh['A'].shape[1], (M, Rp, mesh['A'].shape)
    if rule == 'tensor':
        full = M == mesh['Tsym'].shape[0] and Rp == mesh['Tsym'].shape[1]
        d['Ts'] = mesh['Tsym'] if full else jnp.asarray(mesh['Tsym'][:M, :Rp, :Rp])      # no copy of the full table
    elif rule == 'exact':
        d['G'] = jnp.asarray(mesh['G'][:, :Rp])
    else:
        d['G7'] = jnp.asarray(mesh['G7'][:, :, :Rp])
        d['Pq'] = jnp.asarray(mesh['Pq'][:, :M])
    if kind == 'head':
        d['Z'] = mesh['library']['Z']
        d['RH'] = mesh['library']['H'][:, :Rp] @ mesh['Rq'][:Rp, :Rp].T
    return jax.tree_util.tree_map(lambda a: block(jnp.asarray(a)), d)


def make_rho(n, rule, M, kxyz):
    """Jitted rho over a batch of coefficient rows: rho(u) = ||rule(u) - Phi^T a_upwind(u)|| / ||Phi^T a_upwind(u)||
    for u = G c with G the arm's (N, R') ordered bank at this mesh; also the minimum decoded value per state."""
    idx = tuple(jnp.asarray(np.asarray(kxyz)[:M, a]) for a in range(3))
    dx = 1.0 / (n - 1)
    adv_pts = jax.vmap(lambda us: advect_points(us, dx))

    def one(c, G, data):
        u = G @ c
        ex = phiT(upwind(u, n)[None], n, idx)[0]
        if rule == 'tensor':
            ru = 0.5 * jnp.einsum('mij,i,j->m', data['Ts'], c, c)
        else:
            ru = data['Pq'].T @ adv_pts(jnp.einsum('msr,r->ms', data['G7'], c))
        return jnp.linalg.norm(ru - ex) / jnp.maximum(jnp.linalg.norm(ex), 1e-300), jnp.min(u)

    @jax.jit
    def batch(C, G, data):
        return jax.lax.map(lambda c: one(c, G, data), C)
    return batch


def rel_errors(fields, ref):
    """Per-time ||u - ref|| / ||ref(t=0)|| (initial-normalised, the paper's Burgers metric)."""
    f = np.asarray(fields)
    r = np.asarray(ref)
    n0 = max(float(np.linalg.norm(r[0])), 1e-300)
    return np.linalg.norm(f - r, axis=1) / n0


def restrict_index(n, per_axis=16):
    """Interior indices of the OFFSET audit lattice x = (2k+1)/(2 per_axis), k = 0..per_axis-1 (16^3 nodes common
    to every mesh, disjoint from the lat16 EQ lattice x = k/16)."""
    s = (n - 1) // (2 * per_axis)
    assert s >= 1 and s * 2 * per_axis == n - 1, n
    k = (2 * np.arange(per_axis) + 1) * s
    I, J, K = np.meshgrid(k, k, k, indexing='ij')
    ni = n - 2
    return (((I - 1) * ni + (J - 1)) * ni + (K - 1)).ravel()
