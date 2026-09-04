"""3D viscous Burgers testbed for the separable EQ-decoder (2026-09-03, design
B3D-DESIGN.md r2, branch exp/2026-09-03-burgers3d-tensor).

    u_t + u (u_x + u_y + u_z) = nu lap(u)   on (0,1)^3,  u = 0 on the walls

FOM: interior-only unknowns (n-2)^3 with fixed ghost zeros, sign-upwind
advection on all three axes switching on the SAME centre value, 7-point
centred diffusion, backward Euler dt=0.005 x 50 steps, Newton with a
matrix-free BiCGStab preconditioned by the EXACT Helmholtz inverse in the
3D discrete sine basis (two separable 3D DSTs per application).  Weak form on
the M lowest 3D sine modes, which are exact eigenvectors of the ghost-zero
7-point Laplacian, so all linear terms are exact through A = Phi^T G and only
advection is left (sampled in the `ex` arm, a precomputed quadratic tensor in
the `tensor` arm, the full grid in the `full` arm).

Self-contained on purpose (the 1D precedent, b1d_common.py): importing
blat_common drags in the 2D FiLM stack.  nnls_capped and the _solve_nnls
sequence are copied verbatim from blat_common / ctol_eq (same algorithm, same
tolerances, same EQ seed); the decoder is the sep_common two-track model with
a 3D coordinate and bc(x) = 64 x(1-x) y(1-y) z(1-z); the trainer is the
sep_solvers.train_autodecoder_v2 recipe (explicit jit arguments, per-step
point subsampling) without its optional levers.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time

import numpy as np
import scipy.sparse as sps
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

F64 = jnp.float64

DT = 0.005
NUM_STEPS = 50
NEWTON_ITERS = 8            # truth generator: fixed iterations + skip guard
LIN_TOL = 1e-10
LIN_MAXITER = 2000
MAX_NEWTON = 20             # tolerance ladders
MAX_PICARD = 60
WEAK_ALPHA = 1.0
N_REF = 257                 # reference grid for the family's peak normalisation
EQ_ROWS = 3072              # ctol_eq NNLS row subsample
EQ_SEED = 20259
EQ_SNAPS = 64


def log(*a):
    print(*a, flush=True)


# ------------------------------- grid ---------------------------------------

def grid_coords_3d(n):
    """(n^3, 3) f64, flat index i*n^2 + j*n + k (x fastest-varying index i)."""
    x = np.linspace(0.0, 1.0, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)


def interior_indices_3d(n):
    """Flat full-grid indices of the interior nodes in interior order
    i*(n-2)^2 + j*(n-2) + k."""
    r = np.arange(1, n - 1)
    I, J, K = np.meshgrid(r, r, r, indexing="ij")
    return (I * n * n + J * n + K).reshape(-1)


def bc_poly_3d(x):
    """Smooth Dirichlet mask, max 1 at the centre: 64 x(1-x) y(1-y) z(1-z)."""
    return (64.0 * x[:, 0] * (1.0 - x[:, 0]) * x[:, 1] * (1.0 - x[:, 1])
            * x[:, 2] * (1.0 - x[:, 2]))


# ------------------------------- family --------------------------------------

def draw_param_table(seed, m):
    """ONE raw parameter table (design [A12]): every row consumes the same
    number of draws whatever its blob count, so prefixes are exact.  Returns
    dict of numpy arrays: B (m,), c (m,3,3), w (m,3), rho (m,3), A (m,),
    nu (m,)."""
    rng = np.random.default_rng(seed)
    B = np.zeros(m, dtype=np.int64)
    c = np.zeros((m, 3, 3))
    w = np.zeros((m, 3))
    rho = np.zeros((m, 3))
    A = np.zeros(m)
    nu = np.zeros(m)
    for j in range(m):
        B[j] = rng.integers(1, 4)
        for b in range(3):
            c[j, b] = rng.uniform(0.2, 0.8, 3)
            w[j, b] = rng.uniform(0.10, 0.20)
            rho[j, b] = rng.uniform(0.5, 1.0)
        A[j] = rng.uniform(0.5, 2.0)
        nu[j] = np.exp(rng.uniform(np.log(0.01), np.log(0.1)))
    return dict(B=B, c=c, w=w, rho=rho, A=A, nu=nu, seed=int(seed), m=int(m))


def _blob_sum(coords, row):
    """m(x) * sum_b rho_b exp(-|x-c_b|^2 / 2 w_b^2) at coords (P,3), jax."""
    x = jnp.asarray(coords, dtype=F64)
    s = jnp.zeros((x.shape[0],), dtype=F64)
    for b in range(int(row["B"])):
        d2 = jnp.sum((x - jnp.asarray(row["c"][b])[None, :]) ** 2, axis=1)
        s = s + float(row["rho"][b]) * jnp.exp(-d2 / (2.0 * float(row["w"][b]) ** 2))
    mask = (64.0 * x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])
            * x[:, 2] * (1 - x[:, 2]))
    return s * mask


_blob_sum_j = jax.jit(_blob_sum, static_argnames=())


def table_row(tab, j):
    return dict(B=int(tab["B"][j]), c=np.asarray(tab["c"][j]), w=np.asarray(tab["w"][j]),
                rho=np.asarray(tab["rho"][j]), A=float(tab["A"][j]), nu=float(tab["nu"][j]))


def peak_on_reference_grid(tab, chunk=2 ** 21):
    """s* = max over the N_REF^3 grid of the masked blob sum, per row
    (resolution-independent normalisation, design [A7]).  On device, in
    chunks of coordinates; deterministic."""
    coords = grid_coords_3d(N_REF)
    out = np.zeros(int(tab["m"]))
    for j in range(int(tab["m"])):
        row = table_row(tab, j)
        best = 0.0
        for s in range(0, coords.shape[0], chunk):
            best = max(best, float(jnp.max(_blob_sum(coords[s:s + chunk], row))))
        out[j] = best
    return out


def build_param_table(seed, m, path=None):
    """Draw, normalise, persist (npz) and fingerprint the table."""
    tab = draw_param_table(seed, m)
    tab["s_star"] = peak_on_reference_grid(tab)
    if path:
        np.savez(path, **{k: v for k, v in tab.items()})
    h = hashlib.sha256()
    for k in ("B", "c", "w", "rho", "A", "nu", "s_star"):
        h.update(np.ascontiguousarray(tab[k]).tobytes())
    tab["sha256"] = h.hexdigest()
    return tab


def load_param_table(path):
    d = np.load(path)
    tab = {k: d[k] for k in d.files}
    tab["seed"] = int(tab["seed"]); tab["m"] = int(tab["m"])
    h = hashlib.sha256()
    for k in ("B", "c", "w", "rho", "A", "nu", "s_star"):
        h.update(np.ascontiguousarray(tab[k]).tobytes())
    tab["sha256"] = h.hexdigest()
    return tab


def blob_ic_3d(n, tab, j, coords=None):
    """u0 on the full n^3 grid (walls exactly zero through the mask)."""
    coords = grid_coords_3d(n) if coords is None else coords
    row = table_row(tab, j)
    s = np.asarray(_blob_sum(coords, row))
    return row["A"] * s / float(tab["s_star"][j])


# ------------------------------- model --------------------------------------

def init_mlp(key, sizes):
    params = []
    for i in range(len(sizes) - 1):
        key, k1 = jax.random.split(key)
        w = jax.random.normal(k1, (sizes[i], sizes[i + 1]), dtype=F64) * jnp.sqrt(2.0 / sizes[i])
        params.append((w, jnp.zeros((sizes[i + 1],), dtype=F64)))
    return params


def apply_mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def init_separable_3d(key, k_lat, r_feat, n_ff=64, ff_scale=4.0, g_hidden=128,
                      g_layers=2, h_hidden=128, h_layers=2, out_scale=1.0):
    kb, kg, kh, kl = jax.random.split(key, 4)
    B = jax.random.normal(kb, (3, n_ff), dtype=F64) * ff_scale        # fixed frequencies
    g_mlp = init_mlp(kg, [2 * n_ff] + [g_hidden] * g_layers + [r_feat])
    h_mlp = init_mlp(kh, [k_lat] + [h_hidden] * h_layers + [r_feat])
    h_lin = jax.random.normal(kl, (k_lat, r_feat), dtype=F64) * 0.3
    return dict(B=B, g=g_mlp, h=h_mlp, h_lin=h_lin,
                out_scale=jnp.asarray(float(out_scale), dtype=F64))


def features(params, x):
    """bc(x) * g~(x): (n_pts, r).  ALL x-dependence of the decoder."""
    ang = 2.0 * jnp.pi * (x @ params["B"])
    ff = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    return (params["out_scale"] * bc_poly_3d(x))[..., None] * apply_mlp(params["g"], ff)


def head(params, z):
    return apply_mlp(params["h"], z) + z @ params["h_lin"]


class SeparableDecoder3D:
    def __init__(self, params, k_lat, r_feat):
        self.params = params
        self.k = int(k_lat)
        self.r = int(r_feat)

    def __call__(self, z, x):
        return features(self.params, x) @ head(self.params, z)

    def feat_at(self, x, chunk=0):
        """Bank builder (n_pts, r); chunking bounds the activation footprint and
        is numerically identical (features acts pointwise)."""
        x = jnp.asarray(x, dtype=F64)
        if not chunk or x.shape[0] <= chunk:
            return jnp.asarray(features(self.params, x))
        return jnp.concatenate([features(self.params, x[s:s + chunk])
                                for s in range(0, x.shape[0], chunk)], axis=0)

    def head_fn(self):
        p = self.params
        return lambda z: head(p, z)


# ------------------------------- FOM stencils ---------------------------------

def _pad3(u_int, n):
    ni = n - 2
    return jnp.pad(u_int.reshape(ni, ni, ni), 1)


def upwind_adv_field_3d(u_int, n):
    """N(u) = u (u_x + u_y + u_z), sign-upwind on every axis switching on the
    centre value, ghost zeros on all six faces.  Interior in, interior out."""
    dx = 1.0 / (n - 1)
    U = _pad3(u_int, n)
    c = U[1:-1, 1:-1, 1:-1]
    pos = c > 0
    ux = jnp.where(pos, (c - U[:-2, 1:-1, 1:-1]) / dx, (U[2:, 1:-1, 1:-1] - c) / dx)
    uy = jnp.where(pos, (c - U[1:-1, :-2, 1:-1]) / dx, (U[1:-1, 2:, 1:-1] - c) / dx)
    uz = jnp.where(pos, (c - U[1:-1, 1:-1, :-2]) / dx, (U[1:-1, 1:-1, 2:] - c) / dx)
    return (c * (ux + uy + uz)).reshape(-1)


def backward_adv_field_3d(u_int, n):
    """u (D-x u + D-y u + D-z u) with the FIXED backward branch (the form the
    tensor reproduces exactly)."""
    dx = 1.0 / (n - 1)
    U = _pad3(u_int, n)
    c = U[1:-1, 1:-1, 1:-1]
    d = ((c - U[:-2, 1:-1, 1:-1]) + (c - U[1:-1, :-2, 1:-1]) + (c - U[1:-1, 1:-1, :-2])) / dx
    return (c * d).reshape(-1)


def central_adv_field_3d(u_int, n):
    """Control stencil (gate F5 / gate A controls): central differences."""
    dx = 1.0 / (n - 1)
    U = _pad3(u_int, n)
    c = U[1:-1, 1:-1, 1:-1]
    d = ((U[2:, 1:-1, 1:-1] - U[:-2, 1:-1, 1:-1]) + (U[1:-1, 2:, 1:-1] - U[1:-1, :-2, 1:-1])
         + (U[1:-1, 1:-1, 2:] - U[1:-1, 1:-1, :-2])) / (2.0 * dx)
    return (c * d).reshape(-1)


def downwind_adv_field_3d(u_int, n):
    """Control stencil (gate F5 control): the upwind switch INVERTED
    (forward difference where u_c > 0) -- anti-diffusive, not monotone."""
    dx = 1.0 / (n - 1)
    U = _pad3(u_int, n)
    c = U[1:-1, 1:-1, 1:-1]
    pos = c > 0
    ux = jnp.where(pos, (U[2:, 1:-1, 1:-1] - c) / dx, (c - U[:-2, 1:-1, 1:-1]) / dx)
    uy = jnp.where(pos, (U[1:-1, 2:, 1:-1] - c) / dx, (c - U[1:-1, :-2, 1:-1]) / dx)
    uz = jnp.where(pos, (U[1:-1, 1:-1, 2:] - c) / dx, (c - U[1:-1, 1:-1, :-2]) / dx)
    return (c * (ux + uy + uz)).reshape(-1)


def lap_3d(u_int, n, zscale=1.0):
    """7-point Laplacian with ghost zeros; `zscale` scales the z-direction
    contribution (1.0 = the operator; != 1 only for gate controls)."""
    dx = 1.0 / (n - 1)
    U = _pad3(u_int, n)
    c = U[1:-1, 1:-1, 1:-1]
    lxy = (U[2:, 1:-1, 1:-1] + U[:-2, 1:-1, 1:-1] + U[1:-1, 2:, 1:-1] + U[1:-1, :-2, 1:-1] - 4.0 * c)
    lz = (U[1:-1, 1:-1, 2:] + U[1:-1, 1:-1, :-2] - 2.0 * c)
    return ((lxy + zscale * lz) / dx ** 2).reshape(-1)


def fom_residual_int(u_int, up_int, nu, n, adv="upwind", zscale=1.0, zadv=1.0):
    """Backward-Euler interior residual u - u_prev + dt (N(u) - nu lap u).
    `adv`, `zscale`, `zadv` exist ONLY for gate controls (defaults = the FOM)."""
    if adv == "upwind" and zadv == 1.0:
        Nu = upwind_adv_field_3d(u_int, n)
    elif adv == "central":
        Nu = central_adv_field_3d(u_int, n)
    elif adv == "downwind":
        Nu = downwind_adv_field_3d(u_int, n)
    else:                                                    # upwind with a scaled z term
        dx = 1.0 / (n - 1)
        U = _pad3(u_int, n)
        c = U[1:-1, 1:-1, 1:-1]
        pos = c > 0
        ux = jnp.where(pos, (c - U[:-2, 1:-1, 1:-1]) / dx, (U[2:, 1:-1, 1:-1] - c) / dx)
        uy = jnp.where(pos, (c - U[1:-1, :-2, 1:-1]) / dx, (U[1:-1, 2:, 1:-1] - c) / dx)
        uz = jnp.where(pos, (c - U[1:-1, 1:-1, :-2]) / dx, (U[1:-1, 1:-1, 2:] - c) / dx)
        Nu = (c * (ux + uy + zadv * uz)).reshape(-1)
    return u_int - up_int + DT * (Nu - nu * lap_3d(u_int, n, zscale))


# ------------------------------- sine basis / DST -----------------------------

def dst_matrix(n):
    """Orthonormal DST-I on the (n-2) interior points: S[i, p] =
    sqrt(2/(n-1)) sin(pi p i / (n-1)), p, i = 1..n-2.  S^T S = I."""
    p = np.arange(1, n - 1)
    return np.sqrt(2.0 / (n - 1)) * np.sin(np.pi * np.outer(p, p) / (n - 1))


def lam_1d(n):
    p = np.arange(1, n - 1)
    dx = 1.0 / (n - 1)
    return (4.0 / dx ** 2) * np.sin(np.pi * p / (2 * (n - 1))) ** 2


def lam_3d(n):
    l1 = lam_1d(n)
    return l1[:, None, None] + l1[None, :, None] + l1[None, None, :]     # (ni, ni, ni)


def dst3_mm(V, S):
    """Separable 3D DST by three one-axis matmuls: V (ni,ni,ni) -> S^T V S ... ;
    with orthonormal S the same call is its own inverse (S symmetric)."""
    V = jnp.einsum("ia,ajk->ijk", S, V)
    V = jnp.einsum("jb,ibk->ijk", S, V)
    return jnp.einsum("kc,ijc->ijk", S, V)


def _dst1_fft_axis(V, axis, n):
    """Orthonormal DST-I along one axis through the FFT of the odd extension:
    y = [0, x_1..x_ni, 0, -x_ni..-x_1] (length 2(n-1)); X_k = -Im FFT(y)_k / 2."""
    ni = n - 2
    V = jnp.moveaxis(V, axis, -1)
    shp = V.shape[:-1]
    zero = jnp.zeros(shp + (1,), dtype=V.dtype)
    y = jnp.concatenate([zero, V, zero, -V[..., ::-1]], axis=-1)      # length 2(n-1)
    X = -jnp.imag(jnp.fft.fft(y, axis=-1))[..., 1:ni + 1] / 2.0
    X = X * np.sqrt(2.0 / (n - 1))
    return jnp.moveaxis(X, -1, axis)


def dst3_fft(V, n):
    for ax in range(3):
        V = _dst1_fft_axis(V, ax, n)
    return V


def make_helmholtz_inv(n, dst="mm"):
    """H_nu^{-1} v = S (S^T v / (1 + dt nu lam)) on interior vectors, with
    H_nu = I + dt nu (-L).  Two separable 3D DSTs (six one-axis transforms)."""
    ni = n - 2
    S = jnp.asarray(dst_matrix(n))
    lam = jnp.asarray(lam_3d(n))

    def hinv(v, nu, nu_scale=1.0):
        V = v.reshape(ni, ni, ni)
        C = dst3_mm(V, S) if dst == "mm" else dst3_fft(V, n)
        C = C / (1.0 + DT * (nu * nu_scale) * lam)
        Y = dst3_mm(C, S) if dst == "mm" else dst3_fft(C, n)
        return Y.reshape(-1)
    return hinv


# ------------------------------- truth generator ------------------------------

def make_truth_rollout(n, dst="mm"):
    """Vmapped fixed-iteration Newton (NEWTON_ITERS) with the exact-Helmholtz
    preconditioned BiCGStab, skip guard on converged residuals, non-finite
    steps rejected, NaN-propagating residual audit.  Interior unknowns only.
    rollout(U0_int (B, n_i), nu (B,)) -> snaps (B, T+1, n_i), worst rel res."""
    hinv = make_helmholtz_inv(n, dst)

    def newton_step(u_int, up_int, nu):
        def body(u, _):
            r = fom_residual_int(u, up_int, nu, n)

            def Jv(v):
                return jax.jvp(lambda uu: fom_residual_int(uu, up_int, nu, n), (u,), (v,))[1]

            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=LIN_TOL, maxiter=LIN_MAXITER, M=lambda v: hinv(v, nu))
            ok = jnp.all(jnp.isfinite(du)) & \
                (jnp.linalg.norm(r) > 1e-12 * (jnp.linalg.norm(up_int) + 1e-300))
            return jnp.where(ok, u + du, u), None
        u, _ = jax.lax.scan(body, u_int, None, length=NEWTON_ITERS)
        rfin = jnp.linalg.norm(fom_residual_int(u, up_int, nu, n)) \
            / (jnp.linalg.norm(up_int) + 1e-300)
        return u, rfin

    def rollout(U0_int, nu):
        def body(carry, _):
            u, worst = carry
            u2, r = jax.vmap(newton_step, in_axes=(0, 0, 0))(u, u, nu)
            return (u2, jnp.maximum(worst, jnp.max(r))), u2
        (uT, worst), traj = jax.lax.scan(
            body, (jnp.asarray(U0_int), jnp.asarray(0.0, F64)), None, length=NUM_STEPS)
        snaps = jnp.concatenate([jnp.asarray(U0_int)[None], traj], axis=0)
        return jnp.transpose(snaps, (1, 0, 2)), worst

    return jax.jit(rollout)


def make_truth_rollout_iters(n, iters, dst="mm"):
    """Same generator with a different fixed iteration count (gate F3 control)."""
    hinv = make_helmholtz_inv(n, dst)

    def newton_step(u_int, up_int, nu):
        def body(u, _):
            r = fom_residual_int(u, up_int, nu, n)
            Jv = lambda v: jax.jvp(lambda uu: fom_residual_int(uu, up_int, nu, n), (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=LIN_TOL, maxiter=LIN_MAXITER, M=lambda v: hinv(v, nu))
            ok = jnp.all(jnp.isfinite(du)) & \
                (jnp.linalg.norm(r) > 1e-12 * (jnp.linalg.norm(up_int) + 1e-300))
            return jnp.where(ok, u + du, u), None
        u, _ = jax.lax.scan(body, u_int, None, length=iters)
        rfin = jnp.linalg.norm(fom_residual_int(u, up_int, nu, n)) \
            / (jnp.linalg.norm(up_int) + 1e-300)
        return u, rfin

    def rollout(U0_int, nu):
        def body(carry, _):
            u, worst = carry
            u2, r = jax.vmap(newton_step)(u, u, nu)
            return (u2, jnp.maximum(worst, jnp.max(r))), u2
        (uT, worst), traj = jax.lax.scan(
            body, (jnp.asarray(U0_int), jnp.asarray(0.0, F64)), None, length=NUM_STEPS)
        snaps = jnp.concatenate([jnp.asarray(U0_int)[None], traj], axis=0)
        return jnp.transpose(snaps, (1, 0, 2)), worst
    return jax.jit(rollout)


def make_control_rollout_adv(n, adv, dst="mm"):
    """Gate F5 control: the same Newton generator with a NON-monotone advection
    stencil (adv = 'central' or 'downwind'); must produce negative values."""
    hinv = make_helmholtz_inv(n, dst)

    def newton_step(u_int, up_int, nu):
        def body(u, _):
            r = fom_residual_int(u, up_int, nu, n, adv=adv)
            Jv = lambda v: jax.jvp(lambda uu: fom_residual_int(uu, up_int, nu, n, adv=adv),
                                   (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=LIN_TOL, maxiter=LIN_MAXITER, M=lambda v: hinv(v, nu))
            ok = jnp.all(jnp.isfinite(du))
            return jnp.where(ok, u + du, u), None
        u, _ = jax.lax.scan(body, u_int, None, length=NEWTON_ITERS)
        rfin = jnp.linalg.norm(fom_residual_int(u, up_int, nu, n, adv=adv)) \
            / (jnp.linalg.norm(up_int) + 1e-300)
        return u, rfin

    def rollout(U0_int, nu):
        def body(carry, _):
            u, worst = carry
            u2, r = jax.vmap(newton_step)(u, u, nu)
            return (u2, jnp.maximum(worst, jnp.max(r))), u2
        (uT, worst), traj = jax.lax.scan(
            body, (jnp.asarray(U0_int), jnp.asarray(0.0, F64)), None, length=NUM_STEPS)
        snaps = jnp.concatenate([jnp.asarray(U0_int)[None], traj], axis=0)
        return jnp.transpose(snaps, (1, 0, 2)), worst
    return jax.jit(rollout)


def make_control_rollout_zadv(n, zadv, dst="mm"):
    """Gate F1 control: z-advection coefficient scaled (breaks axis symmetry)."""
    hinv = make_helmholtz_inv(n, dst)

    def newton_step(u_int, up_int, nu):
        def body(u, _):
            r = fom_residual_int(u, up_int, nu, n, zadv=zadv)
            Jv = lambda v: jax.jvp(lambda uu: fom_residual_int(uu, up_int, nu, n, zadv=zadv),
                                   (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=LIN_TOL, maxiter=LIN_MAXITER, M=lambda v: hinv(v, nu))
            ok = jnp.all(jnp.isfinite(du)) & \
                (jnp.linalg.norm(r) > 1e-12 * (jnp.linalg.norm(up_int) + 1e-300))
            return jnp.where(ok, u + du, u), None
        u, _ = jax.lax.scan(body, u_int, None, length=NEWTON_ITERS)
        rfin = jnp.linalg.norm(fom_residual_int(u, up_int, nu, n, zadv=zadv)) \
            / (jnp.linalg.norm(up_int) + 1e-300)
        return u, rfin

    def rollout(U0_int, nu):
        def body(carry, _):
            u, worst = carry
            u2, r = jax.vmap(newton_step)(u, u, nu)
            return (u2, jnp.maximum(worst, jnp.max(r))), u2
        (uT, worst), traj = jax.lax.scan(
            body, (jnp.asarray(U0_int), jnp.asarray(0.0, F64)), None, length=NUM_STEPS)
        snaps = jnp.concatenate([jnp.asarray(U0_int)[None], traj], axis=0)
        return jnp.transpose(snaps, (1, 0, 2)), worst
    return jax.jit(rollout)


# ------------------------------- classical ladders ----------------------------

def make_newton_tol_rollout(n, dst="mm"):
    """`newton` arm: tolerance-terminated Newton (stop when ||R|| <= ntol
    ||u_prev||, at most MAX_NEWTON), BiCGStab preconditioned by the exact
    Helmholtz inverse; (ntol, lin_tol) are runtime arguments.  Single
    trajectory, whole 50-step rollout on device.  Returns (snaps (T+1, n_i)
    incl. u0, iters per step, rel residual per step)."""
    hinv = make_helmholtz_inv(n, dst)

    def step(u_prev, nu, ntol, lin_tol):
        u_scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)

        def cond(s):
            _, it, rn = s
            return (rn > ntol * u_scale) & (it < MAX_NEWTON)

        def body(s):
            u, it, rn = s
            r = fom_residual_int(u, u_prev, nu, n)
            Jv = lambda v: jax.jvp(lambda uu: fom_residual_int(uu, u_prev, nu, n), (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=lin_tol, maxiter=LIN_MAXITER, M=lambda v: hinv(v, nu))
            ok = jnp.isfinite(du).all()
            u2 = u + jnp.where(ok, du, 0.0)
            rn2 = jnp.linalg.norm(fom_residual_int(u2, u_prev, nu, n))
            good = jnp.isfinite(rn2)
            u = jnp.where(good, u2, u)
            rn = jnp.where(good, rn2, rn)
            it2 = jnp.where(good & ok, it + 1, jnp.int32(MAX_NEWTON))
            return (u, it2, rn)

        rn0 = jnp.linalg.norm(fom_residual_int(u_prev, u_prev, nu, n))
        u, its, rn = jax.lax.while_loop(cond, body, (u_prev, jnp.int32(0), rn0))
        return u, its, rn / u_scale

    def roll(u0_int, nu, ntol, lin_tol):
        def body(u, _):
            u2, its, rel = step(u, nu, ntol, lin_tol)
            return u2, (u2, its, rel)
        _, (snaps, its, rels) = jax.lax.scan(body, u0_int, None, length=NUM_STEPS)
        return jnp.concatenate([u0_int[None], snaps], axis=0), its, rels

    return jax.jit(roll)


def make_picard_tol_rollout(n, dst="mm"):
    """`picard` arm (design [A28]): defect correction with the EXACT Helmholtz
    inverse, u_{k+1} = u_k - H_nu^{-1} R(u_k), started from the linear
    extrapolation 2 u^n - u^{n-1} (u^n at the first step), stopped when
    ||R|| <= ntol ||u_prev|| or after max_iter iterations.  Rungs: (ntol,
    MAX_PICARD) tolerance rungs, and the fixed-work rungs (0, k): ntol = 0
    never stops early, so k = 0 is 'extrapolation only' (zero work), k = 1
    exactly one defect correction, etc.  One stencil evaluation and one
    DST pair per iteration.  Returns (snaps, iters per step, rel residual)."""
    hinv = make_helmholtz_inv(n, dst)

    def step(u_prev, u_prev2, first, nu, ntol, max_iter):
        u_scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)
        u0 = jnp.where(first, u_prev, 2.0 * u_prev - u_prev2)
        rn0 = jnp.linalg.norm(fom_residual_int(u0, u_prev, nu, n))

        def cond(s):
            _, it, rn = s
            return (rn > ntol * u_scale) & (it < max_iter)

        def body(s):
            u, it, rn = s
            r = fom_residual_int(u, u_prev, nu, n)
            u2 = u - hinv(r, nu)
            rn2 = jnp.linalg.norm(fom_residual_int(u2, u_prev, nu, n))
            good = jnp.isfinite(rn2) & jnp.all(jnp.isfinite(u2))
            u = jnp.where(good, u2, u)
            rn = jnp.where(good, rn2, rn)
            it2 = jnp.where(good, it + 1, jnp.int32(MAX_PICARD))
            return (u, it2, rn)

        u, its, rn = jax.lax.while_loop(cond, body, (u0, jnp.int32(0), rn0))
        return u, its, rn / u_scale

    def roll(u0_int, nu, ntol, max_iter):
        def body(carry, k):
            u, u2 = carry
            un, its, rel = step(u, u2, k == 0, nu, ntol, max_iter)
            return (un, u), (un, its, rel)
        _, (snaps, its, rels) = jax.lax.scan(body, (u0_int, u0_int), jnp.arange(NUM_STEPS))
        return jnp.concatenate([u0_int[None], snaps], axis=0), its, rels

    return jax.jit(roll)


# ------------------------------- data ------------------------------------------

def build_truth(n, tab, rows, chunk, rollout, coords=None, keep_full=True, log_fn=log):
    """Regenerate the trajectories `rows` of the table at resolution n in
    chunks.  Returns (U_int (len(rows), T+1, n_i) if keep_full else None,
    worst rel residual, min u, max u, fraction of points <= 0, seconds)."""
    interior = interior_indices_3d(n)
    coords = grid_coords_3d(n) if coords is None else coords
    out, worst, umin, umax, nle, npts = [], 0.0, np.inf, -np.inf, 0, 0
    t0 = time.time()
    for s in range(0, len(rows), chunk):
        rr = rows[s:s + chunk]
        U0 = np.stack([blob_ic_3d(n, tab, j, coords)[interior] for j in rr])
        sn, wr = rollout(jnp.asarray(U0), jnp.asarray(tab["nu"][rr]))
        worst = max(worst, float(wr))
        umin = min(umin, float(jnp.min(sn[:, 1:])))
        umax = max(umax, float(jnp.max(sn)))
        nle += int(jnp.sum(sn[:, 1:] <= 0)); npts += int(sn[:, 1:].size)
        if keep_full:
            out.append(np.asarray(sn))
        del sn
    U = np.concatenate(out, axis=0) if keep_full else None
    return U, worst, umin, umax, nle / max(npts, 1), time.time() - t0


# ------------------------------- weak form -----------------------------------

def test_modes_3d(n, M):
    """M lowest 3D sine modes by DISCRETE eigenvalue (stable ties).  Returns
    kx, ky, kz (M,), Phi (n_i, M) with unit-2-norm columns (exact: orthonormal
    1D factors), lam_disc (M,)."""
    ni = n - 2
    l1 = lam_1d(n)
    lam = l1[:, None, None] + l1[None, :, None] + l1[None, None, :]
    order = np.argsort(lam.reshape(-1), kind="stable")[:M]
    kx, ky, kz = np.unravel_index(order, (ni, ni, ni))
    kx, ky, kz = kx + 1, ky + 1, kz + 1
    S = dst_matrix(n)                                                  # (ni, p)
    Phi = np.empty((ni * ni * ni, M))
    for m_ in range(M):
        Phi[:, m_] = np.einsum("i,j,k->ijk", S[:, kx[m_] - 1], S[:, ky[m_] - 1],
                               S[:, kz[m_] - 1]).reshape(-1)
    return kx, ky, kz, Phi, lam.reshape(-1)[order]


def assemble_L_3d(n):
    """Independent scipy-sparse 7-point ghost-zero Laplacian on the interior
    (for gates F2, F4, F6, L)."""
    ni = n - 2
    dx = 1.0 / (n - 1)
    idx = np.arange(ni ** 3).reshape(ni, ni, ni)
    rows, cols, vals = [idx.reshape(-1)], [idx.reshape(-1)], [np.full(ni ** 3, -6.0)]
    for ax in range(3):
        for sgn in (1, -1):
            sl_from = [slice(None)] * 3
            sl_to = [slice(None)] * 3
            if sgn == 1:
                sl_from[ax] = slice(0, ni - 1); sl_to[ax] = slice(1, ni)
            else:
                sl_from[ax] = slice(1, ni); sl_to[ax] = slice(0, ni - 1)
            rows.append(idx[tuple(sl_from)].reshape(-1))
            cols.append(idx[tuple(sl_to)].reshape(-1))
            vals.append(np.ones(rows[-1].size))
    L = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(ni ** 3, ni ** 3))
    return L / dx ** 2


def assemble_Dminus_3d(n):
    """Independent scipy-sparse D-x + D-y + D-z (fixed backward branch, ghost zeros)."""
    ni = n - 2
    dx = 1.0 / (n - 1)
    idx = np.arange(ni ** 3).reshape(ni, ni, ni)
    rows, cols, vals = [idx.reshape(-1)], [idx.reshape(-1)], [np.full(ni ** 3, 3.0)]
    for ax in range(3):
        sl_from = [slice(None)] * 3
        sl_to = [slice(None)] * 3
        sl_from[ax] = slice(1, ni); sl_to[ax] = slice(0, ni - 1)
        rows.append(idx[tuple(sl_from)].reshape(-1))
        cols.append(idx[tuple(sl_to)].reshape(-1))
        vals.append(-np.ones(rows[-1].size))
    D = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(ni ** 3, ni ** 3))
    return D / dx


# ------------------------------- NNLS (verbatim) -------------------------------

def nnls_capped(G, b, max_support, tol=1e-10, inner_max=200):
    """Lawson-Hanson active-set NNLS that STOPS when the support reaches
    max_support (ECSW-style) or at optimality.  Verbatim blat_common."""
    n = G.shape[1]
    w = np.zeros(n)
    P = np.zeros(n, bool)
    r = b - G @ w
    outer = 0
    while outer < 5 * max_support + 10:
        grad = G.T @ r
        cand = np.where(~P)[0]
        if cand.size == 0 or P.sum() >= max_support:
            break
        j = cand[np.argmax(grad[cand])]
        if grad[j] <= tol * (np.linalg.norm(b) + 1e-300):
            break
        P[j] = True
        outer += 1
        for _ in range(inner_max):
            idx = np.where(P)[0]
            s_, *_ = np.linalg.lstsq(G[:, idx], b, rcond=None)
            if np.all(s_ > 0):
                w[:] = 0.0
                w[idx] = s_
                break
            neg = s_ <= 0
            alpha = np.min(w[idx][neg] / (w[idx][neg] - s_[neg] + 1e-300))
            w[idx] = w[idx] + alpha * (s_ - w[idx])
            P[idx[w[idx] <= 1e-14]] = False
            w[~P] = 0.0
        r = b - G @ w
    return w, float(np.linalg.norm(r)), outer


def candidate_pool(n_i, cap, seed=EQ_SEED):
    if n_i <= cap:
        return np.arange(n_i)
    return np.sort(np.random.default_rng(seed).choice(n_i, size=cap, replace=False))


def eq_fit_adv_3d(u_full_int, adv_full, Phi, cand_pos, Z_snap, m, label):
    """exlin_common.eq_fit_burgers_adv + ctol_eq._solve_nnls in one place:
    advection-only rows Phi_c^T * N(u)|cand, targets Phi^T N(u); row scaling;
    capped Lawson-Hanson on an EQ_ROWS subsample; support padding on mean |N|;
    nonnegative refit on ALL rows.  Returns keep (into cand_pos), w, info."""
    t0 = time.time()
    r_eq = np.random.default_rng(EQ_SEED)
    n_s = Z_snap.shape[0]
    pick = r_eq.choice(n_s, size=min(EQ_SNAPS, n_s), replace=False)
    Phi_c = Phi[cand_pos]
    Gs, bs, snap_c = [], [], []
    for i in pick:
        uf = np.asarray(u_full_int(jnp.asarray(Z_snap[i])))
        Nf = np.asarray(adv_full(jnp.asarray(uf)))
        bs.append(Phi.T @ Nf)
        Gs.append(Phi_c.T * Nf[cand_pos][None, :])
        snap_c.append(Nf[cand_pos])
    pad_score = np.abs(np.stack(snap_c)).mean(0)
    G = np.concatenate(Gs, axis=0)
    b = np.concatenate(bs)
    sc = np.linalg.norm(G, axis=1) + 1e-300
    G = G / sc[:, None]
    b = b / sc
    n_c = G.shape[1]
    rows = r_eq.choice(G.shape[0], size=min(G.shape[0], EQ_ROWS), replace=False)
    wts, rnorm, _ = nnls_capped(G[rows], b[rows], max_support=m)
    supp = np.nonzero(wts > 0)[0]
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]
        padded = 0
    else:
        rest = np.setdiff1d(np.arange(n_c), supp)
        pad = rest[np.argsort(-pad_score[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad])
        padded = len(pad)
    Gk = G[:, keep]
    wq, _, _ = nnls_capped(Gk, b, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    res = Gk @ wq - b
    rel_rows = np.abs(res) / (np.abs(b) + 1e-300)
    info = dict(m=int(len(keep)), support=int(len(supp)), padded=int(padded),
                rnorm_capped=float(rnorm), rnorm_final=float(np.linalg.norm(res)),
                rel_fit=float(np.linalg.norm(res) / (np.linalg.norm(b) + 1e-300)),
                row_rel_median=float(np.median(rel_rows)),
                row_rel_p95=float(np.quantile(rel_rows, 0.95)),
                row_rel_max=float(np.max(rel_rows)), n_rows_total=int(G.shape[0]),
                n_rows_fit=int(len(rows)), n_cand=int(n_c), eq_snaps=EQ_SNAPS,
                eq_rows=EQ_ROWS, eq_seed=EQ_SEED, secs=time.time() - t0,
                kind="weak_burgers_adv_only")
    log(f"  NNLS-EQ {label}: pool {n_c} support {len(supp)} (+{padded} pad) rel fit "
        f"{info['rel_fit']:.2e} (row p95 {info['row_rel_p95']:.1e}, max "
        f"{info['row_rel_max']:.1e}) [{info['secs']:.0f}s]")
    return keep, wq, info


# ------------------------------- training ------------------------------------

def train_autodecoder_3d(key, coords, U, k_lat, r_feat, steps=60000, lr=1e-3,
                         lam_orth=1e-4, p_sub=16384, log_every=5000, tag="",
                         recon_chunk=1024, **arch):
    """Joint Adam over (g, h, per-snapshot codes Z): the sep_solvers
    train_autodecoder_v2 recipe with the SAME sampling measure at every N
    (design [A24]): p_sub interior points drawn iid per step from the training
    pool `coords`, an unbiased estimate of the global relative-MSE loss, no
    full-grid finishing steps, no EMA, no weight decay.  U (S, P) and coords
    (P, 3) are EXPLICIT jit arguments (never captured)."""
    coords = jnp.asarray(coords, dtype=F64)
    U = jnp.asarray(U, dtype=F64)
    S, P = U.shape
    key, kz, kp = jax.random.split(key, 3)
    u_rms = float(jnp.sqrt(jnp.mean(U * U)))
    params = init_separable_3d(kp, k_lat, r_feat, out_scale=u_rms, **arch)
    Z = 0.1 * jax.random.normal(kz, (S, k_lat), dtype=F64)
    u_ms = jnp.mean(U * U)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-2)
    opt = optax.adam(sched)
    state = opt.init((params, Z))
    use_sub = 0 < p_sub < P

    def loss_at(pz, U_, C_):
        p, z = pz
        G = features(p, C_)
        H = head(p, z)
        err = H @ G.T - U_
        rel = jnp.mean(err * err) / u_ms
        C = (G.T @ G) / (G.shape[0] * p["out_scale"] ** 2)
        orth = jnp.mean((C - jnp.eye(C.shape[0], dtype=F64)) ** 2)
        return rel + lam_orth * orth, rel

    def _apply(pz, st, U_, C_):
        (val, rel), grads = jax.value_and_grad(loss_at, has_aux=True)(pz, U_, C_)
        grads[0]["out_scale"] = jnp.zeros_like(grads[0]["out_scale"])
        upd, st = opt.update(grads, st)
        return optax.apply_updates(pz, upd), st, rel

    @jax.jit
    def step_sub(pz, st, k_, U_all, C_all):
        pts = jax.random.choice(k_, P, shape=(p_sub,), replace=False)
        return _apply(pz, st, U_all[:, pts], C_all[pts])

    @jax.jit
    def step_full(pz, st, U_all, C_all):
        return _apply(pz, st, U_all, C_all)

    pz = (params, Z)
    t0 = time.time()
    rel = jnp.inf
    for i in range(steps):
        if use_sub:
            key, k_ = jax.random.split(key)
            pz, state, rel = step_sub(pz, state, k_, U, coords)
        else:
            pz, state, rel = step_full(pz, state, U, coords)
        if (i + 1) % log_every == 0 or i == 0:
            log(f"   train3d[{tag}] step {i+1:6d}/{steps}  rel-MSE {float(rel):.3e}  "
                f"[{time.time()-t0:.0f}s]")
    params, Z = pz
    G = features(params, coords)
    H = head(params, Z)
    per = []
    for s in range(0, S, recon_chunk):
        Uh = H[s:s + recon_chunk] @ G.T
        per.append(jnp.linalg.norm(Uh - U[s:s + recon_chunk], axis=1)
                   / jnp.linalg.norm(U[s:s + recon_chunk], axis=1))
    per = jnp.concatenate(per)
    info = dict(final_rel_mse=float(rel), steps=steps, lr=lr, lam_orth=lam_orth,
                p_sub=int(p_sub), used_subsampling=bool(use_sub), seconds=time.time() - t0,
                recon_rel_l2_mean=float(jnp.mean(per)), recon_rel_l2_max=float(jnp.max(per)),
                n_snapshots=int(S), n_points=int(P), arch=dict(arch))
    log(f"   train3d[{tag}] done: recon rel-L2 (training pool) mean "
        f"{info['recon_rel_l2_mean']:.3e} max {info['recon_rel_l2_max']:.3e} "
        f"[{info['seconds']:.0f}s]")
    return params, np.asarray(Z), info


# ------------------------------- io ------------------------------------------

def save_pkl(path, params, Z_tr, cfg):
    host = jax.tree_util.tree_map(np.asarray, params)
    with open(path, "wb") as f:
        pickle.dump(dict(params=host, Z_tr=np.asarray(Z_tr), cfg=cfg), f)


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, d["params"]), d["Z_tr"], d["cfg"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()
