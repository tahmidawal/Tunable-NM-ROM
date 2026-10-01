"""Periodic, exactly divergence-free coordinate-network bank for the NS3D co-moving ROM.

One network g_phi : T^3 -> R^{3 x R} defines the bank as a continuous function of the
coordinate; a mesh only decides where it is read. The network predicts a vector
potential psi(x) in R^{3 x R} from integer-frequency Fourier features (periodic by
construction, no boundary factor) and the bank columns are its curl,

    g_r(x) = curl psi_r(x),

so every column is divergence-free in the continuum. On a mesh n the column is read
at the n^3 grid nodes and scaled by n^{-3/2}, so that the Euclidean Gram matrix of the
sampled bank is the trapezoid rule for the L2(T^3) Gram matrix (exact for trigonometric
polynomials below the Nyquist frequency, hence the same on every resolving mesh).

Derivatives are available two ways and are compared in the gates:
  * autodiff: d_e g_r(x) by a forward-mode JVP of the curl (second derivatives of psi);
  * spectral: FFT derivative of the sampled column (what the operator build uses).

Arrays: a sampled bank is (3 n^3, R) in the (3, n, n, n) field layout of ns3d_fom.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ----------------------------------------------------------------- features --

def frequency_set(kmax):
    """Integer wave vectors with max-norm <= kmax, one of each +/- pair, by |k|^2."""
    ks = [(a, b, c) for a in range(-kmax, kmax + 1) for b in range(-kmax, kmax + 1)
          for c in range(-kmax, kmax + 1)
          if (a > 0 or (a == 0 and b > 0) or (a == 0 and b == 0 and c > 0))]
    ks = sorted(ks, key=lambda v: (v[0] ** 2 + v[1] ** 2 + v[2] ** 2, v))
    return np.asarray(ks, dtype=np.float64).T                       # (3, F)


def init_params(key, rank, width, depth, kmax, dtype=jnp.float64):
    B = frequency_set(kmax)
    sizes = [2 * B.shape[1]] + [width] * depth + [3 * rank]
    layers = []
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        scale = np.sqrt(2.0 / sizes[i]) if i < len(sizes) - 2 else np.sqrt(1.0 / sizes[i])
        layers.append((jax.random.normal(sub, (sizes[i], sizes[i + 1]), dtype=dtype) * scale,
                       jnp.zeros((sizes[i + 1],), dtype=dtype)))
    return dict(B=jnp.asarray(B, dtype=dtype), layers=layers)


def potential(p, x):
    """psi at points x (P, 3) -> (P, 3, R)."""
    angle = 2 * jnp.pi * (x @ p["B"])
    h = jnp.concatenate((jnp.sin(angle), jnp.cos(angle)), axis=-1)
    for w, b in p["layers"][:-1]:
        h = jax.nn.silu(h @ w + b)
    w, b = p["layers"][-1]
    out = h @ w + b
    return out.reshape(x.shape[0], 3, -1)


def _directional(fn, x, axis):
    tangent = jnp.zeros_like(x).at[:, axis].set(1.0)
    return jax.jvp(fn, (x,), (tangent,))[1]


def velocity(p, x):
    """Bank columns at points: curl psi, (P, 3, R). Pointwise, so a JVP with a unit
    tangent at every point is the partial derivative at every point."""
    fn = lambda y: potential(p, y)
    d = [_directional(fn, x, axis) for axis in range(3)]          # d[j][:, i] = d_j psi_i
    return jnp.stack((d[1][:, 2] - d[2][:, 1],
                      d[2][:, 0] - d[0][:, 2],
                      d[0][:, 1] - d[1][:, 0]), axis=1)


def velocity_gradient(p, x):
    """Autodiff d_e g at points: (3_e, P, 3, R)."""
    fn = lambda y: velocity(p, y)
    return jnp.stack([_directional(fn, x, axis) for axis in range(3)])


def divergence(p, x):
    """Autodiff div g at points (P, R): zero up to roundoff for a curl."""
    grad = velocity_gradient(p, x)
    return grad[0, :, 0] + grad[1, :, 1] + grad[2, :, 2]


def grid_points(n):
    s = np.arange(n, dtype=np.float64) / n
    return np.stack(np.meshgrid(s, s, s, indexing="ij"), axis=-1).reshape(-1, 3)


def make_sample_fn(n, chunk=None):
    """Un-jitted sample(p, pts) -> (3 n^3, R), scaled by n^{-3/2}; pts = grid_points(n)
    must be passed explicitly (never captured as a compile-time constant)."""
    total = n ** 3
    chunk = total if chunk is None else int(chunk)
    assert total % chunk == 0

    def sample(p, pts):
        def one(block):
            return velocity(p, block)                              # (chunk, 3, R)
        vals = jax.lax.map(one, pts.reshape(total // chunk, chunk, 3))
        vals = vals.reshape(total, 3, -1)
        return jnp.transpose(vals, (1, 0, 2)).reshape(3 * total, -1) * n ** -1.5
    return sample


def make_sampler(n, chunk=None):
    """jit'd sample(p) -> (3 n^3, R) for evaluation (points passed as a jit argument)."""
    pts = jnp.asarray(grid_points(n))
    fn = jax.jit(make_sample_fn(n, chunk))
    return lambda p: fn(p, pts)


def check_frequencies(p):
    """The Fourier frequencies are fixed integers (periodicity); never trained."""
    B = np.asarray(p["B"])
    kmax = int(np.max(np.abs(B)))
    if B.shape != frequency_set(kmax).shape or np.max(np.abs(B - frequency_set(kmax))) != 0.0:
        raise RuntimeError("Fourier frequencies are not the fixed integer set")
    return kmax


# ------------------------------------------------------------- the objective --

def projection_loss_exact(G, Y):
    """The same objective certified by a NumPy QR (no Gram matrix, no ridge)."""
    U, sv, _ = np.linalg.svd(np.asarray(G), full_matrices=False)
    rank = int(np.sum(sv > sv[0] * 1e-12))
    if rank < G.shape[1]:
        raise RuntimeError(f"bank is numerically rank-deficient: rank {rank} < {G.shape[1]}")
    Y = np.asarray(Y)
    W = U.T @ Y
    return float(1.0 - np.sum(W * W) / np.sum(Y * Y)), sv


def projection_loss(G, Y, ridge=1e-12):
    """Variable-projection objective: fraction of ||Y||_F^2 outside span(G).

    1 - tr(Y^T G (G^T G)^{-1} G^T Y) / ||Y||^2. With Y = U_K S_K (the weighted
    snapshot matrix compressed to its leading K singular pairs) this equals the
    weighted mean squared projection error of the training snapshots, up to the
    truncated tail. A relative ridge keeps the Cholesky defined; it can only
    increase the loss (it is a conservative estimate)."""
    M = G.T @ G
    B = G.T @ Y
    eps = ridge * jnp.trace(M) / M.shape[0]
    L = jnp.linalg.cholesky(M + eps * jnp.eye(M.shape[0], dtype=M.dtype))
    W = jax.scipy.linalg.solve_triangular(L, B, lower=True)
    return 1.0 - jnp.sum(W * W) / jnp.sum(Y * Y)


# ---------------------------------------------------- bank per mesh + order --

def leray_mask(G, n):
    """Project every column on the FOM's discrete space at mesh n (2/3 mask + Leray).

    Returns the projected bank and the fraction of each column's energy removed."""
    import ns3d_fom as F
    geom = F.geometry(n)
    r = G.shape[1]
    cols = jnp.asarray(G).T.reshape(r, 3, n, n, n)
    out = jax.vmap(lambda u: F.project_field(u, geom))(cols)
    out = out.reshape(r, -1).T
    removed = jnp.sum((jnp.asarray(G) - out) ** 2, 0) / jnp.maximum(jnp.sum(jnp.asarray(G) ** 2, 0), 1e-300)
    return np.asarray(out), np.asarray(removed)


def order_bank(G, states, row_norms):
    """The paper's ordering: G = Q R, rows R a_i / ||u_i|| = Q^T u_i / ||u_i||,
    SVD A = U S V^T, T = R^{-1} V, ordered bank G T = Q V (orthonormal).

    states (N, dof) training fields on this mesh, row_norms (N,) their norms.
    Returns T (R, R), singular values, Q V."""
    Q, Rf = np.linalg.qr(np.asarray(G), mode="reduced")
    A = (np.asarray(states) @ Q) / np.asarray(row_norms)[:, None]
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])[None, :]
    T = np.linalg.solve(Rf, V)
    return T, S, Q @ V
