"""spectral-pilot Part A: the POD-Galerkin feasibility comparator (DESIGN 4.2 + A1). PILOT: small, not a result.

Span: POD (method of snapshots, then QR re-orthonormalisation, so every prefix is the POD prefix) of the training
snapshots in sine-coefficient space at N_ref (Euclidean = continuous L2 up to the constant 1/4).
Per reduced size r: Psi = Phi_r V with Phi_r^T diag(lam) Phi_r = V D V^T (lam = -pi^2 |k|^2), so the reduced linear
part is nu * D (diagonal, nu-independent rotation), ETDRK4 as the FOM.
Nonlinear term, two forms (each scored and timed on its own):
  ROM-T  packed tensor:  N(z)_k = sum_{i<=j} Tp[k, p(i,j)] z_i z_j, T built from the FOM's own padded grid
         (M = 3 N_ref / 2), i.e. the Galerkin projection of pscore.nonlinear on span(Psi) (gate G4);
  ROM-Q  grid rule of (Mq-1)^2 points: N(z) = -(2/Mq)^2 Bq^T [(Bq z) * (Dq z)], Bq/Dq the values of psi and of
         (d_x + d_y) psi at x_j = j / Mq (exact synthesis by matrices); Mq = 3 N_ref / 2 reproduces ROM-T.
Error (continuous L2, exact): e_t^2 = ||z_t - Psi^T b_t||^2 + ||(I - Psi Psi^T) b_t||^2.
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

import common as C

P = C.P


# --------------------------------------------------------------------------------------------------- POD -------

def pod(X, rmax, chunk=256):
    """X (n_snap, n, n) -> Phi (n*n, rmax) orthonormal, singular values. Gram accumulated in f64 chunks."""
    S = X.reshape(len(X), -1)
    G = np.zeros((len(S), len(S)))
    for i in range(0, len(S), chunk):
        G[i:i + chunk] = S[i:i + chunk] @ S.T
    G = (G + G.T) / 2
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    rmax = int(min(rmax, np.sum(w > w[0] * 1e-14)))       # sigma/sigma_0 > 1e-7: above Gram round-off (code audit 1)
    Phi = S.T @ (U[:, :rmax] / np.sqrt(w[:rmax]))
    Q, R = np.linalg.qr(Phi)
    Q = Q * np.sign(np.diag(R))                       # keep orientation; prefixes span the POD prefixes
    return Q, np.sqrt(np.maximum(w, 0))


def lam_flat(n):
    k = np.arange(1, n + 1) * np.pi
    return -(k[:, None] ** 2 + k[None, :] ** 2).ravel()


def rotate(Phi_r, lam):
    """Psi = Phi_r V, D (eigenvalues of the reduced Laplacian, ascending magnitude order kept from eigh)."""
    Lr = Phi_r.T @ (lam[:, None] * Phi_r)
    Lr = (Lr + Lr.T) / 2
    D, V = np.linalg.eigh(Lr)
    return Phi_r @ V, D, V


# ------------------------------------------------------------------------------------ padded-grid values -------

def grid_values(B, N, M=None):
    """B (r, N-1, N-1) coefficient arrays -> (u, g) on the padded interior grid (default M = 3N/2, the arithmetic of
    pscore.nonlinear): u = synthesis, g = d_x u + d_y u. Shapes (r, M-1, M-1)."""
    M = M or 3 * N // 2
    bp = jnp.pad(B, ((0, 0), (0, M - N), (0, M - N)))
    n = M - 1
    S = lambda x, ax: P.along(P.sine_synth_last, x, ax, M)
    Cc = lambda x, ax: P.along(P.cos_synth_last, x, ax, M)

    def one(b):
        Ty = S(b, 1)
        u = S(Ty, 0)
        g = Cc(P.kvec(n, 2, 0) * Ty, 0) + S(Cc(P.kvec(n, 2, 1) * b, 1), 0)
        return u, g
    return jax.vmap(one)(bp)


def tensor_full(Phi, N, M=None):
    """T[k, i, j] = -(2/M)^2 sum_g phi_k phi_i (D phi_j) on the padded grid (default M = 3N/2). (r, r, r) numpy."""
    r = Phi.shape[1]
    n = N - 1
    M = M or 3 * N // 2
    U, Gd = [], []
    for i in range(0, r, 64):
        u, g = grid_values(jnp.asarray(Phi[:, i:i + 64].T.reshape(-1, n, n)), N, M)
        U.append(np.asarray(u).reshape(len(u), -1))
        Gd.append(np.asarray(g).reshape(len(g), -1))
    U = jnp.asarray(np.concatenate(U).T)          # (G, r)
    Gd = jnp.asarray(np.concatenate(Gd).T)
    fac = -(2.0 / M) ** 2

    @jax.jit
    def slab(U, Gd, k):                           # large arrays are explicit jit arguments (no captured constants)
        return fac * ((U * U[:, k][:, None]).T @ Gd)
    T = np.empty((r, r, r))
    for k in range(r):
        T[k] = np.asarray(slab(U, Gd, k))
    return T


def rotate_tensor(T, V):
    """T'[k,i,j] = sum V[a,k] V[b,i] V[c,j] T[a,b,c]."""
    T = np.tensordot(V.T, T, axes=(1, 0))                     # k b c
    T = np.tensordot(T, V, axes=(1, 0))                       # k c i
    T = np.tensordot(T, V, axes=(1, 0))                       # k i j
    return T


def pack(T):
    """Symmetrise in (i, j) and pack i <= j with the pair factor (2 off the diagonal)."""
    r = T.shape[0]
    iu, ju = np.triu_indices(r)
    Ts = (T + T.transpose(0, 2, 1)) / 2
    fac = np.where(iu == ju, 1.0, 2.0)
    return Ts[:, iu, ju] * fac[None, :], iu, ju


class NLOp:
    """A nonlinear term f(z, ops) with its operator arrays `ops` passed as explicit jit arguments; calling the object
    with z alone uses its own arrays (convenience for eager checks)."""

    def __init__(self, fn, ops):
        self.fn, self.ops = fn, ops

    def __call__(self, z):
        return self.fn(z, self.ops)


def _nl_t(z, o):
    return o[0] @ (z[o[1]] * z[o[2]])


def nl_tensor(Tp, iu, ju):
    return NLOp(_nl_t, (jnp.asarray(Tp), jnp.asarray(iu), jnp.asarray(ju)))


def nl_exact(Psi, N):
    """Psi^T pscore.nonlinear(Psi z) at N (gate G4)."""
    n = N - 1
    return NLOp(lambda z, o: o[0].T @ P.nonlinear((o[0] @ z).reshape(n, n), N, 2)[0].reshape(-1), (jnp.asarray(Psi),))


# ------------------------------------------------------------------------------------------ grid rule (Q) -------

def q_values(Psi, N, Mq):
    """Bq, Dq (m, r): psi and (d_x + d_y) psi at x_j = j / Mq, j = 1..Mq-1, by exact matrix synthesis."""
    n = N - 1
    k = np.arange(1, N)
    x = np.arange(1, Mq) / Mq
    S = np.sin(np.pi * np.outer(x, k))                         # (Mq-1, n)
    Cd = np.pi * k[None, :] * np.cos(np.pi * np.outer(x, k))   # d/dx of sin(k pi x)
    r = Psi.shape[1]
    Bq = np.empty(((Mq - 1) ** 2, r))
    Dq = np.empty(((Mq - 1) ** 2, r))
    Pr = Psi.T.reshape(r, n, n)
    for i in range(r):
        A = Pr[i]
        Bq[:, i] = (S @ A @ S.T).ravel()
        Dq[:, i] = (Cd @ A @ S.T + S @ A @ Cd.T).ravel()
    return Bq, Dq


def nl_quad(Bq, Dq, Mq):
    fac = -(2.0 / Mq) ** 2
    return NLOp(lambda z, o: fac * (o[0].T @ ((o[0] @ z) * (o[1] @ z))), (jnp.asarray(Bq), jnp.asarray(Dq)))


# ------------------------------------------------------------------------------------------ the ROM query -------

def make_rom_batch(NL, D, n):
    """Batched accuracy rollout: (z0 (B, r), nu (B,)) -> (B, 6, r). Operator arrays enter as jit arguments."""
    Dj = jnp.asarray(D)
    fn = NL.fn

    @jax.jit
    def batch(z0, nu, ops, Dj):
        return jax.vmap(lambda z, v: C.rollout(z, v * Dj, lambda x: fn(x, ops), n, 5))(z0, nu)
    return lambda z0, nu: batch(z0, nu, NL.ops, Dj)


def ic_project(Psi, N, D_cases, Ninit=None):
    """z0 = Psi^T b0 with b0 the IC coefficients truncated to Ninit (default N): (B, r)."""
    n = N - 1
    Ni = Ninit or N
    rule = C.ICRule(Ni)
    Pr = jnp.asarray(Psi.T.reshape(-1, n, n)[:, :Ni - 1, :Ni - 1])

    @jax.jit
    def one(d):
        gx, gy, a = rule.coefs(d)
        return a * jnp.einsum('kij,i,j->k', Pr, gx, gy)
    return np.stack([np.asarray(one(jnp.asarray(d))) for d in D_cases])


def rom_errors(Z, Cref, floor2, den):
    """Z (B, 6, r) ROM states; Cref (B, 6, r) = Psi^T b_ref; floor2 (B, 6) = ||(I - PP^T) b_ref||^2; den (B,).
    Returns per-case e (max over t_1..t_5) and per-time arrays."""
    d2 = np.sum((Z - Cref) ** 2, axis=-1) + floor2
    et = np.sqrt(d2) / den[:, None]
    e = np.max(et[:, 1:], axis=1)
    bad = ~np.all(np.isfinite(Z), axis=(1, 2))
    e[bad] = np.inf
    return e, et


def timer(fn, *args, reps=5):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t)
    return ts
