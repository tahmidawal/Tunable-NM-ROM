"""Shared pieces for training the Burgers head in one frozen separable bank.

Everything here serves the exact identity established in
`experiments/separable-decoder/sep_hfit.py`: with the spatial bank G frozen,

    ||G h - u||^2 = ||Lam^T h - a||^2 + f^2,      a = Lam^T c*,   Gram = Lam Lam^T,

so fitting the head against full fields and fitting the whitened head
q = Lam^T h against precomputed span coefficients are the SAME problem. The
bank is a cached (n x R) matrix at any fixed point set, so the whole training
ladder runs on (S x R) arrays instead of (S x n) fields.

The data generator, the family and the FOM are the evaluation lane's own
(`engines`), so a training trajectory and an evaluation case are drawn from one
generator and the disjointness assertion is the same one `ablation.py` makes.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A

F64 = jnp.float64


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


# --------------------------------------------------------------- the draw ----

def incumbent_draw(canonical=576, extra_traj=4032, canonical_seed=0, extra_seed=1000):
    """The incumbent checkpoint's own training trajectories, reproduced in this lane.

    `burgers2d_film.sample_params` and `engines.params_draw` are the same
    sequential RNG draw over the same ranges, so the 4608 trajectories job
    2837431 used are exactly this concatenation. The density ladder is a nested
    prefix of it, which makes density the only variable across the rungs.
    """
    return np.concatenate((e.params_draw(canonical_seed, canonical),
                           e.params_draw(extra_seed, extra_traj)))


def assert_disjoint(train_physical, evaluation_physical, label='training/eval'):
    for t in train_physical:
        for s in evaluation_physical:
            assert not np.allclose(t, s), f'{label} overlap'


# ----------------------------------------------------------- frozen bank ----

def bank_algebra(G, jitter=1e-12):
    """Cholesky of the frozen bank's Gram, and the exact whitening maps.

    Returns (Gram, Lam, Linv) with Gram = Lam Lam^T lower-triangular Cholesky
    and Linv = Lam^{-1}. Row conventions match `sep_hfit`: a = c @ Lam and
    c = a @ Linv, so `sep_hfit.to_q` / `to_h` apply unchanged.
    """
    G = jnp.asarray(G)
    R = G.shape[1]
    Gram = G.T @ G
    eps = jitter * jnp.trace(Gram) / R
    Lam = jnp.linalg.cholesky(Gram + eps * jnp.eye(R, dtype=F64))
    Linv = jax.scipy.linalg.solve_triangular(Lam, jnp.eye(R, dtype=F64), lower=True)
    return Gram, Lam, Linv


def make_projector(G, Lam):
    """U (S, n) -> (a, un2, fl2). The floor is the residual's own norm, never a
    difference of two large squares, so it keeps full precision where the fit
    is good."""
    G = jnp.asarray(G)
    Lam = jnp.asarray(Lam)

    @jax.jit
    def project(U):
        GtU = G.T @ U.T                                           # (R, S)
        a = jax.scipy.linalg.solve_triangular(Lam, GtU, lower=True)
        c = jax.scipy.linalg.solve_triangular(Lam.T, a, lower=False)
        E = U - (G @ c).T
        return a.T, jnp.sum(U * U, axis=1), jnp.sum(E * E, axis=1)
    return project


# ------------------------------------------------------------ data stream ----

def generate_projected(L, dt, physical, project, ntol=1e-9, ltol=1e-7,
                       stride=1, progress=None, points=None):
    """Roll every trajectory out with the evaluation lane's FOM and keep only
    the whitened span coefficients. Fields are discarded per trajectory, so the
    resident cost is (states x R), not (states x n).

    `points`: optional flat interior indices; when given, the raw field values
    at those points are also retained (for the joint bank+head arms, which
    cannot use the identity because their bank moves)."""
    q, _ = e.make_fom(L, dt, None, .25, dt)
    keep = np.arange(0, int(round(.25 / dt)) + 1, stride)
    As, un2s, fl2s, traj, tidx, nus, pts = [], [], [], [], [], [], []
    worst = 0.
    t0 = time.perf_counter()
    pj = None if points is None else jnp.asarray(np.asarray(points))
    for i, phys in enumerate(physical):
        f, it, rn = q(jnp.asarray(e.initial(L, phys)), float(phys[4]), ntol, ltol)
        f = jax.block_until_ready(f)
        worst = max(worst, float(np.max(np.asarray(rn))))
        U = f[keep][:, 1:-1, 1:-1].reshape(len(keep), -1)
        a, un2, fl2 = project(U)
        As.append(np.asarray(a))
        un2s.append(np.asarray(un2))
        fl2s.append(np.asarray(fl2))
        if pj is not None:
            pts.append(np.asarray(U[:, pj]))
        traj.append(np.full(len(keep), i, dtype=np.int32))
        tidx.append(keep.astype(np.int32))
        nus.append(np.full(len(keep), float(phys[4])))
        del f, U
        if progress and (i + 1) % progress == 0:
            print(f'  GEN {i + 1}/{len(physical)} worst_fom_residual {worst:.2e} '
                  f'[{time.perf_counter() - t0:.0f}s]', flush=True)
    del q
    jax.clear_caches()
    assert np.isfinite(worst) and worst <= 1e-8, f'FOM residual {worst:.2e}'
    out = dict(a=np.concatenate(As), un2=np.concatenate(un2s), fl2=np.concatenate(fl2s),
               traj=np.concatenate(traj), t=np.concatenate(tidx), nu=np.concatenate(nus))
    info = dict(trajectories=int(len(physical)), states_per_trajectory=int(len(keep)),
                snapshots=int(out['a'].shape[0]), state_stride=int(stride),
                newton_tolerance=ntol, linear_tolerance=ltol,
                max_relative_residual=worst, seconds=time.perf_counter() - t0,
                whitened_target_sha256=sha_array(out['a']))
    if pj is not None:
        out['points'] = np.concatenate(pts)
    return out, info


def previous_index(traj, t):
    """For every state, the index of the preceding state of the same trajectory;
    -1 for the first state of a trajectory. The stride is 1 by design, so a
    consecutive pair is exactly one backward-Euler step apart."""
    order = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(traj, t))}
    return np.array([order.get((int(a), int(b) - 1), -1) for a, b in zip(traj, t)], dtype=np.int32)


def neighbour_index(physical, traj, t):
    """Same time index on the nearest OTHER trajectory in standardised parameter
    space: the code-smoothness term's parameter-adjacent partner."""
    P = np.asarray(physical, dtype=float).copy()
    P[:, 4] = np.log(P[:, 4])
    P = (P - P.mean(0)) / (P.std(0) + 1e-30)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argmin(d2, axis=1)
    order = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(traj, t))}
    return np.array([order.get((int(nn[a]), int(b)), i)
                     for i, (a, b) in enumerate(zip(traj, t))], dtype=np.int32)


# ------------------------------------------------- weak residual in q-space --

def weak_pieces(G, Linv, L, M):
    """The exact-linear weak operators, expressed for the WHITENED head.

    With q = Lam^T h, the state is u = G_w q and the projected state A h = A_w q,
    where A_w = (Phi^T G) Lam^{-T} and G_w = G Lam^{-T}. Nothing about the
    discretization changes; this is the same r_w `arms.weak_dense` evaluates,
    written in the parameterisation the trainer optimises."""
    Phi, lam, _ = e.modes(L, M)
    P = jnp.asarray(Phi)
    LinvT = jnp.asarray(Linv).T
    return dict(Phi=P, lam=jnp.asarray(lam), Aw=(P.T @ jnp.asarray(G)) @ LinvT,
                Gw=jnp.asarray(G) @ LinvT)


def weak_residual(wp, q, prev, nu, L, dt):
    """r_w for a batch: q (B, R) whitened coefficients, prev (B, M) the previous
    state's projection, nu (B,). Exact dense advection, matching `arms.weak_dense`."""
    u = q @ wp['Gw'].T                                             # (B, n)
    adv = jax.vmap(lambda v: e.spatial(v, L)[0])(u)                # (B, n)
    ah = q @ wp['Aw'].T                                            # (B, M)
    lam = wp['lam'][None, :]
    dtn = dt * nu[:, None]
    # the advection term carries dt alone; only the diffusion term carries nu
    return (ah - prev + dt * (adv @ wp['Phi']) + dtn * lam * ah) / (1 + dtn * lam)
