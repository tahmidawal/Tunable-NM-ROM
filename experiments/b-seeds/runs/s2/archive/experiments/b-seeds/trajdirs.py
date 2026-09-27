"""Trajectory-fitted correction directions, and the shared POD that both rules use.

The incumbent rule (`cheap-corrections/directions.py::audited`, itself
`ladder.residual_directions`) decomposes the *static* reconstruction residual

    rho_i^old = eta_i - h_theta(z*_i),      z*_i = argmin_z || R_G (h_theta(z) - eta_i) ||,

that is, what the frozen head cannot represent when it is allowed to fit each snapshot
on its own. It never asks what the head's TRAJECTORY does.

This module builds the trajectory analogue. For a training trajectory j, the same-mesh
full-order solver is run at the `fft_tight` setting with output spacing dt, giving the
converged state at EVERY internal step; the q = 0 ROM is run on the same supplied field,
giving its internal latents z_{j,t}; and the coefficient-space error is

    rho_{j,t}^traj = c*_{j,t} - h_theta(z_{j,t}),   c*_{j,t} = R_G^{-1} Q_G^T u^FOM_{j,t}.

`pod_from_residual` is the decomposition step, lifted out of `directions._finish` with
its arithmetic unchanged, so `old` and `traj` differ ONLY in which residual matrix is
decomposed. `smoke_qtd.py` checks that feeding it the incumbent's own rho reproduces
`directions.audited`'s matrix bitwise.
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


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


# ------------------------------------------------------------ the shared POD ---

def pod_from_residual(rho, Rb, coef_dim, q_ladder):
    """Field-metric POD of a residual matrix. `directions._finish`, arithmetic unchanged.

    rho      (S, R)  coefficient-space residuals
    Rb       (R, R)  the bank's thin-QR factor; ||G d|| == ||Rb d||
    returns  C (R, rank), Ct (R, rank) field-orthonormal, and the singular values.
    """
    tilde = jnp.asarray(rho) @ Rb.T
    _, sv, Vt = jnp.linalg.svd(tilde, full_matrices=False)
    rank = int(min(Vt.shape[0], coef_dim))
    Ct = Vt[:rank].T
    C = jnp.linalg.solve(Rb, Ct)
    total = float(jnp.sum(sv ** 2))
    captured = {str(q): float(np.sum(np.asarray(sv[:q]) ** 2)) / max(total, 1e-300)
                for q in q_ladder}
    info = dict(available_rank=rank, residual_rows=int(tilde.shape[0]),
                singular_values=np.asarray(sv[:min(len(sv), 600)]).tolist(),
                residual_energy_total=total, residual_energy_captured=captured,
                directions_sha256=sha_array(np.asarray(C)),
                field_orthonormal_sha256=sha_array(np.asarray(Ct)))
    return C, Ct, sv, tilde, info


# ------------------------------------------------- the trajectory residuals ----

def trajectory_residuals(query0, head0, bank_Qb, bank_Rb, physical, L, dt, ntol, ltol):
    """rho_{j,t} = c*_{j,t} - h_theta(z_{j,t}) for every trajectory j and step t.

    `query0` is the q = 0 ROM query (dense, M = 4K, retained contract); `head0` is
    h_theta. The FOM is run with output spacing dt so every internal step is available,
    and the fields are projected and discarded one trajectory at a time.
    """
    t0 = time.perf_counter()
    fom, _ = e.make_fom(L, dt, None, .25, dt)
    hv = jax.jit(jax.vmap(head0))
    rows, codes, per = [], [], []
    worst_fom = 0.
    for j, phys in enumerate(physical):
        u0 = e.initial(L, phys)
        nu = float(phys[4])
        f, it, rn = host(fom(jnp.asarray(u0), nu, ntol, ltol))
        assert np.isfinite(f).all(), j
        worst_fom = max(worst_fom, float(np.max(rn)))
        U = jnp.asarray(f[:, 1:-1, 1:-1].reshape(len(f), -1))           # (T, n)
        cstar = jnp.linalg.solve(bank_Rb, bank_Qb.T @ U.T).T            # (T, R)
        bank_rel = np.asarray(
            jnp.linalg.norm(U.T - bank_Qb @ (bank_Qb.T @ U.T), axis=0)
            / jnp.maximum(jnp.linalg.norm(U.T, axis=0), 1e-300))
        v = query0(jnp.asarray(u0), nu)
        Z = np.asarray(v[7])                                            # (T, K)
        assert Z.shape[0] == U.shape[0], (Z.shape, U.shape)
        H = hv(jnp.asarray(Z))                                          # (T, R)
        rho = np.asarray(cstar - H)
        rows.append(rho)
        codes.append(Z)
        step_norm = np.asarray(jnp.linalg.norm((cstar - H) @ bank_Rb.T, axis=1)
                               / jnp.maximum(jnp.linalg.norm(cstar @ bank_Rb.T, axis=1), 1e-300))
        per.append(dict(trajectory=int(j), steps=int(Z.shape[0]),
                        fom_max_relative_residual=float(np.max(rn)),
                        fom_newton_iterations_median=float(np.median(it)),
                        bank_projection_relative_max=float(np.max(bank_rel)),
                        rom_step_exit_reasons=np.asarray(v[3]).tolist(),
                        rom_budget_exits=int(np.sum(np.asarray(v[3]) == 0)),
                        rom_ic_reason=int(v[6]),
                        rom_iterations_median=float(np.median(np.asarray(v[1]))),
                        trajectory_relative_residual_per_step=step_norm.tolist(),
                        trajectory_relative_residual_max=float(np.max(step_norm))))
        del f, U, cstar
    del fom
    jax.clear_caches()
    P = np.concatenate(rows, axis=0)
    Zall = np.concatenate(codes, axis=0)
    info = dict(trajectories=int(len(physical)), steps_per_trajectory=int(rows[0].shape[0]),
                residual_rows=int(P.shape[0]), fom_ntol=ntol, fom_ltol=ltol,
                fom_max_relative_residual=worst_fom,
                relative_residual_median=float(np.median([p['trajectory_relative_residual_max']
                                                          for p in per])),
                relative_residual_worst=float(np.max([p['trajectory_relative_residual_max']
                                                      for p in per])),
                per_trajectory=per, seconds=time.perf_counter() - t0)
    return P, Zall, info


def build(query0, head0, Qb, Rb, physical, L, dt, ntol, ltol, coef_dim, q_ladder, rule):
    """One complete trajectory-fitted direction set."""
    t0 = time.perf_counter()
    P, Z, tinfo = trajectory_residuals(query0, head0, Qb, Rb, physical, L, dt, ntol, ltol)
    assert Z.shape[0] == P.shape[0], (Z.shape, P.shape)
    C, Ct, sv, tilde, info = pod_from_residual(P, Rb, coef_dim, q_ladder)
    info.update(rule=rule, trajectory=tinfo, seconds=time.perf_counter() - t0,
                codes_sha256=sha_array(Z), residual_sha256=sha_array(P))
    return C, Ct, Z, P, tilde, info


# -------------------------------------------------- direction-set comparisons --

def cross_capture(tilde, Ct, q_list):
    """|| P C_{:q} ||_F^2 / || P ||_F^2 : how much of residual matrix P the directions reach.

    `tilde` is the WHITENED residual matrix (rows R_G rho) and `Ct` the field-orthonormal
    directions, so the projection is an ordinary orthogonal projection.
    """
    T = jnp.asarray(tilde)
    Cm = jnp.asarray(Ct)
    total = float(jnp.sum(T * T))
    out = {}
    for q in q_list:
        if q == 0:
            out[str(q)] = 0.
            continue
        k = int(min(q, Cm.shape[1]))
        proj = T @ Cm[:, :k]
        out[str(q)] = float(jnp.sum(proj * proj)) / max(total, 1e-300)
    return out


def principal_angles(Ct_a, Ct_b, q_list):
    """Field-metric principal angles between two nested field-orthonormal direction sets.

    `arccos` is sqrt-ill-conditioned at sigma = 1, so a self-comparison returns angles of
    order 1e-6 degrees rather than exactly zero. The well-conditioned summary is
    `overlap` = mean(sigma^2), which is 1 exactly for identical subspaces.
    """
    A_ = jnp.asarray(Ct_a)
    B_ = jnp.asarray(Ct_b)
    out = {}
    for q in q_list:
        k = int(min(q, A_.shape[1], B_.shape[1]))
        if k == 0:
            continue
        s = np.asarray(jnp.linalg.svd(A_[:, :k].T @ B_[:, :k], compute_uv=False))
        s = np.clip(s, -1., 1.)
        ang = np.degrees(np.arccos(s))
        out[str(q)] = dict(q=int(k), overlap=float(np.sum(s ** 2) / k),
                           cos_max=float(np.max(s)), cos_min=float(np.min(s)),
                           angle_min_degrees=float(np.min(ang)),
                           angle_median_degrees=float(np.median(ang)),
                           angle_max_degrees=float(np.max(ang)),
                           angles_degrees=ang.tolist())
    return out


def prefix_hashes(C, q_list):
    """sha256 of the first q columns of C, for every q on the ladder.

    The arm builder slices `Cfull[:, :q]` and records the hash of the slice it was
    actually handed; the audit recomputes these from the saved full matrix and compares,
    so the nesting claim is checked against the artifact rather than asserted.
    """
    Cn = np.ascontiguousarray(np.asarray(C))
    return {str(q): sha_array(np.ascontiguousarray(Cn[:, :int(q)])) for q in q_list}
