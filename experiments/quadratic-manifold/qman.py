"""The quadratic-manifold trial map (Geelen-Wright-Willcox 2022; Barnett-Farhat 2022).

    u(a) = u_ref + V_r a + W vech(a a^T),      a in R^r,

fitted offline by regularised least squares from the SAME truth snapshots the panel's
classical POD-LSPG arms use. There is no network and no training run: `V_r` is a POD basis
of the centred snapshots and `W` is one ridge-regularised linear solve against the residual
the linear term leaves.

Online the map is handed to the panel's own machinery unchanged: the columns
`B = [u_ref | V_r | W]` become a `GridBank`, the head is

    eta(a) = [1, a, vech(a a^T)]     so    B eta(a) = u_ref + V_r a + W vech(a a^T),

and `arms.make_query` builds the same weak residual, the same test projection, the same
initializer and the same damped Levenberg-Marquardt driver every other reduced subject uses.
The Jacobian V + 2 W (a (x) .) comes out of `jax.jacfwd` of that head, so no solver, residual
or quadrature code is duplicated here. `quadratic=False` drops the W block and gives the
affine linear-subspace control u_ref + V_r a on the identical basis.
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from ablation import pod_basis


def terms(r):
    """Number of distinct quadratic monomials, P = r(r+1)/2."""
    return int(r * (r + 1) // 2)


def head(r, quadratic):
    """eta(a), the coefficient map against the columns [u_ref | V_r | (W)]."""
    one = jnp.ones(1)
    if not quadratic:
        return lambda a: jnp.concatenate((one, a))
    iu, ju = np.triu_indices(int(r))
    iu, ju = jnp.asarray(iu), jnp.asarray(ju)
    return lambda a: jnp.concatenate((one, a, a[iu] * a[ju]))


def fit(Ut, r, gammas, seed, holdout, states_per_trajectory):
    """Fit (u_ref, V_r, W) on the snapshot matrix `Ut` of shape (n, Ns).

    The ridge gamma is chosen on a seeded held-out split of the TRAINING SNAPSHOTS -- never on
    the evaluation cohort -- by relative reconstruction error of the quadratic term against the
    linear term's residual, then W is refitted on all snapshots at the chosen gamma. Every
    gamma's held-out error is recorded, so the selection is auditable from the JSON alone.

    The split is BY TRAJECTORY, not by snapshot column (DESIGN A1, finding 5a). The snapshot
    matrix concatenates trajectories, so consecutive columns are states dt apart in a smooth
    viscous flow -- near-duplicates. A uniform column holdout leaves almost every held-out
    column's own temporal neighbours in the training half, the criterion cannot see overfitting,
    and it rewards interpolation: at 64 intervals it chose gamma = 0 at r = 8, whose W then
    reconstructed the snapshots 800x better and made the ROM arm non-convergent, 25 % WORSE than
    its own linear control and 10x more expensive (`checks/probe64-*.json`). Holding out whole
    trajectories removes that leak.
    """
    t0 = time.perf_counter()
    r = int(r)
    n, Ns = Ut.shape
    per = int(states_per_trajectory)
    assert per > 0 and Ns % per == 0, (Ns, per)
    P = terms(r)
    uref = jnp.mean(Ut, axis=1)
    Sc = Ut - uref[:, None]
    V, eig, energy = pod_basis(Sc, r)                      # (n, r), centred POD
    coefficients = V.T @ Sc                                # (r, Ns)
    E = Sc - V @ coefficients                              # what the linear term leaves
    iu, ju = np.triu_indices(r)
    Pi = coefficients[jnp.asarray(iu)] * coefficients[jnp.asarray(ju)]      # (P, Ns)

    rng = np.random.default_rng(int(seed))
    traj = np.arange(Ns) // per
    ntraj = int(traj[-1]) + 1
    order = rng.permutation(ntraj)
    nte_traj = max(1, int(round(float(holdout) * ntraj)))
    held = np.isin(traj, order[:nte_traj])
    te, tr = np.flatnonzero(held), np.flatnonzero(~held)
    assert len(te) and len(tr)
    Ptr, Pte = Pi[:, tr], Pi[:, te]
    Mtr = Ptr @ Ptr.T
    scale = float(jnp.trace(Mtr)) / P
    Ctr = E[:, tr] @ Ptr.T
    eye = jnp.eye(P)
    norm_te = float(jnp.linalg.norm(Sc[:, te]))
    trace, best = [], None
    for g in gammas:
        Wg = jnp.linalg.solve(Mtr + float(g) * scale * eye, Ctr.T).T
        err = float(jnp.linalg.norm(Wg @ Pte - E[:, te])) / max(norm_te, 1e-300)
        finite = bool(np.isfinite(err))
        trace.append(dict(gamma=float(g), heldout_relative=(err if finite else None), finite=finite))
        # A non-finite score must never win. `best` used to be seeded by the FIRST gamma, and the
        # grid starts at 0: one NaN there locked gamma = 0 in with a NaN W, which then failed the
        # driver's finiteness assertion in the middle of the timed loop (DESIGN A1, finding 5b).
        if finite and (best is None or err < best[1]):
            best = (float(g), err)
        del Wg
    assert best is not None, f'every ridge in {list(gammas)} gave a non-finite held-out score'
    gamma, heldout = best
    Mfull = Pi @ Pi.T
    scale_full = float(jnp.trace(Mfull)) / P          # the refit is scaled by its OWN Gram
    W = jnp.linalg.solve(Mfull + gamma * scale_full * eye, (E @ Pi.T).T).T   # (n, P), all snapshots
    jax.block_until_ready(W)
    assert bool(jnp.all(jnp.isfinite(W))), f'non-finite W at r={r}, gamma={gamma}'

    nrm = float(jnp.linalg.norm(Sc))
    linear_only = float(jnp.linalg.norm(E)) / max(nrm, 1e-300)
    quadratic_in = float(jnp.linalg.norm(W @ Pi - E)) / max(nrm, 1e-300)
    ev = np.asarray(jnp.linalg.eigvalsh(Mfull))
    # eigvalsh returns a small NEGATIVE smallest eigenvalue for a numerically singular PSD Gram,
    # which would make the headline conditioning diagnostic negative and meaningless; the sign is
    # reported beside a floored ratio instead (DESIGN A1, finding 5c).
    ev_min = float(ev[0])
    info = dict(rank=r, quadratic_terms=P, bank_columns=1 + r + P, snapshots=int(Ns),
                ridge=gamma, ridge_trace=trace, ridge_grid=[float(g) for g in gammas],
                heldout_relative=heldout, heldout_fraction=float(holdout),
                heldout_trajectories=int(nte_traj), trajectories=int(ntraj),
                heldout_snapshots=int(len(te)), states_per_trajectory=per,
                split='by trajectory (DESIGN A1)',
                selection_seed=int(seed), gram_scale=scale, gram_scale_full=scale_full,
                weight_frobenius_norm=float(jnp.linalg.norm(W)),
                gram_min_eigenvalue_negative=bool(ev_min < 0),
                gram_condition=float(ev[-1] / max(ev_min, np.finfo(float).tiny)), gram_min_eigenvalue=ev_min,
                snapshot_relative_linear_only=linear_only,
                snapshot_relative_with_quadratic=quadratic_in,
                pod_energy_total=float(energy), pod_eigenvalues=np.asarray(eig).tolist(),
                pod_tail_fraction=float(max(float(energy) - float(np.sum(np.asarray(eig))), 0.) / max(float(energy), 1e-300)),
                centred=True, basis='POD of the centred snapshots (u_ref = snapshot mean)',
                seconds=time.perf_counter() - t0)
    out = dict(uref=uref, V=V, W=W, coefficients=np.asarray(coefficients.T), info=info)
    del Sc, E, Pi, Ptr, Pte, Mtr, Mfull, Ctr
    return out


def bank_columns(m, quadratic):
    """B = [u_ref | V_r | W] (or [u_ref | V_r] without the quadratic block)."""
    cols = [m['uref'][:, None], m['V']] + ([m['W']] if quadratic else [])
    return jnp.concatenate(cols, axis=1)


def demo():
    """Self-check of the three things that can silently be wrong: the bank/head pair must
    evaluate to u_ref + V a + W vech(a a^T) exactly, `jax.jacfwd` of it must be the analytic
    V + 2 W (a (x) .), and the quadratic term must buy a real reduction on data that has one.

    Exact recovery of a planted (V, W) is NOT asserted and must not be: like GWW/BF, V is the
    POD basis of the data itself, so the quadratic part leaks into the leading modes and the
    linear part of the planted map is not the one the fit finds."""
    rng = np.random.default_rng(0)
    n, Ns, r = 60, 400, 3
    iu, ju = np.triu_indices(r)
    uref = rng.normal(size=n)
    V0, _ = np.linalg.qr(rng.normal(size=(n, r)))
    W0 = 0.05 * rng.normal(size=(n, terms(r)))
    a = rng.normal(size=(r, Ns))
    U = uref[:, None] + V0 @ a + W0 @ (a[iu] * a[ju])
    m = fit(jnp.asarray(U), r, (0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2), seed=0, holdout=0.2,
            states_per_trajectory=20)
    i = m['info']
    assert i['ridge'] in i['ridge_grid'] and 0 < i['heldout_relative'] < 1
    assert i['trajectories'] == 20 and i['heldout_trajectories'] == 4 and i['heldout_snapshots'] == 80
    assert i['split'].startswith('by trajectory') and i['weight_frobenius_norm'] > 0
    assert all(t['finite'] for t in i['ridge_trace'])
    # a grid of only non-finite scores must raise, not silently keep the first gamma
    try:
        fit(jnp.asarray(np.full_like(U, np.nan)), r, (0.,), seed=0, holdout=0.2, states_per_trajectory=20)
    except (AssertionError, FloatingPointError):
        pass
    else:
        raise AssertionError('a non-finite fit was accepted')
    assert i['snapshot_relative_with_quadratic'] < 0.6 * i['snapshot_relative_linear_only'], i
    assert i['bank_columns'] == 1 + r + terms(r) and i['quadratic_terms'] == 6

    B, h = bank_columns(m, True), head(r, True)
    z = jnp.asarray(rng.normal(size=r))          # a code the fit never saw
    want = np.asarray(m['uref']) + np.asarray(m['V']) @ np.asarray(z) \
        + np.asarray(m['W']) @ (np.asarray(z)[iu] * np.asarray(z)[ju])
    got = np.asarray(B @ h(z))
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-13, np.linalg.norm(got - want)
    Bl, hl = bank_columns(m, False), head(r, False)
    wl = np.asarray(m['uref']) + np.asarray(m['V']) @ np.asarray(z)
    assert np.linalg.norm(np.asarray(Bl @ hl(z)) - wl) / np.linalg.norm(wl) < 1e-13

    J = np.asarray(jax.jacfwd(lambda y: B @ h(y))(z))
    Jan = np.asarray(m['V']) + np.asarray(m['W']) @ np.stack(
        [np.eye(r)[i_][iu] * np.asarray(z)[ju] + np.asarray(z)[iu] * np.eye(r)[i_][ju] for i_ in range(r)], 1)
    assert np.linalg.norm(J - Jan) / np.linalg.norm(J) < 1e-13, np.linalg.norm(J - Jan) / np.linalg.norm(J)
    print('qman demo OK', {k: i[k] for k in ('ridge', 'heldout_relative', 'snapshot_relative_linear_only',
                                             'snapshot_relative_with_quadratic')})


if __name__ == '__main__':
    demo()
