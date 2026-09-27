"""Streamed snapshot fits for the POD-LSPG and quadratic-manifold baselines (DESIGN.md section 3.1).

The algebra is exactly `ablation.pod_basis` (uncentred method of snapshots) and `qman.fit` (centred POD + one
ridge-regularised solve for W, ridge chosen on a held-out split BY TRAJECTORY). What changes is only where the
n x Ns snapshot matrix lives: on the host, visited in row blocks, so that at 2048^2 (n = 4.19e6, Ns = 3328,
111 GB) it never has to sit on the device, and so that the concatenation that builds it never doubles it.
Every device product is a sum over row blocks of the same products, so the results agree with the in-memory
versions up to summation order; `check_parity()` asserts that on a small mesh.

Also here: `LeanGridBank`, an `arms.GridBank` that does not keep the padded transposed copy of the whole bank on
the device (it pads column chunks on demand in `at`), so a 72 GB quadratic-manifold bank is held once, not
twice. `at` is bitwise the parent's (the same per-column bilinear sample).
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


def snapshots(L, dt, physical, stride, ntol, ltol):
    """`ladder.generate_snapshots` written into ONE preallocated host array Ut of shape (n, Ns)."""
    q, _ = e.make_fom(L, dt, None, .25, dt)
    keep = np.arange(0, int(round(.25 / dt)) + 1, stride)
    per, n = len(keep), (L - 1) ** 2
    Ut = np.empty((n, per * len(physical)), dtype=np.float64)
    worst = 0.
    t0 = time.perf_counter()
    h = hashlib.sha256()
    for j, phys in enumerate(physical):
        f, it, rn = q(jnp.asarray(e.initial(L, phys)), float(phys[4]), ntol, ltol)
        f = np.asarray(f)
        rn = np.asarray(rn)
        assert np.isfinite(f).all()
        worst = max(worst, float(np.max(rn)))
        block = np.ascontiguousarray(f[keep][:, 1:-1, 1:-1].reshape(per, -1))
        h.update(block.tobytes())                      # = sha of ladder's U (Ns, n), row-major
        Ut[:, j * per:(j + 1) * per] = block.T
        del f, block
    jax.clear_caches()
    return Ut, dict(trajectories=int(len(physical)), states_per_trajectory=int(per), snapshots=int(Ut.shape[1]),
                    state_stride=int(stride), newton_tolerance=ntol, linear_tolerance=ltol,
                    max_relative_residual=worst, seconds=time.perf_counter() - t0, snapshot_sha256=h.hexdigest(),
                    layout='host (n, Ns) float64; sha256 is of the (Ns, n) row-major matrix ladder.generate_snapshots returns')


def _blocks(n, Ns, target_bytes):
    nb = max(1, int(target_bytes // (8 * Ns)))
    return [(s, min(n, s + nb)) for s in range(0, n, nb)]


def pod(Ut, kmax, target_bytes=2e9):
    """ablation.pod_basis(Ut, kmax) with Ut on the host. Returns host modes (n, kmax), eigenvalues, total energy,
    and coords = Ut^T modes (Ns, kmax)."""
    n, Ns = Ut.shape
    bl = _blocks(n, Ns, target_bytes)
    gram = jnp.zeros((Ns, Ns))
    for s, t in bl:
        X = jnp.asarray(Ut[s:t])
        gram = gram + X.T @ X
    w, V = jnp.linalg.eigh(gram)
    w, V = w[::-1], V[:, ::-1]
    total = float(jnp.sum(jnp.clip(w, 0., None)))
    wk = jnp.clip(w[:kmax], 1e-300, None)
    Vs = V[:, :kmax] / jnp.sqrt(wk)[None, :]
    modes = np.empty((n, kmax))
    coords = jnp.zeros((Ns, kmax))
    for s, t in bl:
        X = jnp.asarray(Ut[s:t])
        mb = X @ Vs
        coords = coords + X.T @ mb
        modes[s:t] = np.asarray(mb)
    return modes, np.asarray(w[:kmax]), total, np.asarray(coords)


class CentredGram:
    """Pass 1 of the quadratic-manifold fit, shared by every rank: row means and the centred Gram."""

    def __init__(self, Ut, target_bytes=2e9):
        n, Ns = Ut.shape
        self.Ut, self.bl = Ut, _blocks(n, Ns, target_bytes)
        self.uref = np.empty(n)
        gram = jnp.zeros((Ns, Ns))
        for s, t in self.bl:
            X = jnp.asarray(Ut[s:t])
            m = jnp.mean(X, axis=1)
            self.uref[s:t] = np.asarray(m)
            S = X - m[:, None]
            gram = gram + S.T @ S
        w, V = jnp.linalg.eigh(gram)
        self.w, self.V = w[::-1], V[:, ::-1]
        self.energy = float(jnp.sum(jnp.clip(self.w, 0., None)))

    def centred(self, s, t):
        return jnp.asarray(self.Ut[s:t]) - jnp.asarray(self.uref[s:t])[:, None]


def qman_fit(cg, r, gammas, seed, holdout, states_per_trajectory):
    """qman.fit(Ut, r, ...) streamed. Returns dict(bank=host [u_ref | V_r | W] (n, 1+r+P), coefficients (Ns, r),
    info) with info carrying the same keys qman.fit records."""
    t0 = time.perf_counter()
    Ut, bl = cg.Ut, cg.bl
    r = int(r)
    n, Ns = Ut.shape
    per = int(states_per_trajectory)
    assert per > 0 and Ns % per == 0, (Ns, per)
    P = int(r * (r + 1) // 2)
    wk = jnp.clip(cg.w[:r], 1e-300, None)
    Vs = cg.V[:, :r] / jnp.sqrt(wk)[None, :]           # V_r = Sc @ Vs, exactly as pod_basis(Sc, r)
    coefficients = jnp.zeros((r, Ns))
    for s, t in bl:
        S = cg.centred(s, t)
        coefficients = coefficients + (S @ Vs).T @ S
    iu, ju = np.triu_indices(r)
    Pi = coefficients[jnp.asarray(iu)] * coefficients[jnp.asarray(ju)]
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
    eye = jnp.eye(P)
    lhs = [Mtr + float(g) * scale * eye for g in gammas]
    err2 = np.zeros(len(gammas))
    norm_te2 = 0.
    for s, t in bl:                                        # pass: held-out ridge scores for every gamma at once
        S = cg.centred(s, t)
        E = S - (S @ Vs) @ coefficients
        Ctr = E[:, tr] @ Ptr.T
        Ete = E[:, te]
        norm_te2 += float(jnp.sum(S[:, te] ** 2))
        for i, Mg in enumerate(lhs):
            Wg = jnp.linalg.solve(Mg, Ctr.T).T
            err2[i] += float(jnp.sum((Wg @ Pte - Ete) ** 2))
    norm_te = float(np.sqrt(norm_te2))
    trace, best = [], None
    for g, e2 in zip(gammas, err2):
        err = float(np.sqrt(e2)) / max(norm_te, 1e-300)
        finite = bool(np.isfinite(err))
        trace.append(dict(gamma=float(g), heldout_relative=(err if finite else None), finite=finite))
        if finite and (best is None or err < best[1]):
            best = (float(g), err)
    assert best is not None, f'every ridge in {list(gammas)} gave a non-finite held-out score'
    gamma, heldout = best
    Mfull = Pi @ Pi.T
    scale_full = float(jnp.trace(Mfull)) / P
    Mg = Mfull + gamma * scale_full * eye
    bank = np.empty((n, 1 + r + P))
    wn2 = e2 = s2 = q2 = 0.
    for s, t in bl:                                        # pass: the final W on all snapshots, written into the bank
        S = cg.centred(s, t)
        Vb = S @ Vs
        E = S - Vb @ coefficients
        W = jnp.linalg.solve(Mg, (E @ Pi.T).T).T
        assert bool(jnp.all(jnp.isfinite(W))), f'non-finite W at r={r}, gamma={gamma}'
        bank[s:t, 0] = cg.uref[s:t]
        bank[s:t, 1:1 + r] = np.asarray(Vb)
        bank[s:t, 1 + r:] = np.asarray(W)
        wn2 += float(jnp.sum(W ** 2))
        e2 += float(jnp.sum(E ** 2))
        s2 += float(jnp.sum(S ** 2))
        q2 += float(jnp.sum((W @ Pi - E) ** 2))
    nrm = float(np.sqrt(s2))
    ev = np.asarray(jnp.linalg.eigvalsh(Mfull))
    ev_min = float(ev[0])
    cond = float(ev[-1] / ev_min) if ev_min > 0 else float('inf')
    rank_deficient = not (ev_min > 0 and np.isfinite(cond))
    eig = np.asarray(cg.w[:r])
    info = dict(rank=r, quadratic_terms=P, bank_columns=1 + r + P, snapshots=int(Ns),
                ridge=gamma, ridge_trace=trace, ridge_grid=[float(g) for g in gammas],
                heldout_relative=heldout, heldout_fraction=float(holdout),
                heldout_trajectories=int(nte_traj), trajectories=int(ntraj),
                heldout_snapshots=int(len(te)), states_per_trajectory=per,
                split='by trajectory (DESIGN A1)',
                selection_seed=int(seed), gram_scale=scale, gram_scale_full=scale_full,
                weight_frobenius_norm=float(np.sqrt(wn2)),
                gram_min_eigenvalue_negative=bool(ev_min < 0), gram_rank_deficient=bool(rank_deficient),
                gram_condition=(None if rank_deficient else cond), gram_min_eigenvalue=ev_min,
                snapshot_relative_linear_only=float(np.sqrt(e2)) / max(nrm, 1e-300),
                snapshot_relative_with_quadratic=float(np.sqrt(q2)) / max(nrm, 1e-300),
                pod_energy_total=float(cg.energy), pod_eigenvalues=eig.tolist(),
                pod_tail_fraction=float(max(float(cg.energy) - float(np.sum(eig)), 0.) / max(float(cg.energy), 1e-300)),
                centred=True, basis='POD of the centred snapshots (u_ref = snapshot mean)',
                streamed=True, row_blocks=len(bl), seconds=time.perf_counter() - t0)
    return dict(bank=bank, coefficients=np.asarray(coefficients.T), info=info)


class LeanGridBank(A.GridBank):
    """GridBank without the resident padded copy: `at` pads column chunks on demand (bitwise the same values)."""

    def __init__(self, V, L, chunk=64):
        self.L = int(L)
        self.V = jnp.asarray(V)
        self.dim = int(self.V.shape[1])
        self.chunk = int(chunk)

    def at(self, xy, chunk=8192):
        xy = jnp.asarray(np.asarray(xy))
        L = self.L
        out = []
        for c0 in range(0, self.dim, self.chunk):
            pad = jnp.pad(self.V[:, c0:c0 + self.chunk].T.reshape(-1, L - 1, L - 1), ((0, 0), (1, 1), (1, 1)))
            out.append(jax.vmap(lambda f: e.sample_field(f, xy, L), out_axes=1)(pad))
            del pad
        return jnp.concatenate(out, axis=1)

    def stencil(self, ij, L):
        raise NotImplementedError('LeanGridBank serves dense-residual arms only')


def check_parity(L=48, ntraj=12, stride=2):
    """Local check (DESIGN section 6): streamed == in-memory to 1e-9 on a small mesh, with blocks forced small."""
    import qman as QM
    from ablation import pod_basis
    phys = e.params_draw(0, ntraj)
    Ut, info = snapshots(L, .005, phys, stride, 1e-9, 1e-7)
    tb = 8 * Ut.shape[1] * 97                             # ~97-row blocks: many blocks, a ragged last one
    out = dict(mesh=L, trajectories=ntraj, snapshots=int(Ut.shape[1]), block_rows=97)
    modes, eig, energy, coords = pod(Ut, 20, tb)
    m2, e2, en2 = pod_basis(jnp.asarray(Ut), 20)
    c2 = np.asarray(jnp.asarray(Ut).T @ m2)
    rel = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))
    out['pod'] = dict(modes=rel(modes, m2), eig=rel(eig, e2), energy=abs(energy - en2) / en2, coords=rel(coords, c2))
    cg = CentredGram(Ut, tb)
    gam = [0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.]
    out['qman'] = []
    for r in (3, 6):
        s = qman_fit(cg, r, gam, 20260922, .2, info['states_per_trajectory'])
        ref = QM.fit(jnp.asarray(Ut), r, gam, 20260922, .2, info['states_per_trajectory'])
        Bref = np.asarray(QM.bank_columns(ref, True))
        z = np.random.default_rng(1).normal(size=r) * np.std(ref['coefficients'], axis=0)
        h = np.asarray(QM.head(r, True)(jnp.asarray(z)))
        tr_s = [t_['heldout_relative'] for t_ in s['info']['ridge_trace']]
        tr_r = [t_['heldout_relative'] for t_ in ref['info']['ridge_trace']]
        out['qman'].append(dict(r=r, ridge=s['info']['ridge'], ridge_ref=ref['info']['ridge'],
                                same_ridge=s['info']['ridge'] == ref['info']['ridge'],
                                bank=rel(s['bank'], Bref), field_at_random_code=rel(s['bank'] @ h, Bref @ h),
                                coefficients=rel(s['coefficients'], ref['coefficients']),
                                ridge_trace=rel(tr_s, tr_r),
                                linear_only=abs(s['info']['snapshot_relative_linear_only'] - ref['info']['snapshot_relative_linear_only']),
                                with_quadratic=abs(s['info']['snapshot_relative_with_quadratic'] - ref['info']['snapshot_relative_with_quadratic']),
                                split=s['info']['split']))
    V = jnp.asarray(np.random.default_rng(2).normal(size=((L - 1) ** 2, 10)))
    xy = jnp.asarray(np.random.default_rng(3).uniform(size=(50, 2)))
    lean, full = LeanGridBank(V, L, chunk=3), A.GridBank(V, L)
    out['lean_bank_at_bitwise'] = bool(np.array_equal(np.asarray(lean.at(xy)), np.asarray(full.at(xy))))
    worst = max([out['pod'][k] for k in ('modes', 'eig', 'coords')] +
                [q[k] for q in out['qman'] for k in ('bank', 'field_at_random_code', 'coefficients', 'ridge_trace')])
    out['worst_relative'] = worst
    out['passed'] = bool(worst <= 1e-9 and all(q['same_ridge'] for q in out['qman']) and out['lean_bank_at_bitwise'])
    return out


if __name__ == '__main__':
    import json
    import sys
    res = check_parity()
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 1:
        open(sys.argv[1], 'w').write(json.dumps(res, indent=1) + '\n')
    assert res['passed'], 'streamed fit does not reproduce the in-memory fit'
