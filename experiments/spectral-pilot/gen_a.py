"""spectral-pilot Part A: references (gate per DESIGN A1.7, SW2) and training snapshots for one level.
PILOT: small, not a result.

  python gen_a.py A2 [--threads 8] [--skip-train]

Writes runs/A/<level>/ref.json, ref_coef.npy (12, 6, Nref-1, Nref-1), train_coef.npy (n_train, 21, Nref-1, Nref-1)
(large; untracked, SHA256 in ref.json).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import jax.numpy as jnp

import common as C


def run_batched(fn, D, threads, chunk=4):
    parts = C.chunks(np.arange(len(D)), chunk)
    with ThreadPoolExecutor(threads) as ex:
        outs = list(ex.map(lambda ix: np.asarray(fn(jnp.asarray(D[ix]))), parts))
    return np.concatenate(outs)


def gate_diff(Bc, Bref):
    """Per-case continuous-norm difference of a check run (any N) from the reference (DESIGN A1.7)."""
    return np.asarray(C.err_coef(C.pad_to(jnp.asarray(Bc), Bref.shape[-1] + 1), jnp.asarray(Bref)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('level')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--nref', type=int, default=None, help='override N_ref (dated amendment only)')
    a = ap.parse_args()
    lv = a.level
    nu_lo, Nref, ntr = C.LEVELS[lv]
    Nref = a.nref or Nref
    out = C.RUNS / 'A' / lv
    out.mkdir(parents=True, exist_ok=True)
    tr, te = C.level_cohorts(lv)
    n = C.ref_cfl_steps(Nref, factor=2.0)
    rep = dict(level=lv, env=C.env_record(), Nref=Nref, n_ref=n, nu=(float(min(np.r_[tr[:, 4], te[:, 4]])), float(max(np.r_[tr[:, 4], te[:, 4]]))), n_train=len(tr),
               n_test=len(te), test_desc=te.tolist(), train_sha=C.sha(tr), s_test=C.steep(te).tolist(),
               s_train=C.steep(tr).tolist(), label=C.PILOT)
    t0 = time.perf_counter()
    coef = run_batched(C.make_fom(Nref, n, 5, lattice=False), te, a.threads)
    rep['t_ref'] = time.perf_counter() - t0
    checks = {}
    for tag, (N2, n2) in dict(space=(3 * Nref // 4, n), time=(Nref, 2 * n), diag_coarse=(Nref // 4, n)).items():
        t1 = time.perf_counter()
        e = gate_diff(run_batched(C.make_fom(N2, n2, 5, lattice=False), te, a.threads), coef)
        checks[tag] = dict(N=N2, n=n2, err=e.tolist(), max=float(np.max(e)), seconds=time.perf_counter() - t1)
        print(lv, tag, N2, n2, 'max diff %.3e' % float(np.max(e)), flush=True)
    rep['checks'] = checks
    rep['accept_space'] = bool(checks['space']['max'] <= 2e-5)
    rep['accept_time'] = bool(checks['time']['max'] <= 2e-6)
    rep['accepted'] = rep['accept_space'] and rep['accept_time']
    rep['tau_min_eligible'] = 20 * max(checks['space']['max'], checks['time']['max'])
    rep['diag_coarse_rejected'] = bool(checks['diag_coarse']['max'] > 2e-5)
    # SW2: the gate must reject a real reference perturbed by 5x its space threshold
    pert = coef.copy()
    rng = np.random.default_rng(7)
    for j in range(len(pert)):
        for k in range(1, 6):
            d = rng.standard_normal(pert.shape[-2:])
            pert[j, k] += 5 * 2e-5 * np.linalg.norm(coef[j, 0]) * d / np.linalg.norm(d)
    rep['SW2_fired'] = bool(np.max(gate_diff(pert, coef)) > 2e-5)
    np.save(out / 'ref_coef.npy', coef)
    rep['ref_coef_sha'] = C.sha(coef)
    print(lv, 'accepted', rep['accepted'], 'SW2', rep['SW2_fired'], 'tau_min', rep['tau_min_eligible'], flush=True)

    if not a.skip_train:
        t1 = time.perf_counter()
        X = run_batched(C.make_fom(Nref, n, 20, lattice=False), tr, a.threads)
        assert np.isfinite(X).all()
        np.save(out / 'train_coef.npy', X)
        rep['train_coef_sha'] = C.sha(X)
        rep['train_shape'] = list(X.shape)
        rep['t_train'] = time.perf_counter() - t1
        # training qualification: 4 trajectories at 3N/4 (the 5 output-time subset of the 21 snapshots)
        sub = X[:4, ::4]
        e = gate_diff(run_batched(C.make_fom(3 * Nref // 4, n, 5, lattice=False), tr[:4], a.threads), sub)
        rep['train_check'] = dict(err=e.tolist(), max=float(np.max(e)), ok=bool(np.max(e) <= 1e-4))
        print(lv, 'train check max %.3e' % float(np.max(e)), flush=True)
    rep['seconds'] = time.perf_counter() - t0
    C.dump(out / 'ref.json', rep)
    print(lv, 'done %.1f s' % rep['seconds'], flush=True)


if __name__ == '__main__':
    main()
