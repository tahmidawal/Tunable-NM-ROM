"""spectral-pilot Part A: references (with the acceptance gate and MF1) and training snapshots for one level.
PILOT: small, not a result.

  python gen_a.py A2 [--threads 8] [--skip-train]

Writes runs/A/<level>/ref.json, ref_lat.npy (12, 6, 257, 257), ref_coef.npy (12, 6, N-1, N-1),
train_coef.npy (n_train, 21, N-1, N-1) (large; untracked, SHA256 in ref.json).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('level')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--nref', type=int, default=None, help='override N_ref (amendment only)')
    a = ap.parse_args()
    lv = a.level
    nu_lo, nu_hi, Nref, ntr, *_ = C.LEVELS[lv]
    Nref = a.nref or Nref
    out = C.RUNS / 'A' / lv
    out.mkdir(parents=True, exist_ok=True)
    tr, te = C.level_cohorts(lv)
    n = C.ref_cfl_steps(Nref, factor=2.0)
    rep = dict(level=lv, env=C.env_record(), Nref=Nref, n_ref=n, nu=(nu_lo, nu_hi), n_train=len(tr), n_test=len(te),
               test_desc=te.tolist(), train_sha=C.sha(tr), s_test=C.steep(te).tolist(), s_train=C.steep(tr).tolist())
    t0 = time.perf_counter()

    # references on the test cases: N_ref at n steps (lattice + coefficients)
    fref = C.make_fom(Nref, n, 5, lattice=False)
    coef = run_batched(fref, te, a.threads)
    lat = np.stack([np.asarray(jnp.stack([C.synth_lattice(jnp.asarray(b), Nref) for b in cs])) for cs in coef])
    rep['t_ref'] = time.perf_counter() - t0
    # gate: space (3N/4, same n), time (N, 2n); MF1: N/4 must fail the space check
    checks = {}
    for tag, (N2, n2) in dict(space=(3 * Nref // 4, n), time=(Nref, 2 * n), mf1_coarse=(Nref // 4, n)).items():
        t1 = time.perf_counter()
        f2 = C.make_fom(N2, n2, 5, lattice=True)
        l2 = run_batched(f2, te, a.threads)
        e = np.asarray(C.err_runs(jnp.asarray(l2), jnp.asarray(lat)))
        checks[tag] = dict(N=N2, n=n2, err=e.tolist(), max=float(np.max(e)), seconds=time.perf_counter() - t1)
        print(lv, tag, N2, n2, 'max diff', float(np.max(e)), flush=True)
    rep['checks'] = checks
    rep['accept_space'] = bool(checks['space']['max'] <= 1e-4)
    rep['accept_time'] = bool(checks['time']['max'] <= 1e-5)
    rep['accepted'] = rep['accept_space'] and rep['accept_time']
    rep['MF1_fired'] = bool(checks['mf1_coarse']['max'] > 1e-4)
    np.save(out / 'ref_lat.npy', lat)
    np.save(out / 'ref_coef.npy', coef)
    rep['ref_lat_sha'] = C.sha(lat)
    print(lv, 'accepted', rep['accepted'], 'MF1 fired', rep['MF1_fired'], flush=True)

    if not a.skip_train:
        t1 = time.perf_counter()
        ftr = C.make_fom(Nref, n, 20, lattice=False)
        X = run_batched(ftr, tr, a.threads)
        assert np.isfinite(X).all()
        np.save(out / 'train_coef.npy', X)
        rep['train_coef_sha'] = C.sha(X)
        rep['train_shape'] = list(X.shape)
        rep['t_train'] = time.perf_counter() - t1
    rep['seconds'] = time.perf_counter() - t0
    C.dump(out / 'ref.json', rep)
    print(lv, 'done', rep['seconds'], 's', flush=True)


if __name__ == '__main__':
    main()
