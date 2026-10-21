"""spectral-pilot Part B: references with the four-grid order gate and time gate (DESIGN A2.5, A3.4, A4), and the
L=512 training snapshots. PILOT: small, not a result.

  python gen_b.py [--procs 6] [--skip-train]

Writes runs/B/ref.json, ref_lat256.npy (12, 6, P256) = the L=1024 reference on the i/256 lattice (contains i/64, i/128),
train_flat.npy (96, 21, n_in(512)) (large, untracked; SHA256 in ref.json).
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np

import common as C
import fd_l as F

OUT = C.RUNS / 'B'
_A = {}


def _work(args):
    d, L, n, lat, n_out, flat = args
    if L not in _A:
        _A[L] = F.lap_matrix(L)
    A, m = _A[L]
    O = F.imex_query(np.asarray(d), L, n, A, m, out_lattice=False, n_out=n_out, flat=flat)
    return O if flat else F.lat_vals(O, L, lat)


def run(D, L, n, procs, lat=64, n_out=5, flat=False):
    with Pool(procs) as p:
        return np.stack(p.map(_work, [(d, L, n, lat, n_out, flat) for d in D]))


def sub(lat_from, lat_to, Lref=1024):
    """Map lattice values at i/lat_from to the coarser lattice i/lat_to (both restricted to Omega_L)."""
    I1, J1 = F.lattice_idx(Lref, lat_from)
    I2, J2 = F.lattice_idx(Lref, lat_to)
    pos = {(i, j): k for k, (i, j) in enumerate(zip(I1, J1))}
    return np.array([pos[(i, j)] for i, j in zip(I2, J2)])


def maxdiff(Fv, Rv):
    return float(np.max(F.err_lat(Fv, Rv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--skip-train', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = F.cohorts_b()
    n = 2000
    rep = dict(env=C.env_record(), n_ref=n, L_ref=1024, test_desc=te.tolist(), s_test=C.steep(te).tolist(),
               train_sha=C.sha(tr), label=C.PILOT, seconds={})
    t0 = time.perf_counter()
    R256 = run(te, 1024, n, a.procs, lat=256)
    rep['seconds']['ref1024'] = time.perf_counter() - t0
    np.save(OUT / 'ref_lat256.npy', R256)
    rep['ref_sha'] = C.sha(R256)
    i64, i128 = sub(256, 64), sub(256, 128)
    lv = {1024: R256}
    for L in (128, 256, 512):
        t1 = time.perf_counter()
        lv[L] = run(te, L, n, a.procs, lat=128)               # i/128 lattice (contains i/64)
        rep['seconds'][f'L{L}'] = time.perf_counter() - t1
    j64 = sub(128, 64)
    on64 = {L: (v[..., i64] if L == 1024 else v[..., j64]) for L, v in lv.items()}
    on128 = {L: (v[..., i128] if L == 1024 else v) for L, v in lv.items()}
    order = {}
    for tag, S in (('lat64', on64), ('lat128', on128)):
        d1 = maxdiff(S[128], S[256])
        d2 = maxdiff(S[256], S[512])
        d3 = maxdiff(S[512], S[1024])
        p1, p2 = np.log2(d1 / d2), np.log2(d2 / d3)
        order[tag] = dict(d128_256=d1, d256_512=d2, d512_1024=d3, p1=float(p1), p2=float(p2))
    o = order['lat64']
    gate = bool(o['p2'] >= 1 and abs(o['p1'] - o['p2']) <= 0.5 and o['d128_256'] > o['d256_512'] > o['d512_1024'] > 0)
    e_ref = o['d512_1024'] / (2 ** o['p2'] - 1) if gate else None
    rep['order'] = order
    rep['order_gate'] = gate
    rep['e_ref_hat'] = e_ref
    # time gate: n vs 2n at L=1024 on the 4 largest-s cases; at L=512 on all 12
    t1 = time.perf_counter()
    hard = np.argsort(-C.steep(te))[:4]
    R2 = run(te[hard], 1024, 2 * n, a.procs, lat=64)
    dt1024 = maxdiff(R2, on64[1024][hard])
    R512b = run(te, 512, 2 * n, a.procs, lat=64)
    dt512 = maxdiff(R512b, on64[512])
    rep['seconds']['time_gate'] = time.perf_counter() - t1
    rep['time_gate'] = dict(hard_cases=hard.tolist(), d1024_4cases=dt1024, d512_12cases=dt512)
    delta_t = max(dt1024, dt512)
    rep['delta_t'] = delta_t
    rep['U_B'] = (e_ref + delta_t) if gate else None
    rep['U_numerical'] = delta_t
    rep['scoring'] = 'physical (Richardson allowance)' if gate else 'numerical reference only (order gate failed)'
    rep['fd512_err_vs_ref'] = dict(worst=float(np.max(F.err_lat(on64[512], on64[1024]))),
                                   median=float(np.median(F.err_lat(on64[512], on64[1024]))))
    # diagnostic (MF1 -> physical diagnostic): the L=128 run as a "reference" for the 256->512 gate
    rep['diag_coarse_L128_vs_ref'] = float(np.max(F.err_lat(on64[128], on64[1024])))
    print('order', order, 'gate', gate, 'e_ref', e_ref, 'delta_t', delta_t, flush=True)
    C.dump(OUT / 'ref.json', rep)
    if not a.skip_train:
        t1 = time.perf_counter()
        X = run(tr, 512, n, a.procs, n_out=20, flat=True)
        assert np.isfinite(X).all()
        np.save(OUT / 'train_flat.npy', X)
        rep['train_sha_flat'] = C.sha(X)
        rep['train_shape'] = list(X.shape)
        rep['seconds']['train'] = time.perf_counter() - t1
        # diagnostic: 4 training trajectories at L=512 vs L=1024 (lattice i/64)
        T1024 = run(tr[:4], 1024, n, a.procs, lat=64)
        T512 = run(tr[:4], 512, n, a.procs, lat=64)
        rep['train_diag'] = float(np.max(F.err_lat(T512, T1024)))
    rep['seconds']['total'] = time.perf_counter() - t0
    C.dump(OUT / 'ref.json', rep)
    print('done', rep['seconds'], flush=True)


if __name__ == '__main__':
    main()
