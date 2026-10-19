"""Independent CPU establishment of the bad-generating-vector fixture (DESIGN A2.2, A4.1). NumPy only (no JAX, no job
code): 2D bank values/gradients from the vendored NumPy bank (`npcheck` -> `npbank`), 3D from the vendored NumPy core
(`vendor/quad3d/npcore.py`); Gauss targets from numpy.polynomial.legendre; the bad rank-1 lattice z = (1, ..., 1) with
shift numpy default_rng(0) written out here. At every frozen state c of S_fix:

    rho_tgt = ||N_bad(c) - N_tgt(c)|| / ||N_tgt(c)||

The fixture is ADMITTED iff every input, N_bad, N_tgt is finite, ||N_tgt|| > 0 and the finite ratio >= 1 at every state.
Writes checks/fixture-badlat.json (one entry per panel). Never re-run with other fixtures after seeing the result.

    JAX_PLATFORMS=cpu python establish_fixture.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import npcheck as NC  # noqa: E402
import npcore as N3  # noqa: E402

QROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs')
FIX2D = (QROOT / 'smkA/archive/output/qacc_smkA/R_R384_LSPG_CNR_1.npz',
         '3b00ba0f24261e4792fa05468313f7a6e1ab349ce883b53ce18fdd65a3c93dee')
FIX3D = (QROOT / 'q3d/archive/output/q3run_q3d/W_q2.npz',
         '0ca5015571184c2feba922cf014d18bdcbc7feba99fc009e60fa863989583f5b')
KEY3D = 'W512|gl24|fs|CN|0.01|K1#q|fs|CN|0.01|K1'
W1024 = HERE / 'inputs' / 'model_W1024'


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def badlat(n, d):
    i = np.arange(n)
    X = (np.outer(i, np.ones(d)) / n + np.random.default_rng(0).uniform(size=d)) % 1.0
    return X, np.full(n, 1.0 / n)


def gauss(p, d):
    t, w = np.polynomial.legendre.leggauss(p)
    x, w = (t + 1) / 2, w / 2
    X = np.stack(np.meshgrid(*([x] * d), indexing='ij'), -1).reshape(-1, d)
    W = w
    for _ in range(d - 1):
        W = np.multiply.outer(W, w)
    return X, W.ravel()


def admit(C, Nb, Nt):
    out = []
    for c, a, b in zip(C, Nb, Nt):
        fin = bool(np.isfinite(c).all() and np.isfinite(a).all() and np.isfinite(b).all())
        nb = float(np.linalg.norm(b)) if fin else float('nan')
        r = float(np.linalg.norm(a - b) / nb) if (fin and nb > 0) else float('nan')
        out.append(dict(finite=fin, target_norm=nb, rho_tgt=r, ok=bool(fin and nb > 0 and np.isfinite(r) and r >= 1.)))
    return dict(states=out, admitted=all(o['ok'] for o in out))


def tested2d(X, w, r, kx, ky, C):
    t = NC.tables(X, w, r, kx, ky, 1024)
    return np.stack([NC.N(c, t) for c in C])


def panel2d(r, M, nbad=121393, ptgt=384):
    z = np.load(FIX2D[0])
    assert str(z['case_ids'][0]) == 'dev6|0' and int(z['Rp']) == 384 and str(z['rule']) == 'gauss128'
    C = np.asarray(z['W'][0])[:, :r]                    # six output states of dev6|0, prefix r
    kx, ky = np.asarray(z['kx'])[:M], np.asarray(z['ky'])[:M]
    Xb, wb = badlat(nbad, 2)
    Xt, wt = gauss(ptgt, 2)
    t0 = time.time()
    res = admit(C, tested2d(Xb, wb, r, kx, ky, C), tested2d(Xt, wt, r, kx, ky, C))
    res.update(panel=f'2d|R{r}|M{M}', bad=f'badlat{nbad}', target=f'gauss{ptgt}', fixture=str(FIX2D[0]),
               fixture_sha256=FIX2D[1], case='dev6|0', seconds=time.time() - t0,
               note='R128 panel: first 128 coefficients of R384 states (kernel fixture, not reached states of R128)' if r != 384 else '')
    return res


def tested3d(p, T, X, w, k, C, chunk=1 << 15):
    acc = np.zeros((len(C), len(k)))
    for s in range(0, len(X), chunk):
        G, D = N3.bank_np(p, T, X[s:s + chunk], deriv=True)
        acc += N3.offmesh_adv_np(G, D, X[s:s + chunk], w[s:s + chunk], 65, k, C)
    return acc


def panel3d(r=512, Mreq=2048, nbad=262144, ptgt=64):
    z = np.load(FIX3D[0])
    cases = list(np.asarray(z['cases|' + KEY3D]))
    C = np.asarray(z[KEY3D][cases.index(0)])            # six output states of case 0
    assert C.shape == (6, r)
    p, T = N3.load_bank(W1024)
    T = T[:, :r]
    k, _ = N3.modes(65, Mreq)
    assert len(k) == 2052
    Xb, wb = badlat(nbad, 3)
    Xt, wt = gauss(ptgt, 3)
    t0 = time.time()
    res = admit(C, tested3d(p, T, Xb, wb, k, C), tested3d(p, T, Xt, wt, k, C))
    res.update(panel=f'3d|W{r}|M{len(k)}', bad=f'badlat{nbad}', target=f'gl{ptgt}', fixture=str(FIX3D[0]),
               fixture_sha256=FIX3D[1], key=KEY3D, case=0, seconds=time.time() - t0)
    return res


def main():
    for p, want in (FIX2D, FIX3D):
        assert sha_file(p) == want, p
    out = dict(rule='admitted iff finite inputs and evaluations, ||N_tgt|| > 0 and finite rho_tgt >= 1 at every state '
                    '(DESIGN A2.2, A4.1)', panels=[])
    for f in (lambda: panel2d(384, 1536), lambda: panel2d(128, 512), lambda: panel3d()):
        r = f()
        print(r['panel'], 'admitted', r['admitted'], [round(s['rho_tgt'], 3) for s in r['states']], flush=True)
        out['panels'].append(r)
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'fixture-badlat.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
