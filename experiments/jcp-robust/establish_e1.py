"""jcp-robust: independent CPU establishment of the E1 bad-generating-vector fixtures (DESIGN 2.4 C-E1.1, A1.3, A4.2).
Copied from jcp-dims establish_fixture.py (helpers badlat, gauss, admit, tested2d, tested3d unchanged); the panels are this
lane's own fixtures, fixed before any GPU use and never re-chosen:
  2D: R128 (M = 512, frozen 1024 list), six saved output states of dev6|0 of P1 acc1024 R_R128_BE.npz (LSPG-BE dt0, fib4181)
      -> inputs/fixture_e1_2d.npz; bad = badlat4181 (z = (1, 1)), target gauss384
  3D: W256 (M = complete_M(65, 1024)), six saved output states of case 0 of qualify q3d3 W256|lat8192|qn|BDF2|0.025|k4
      -> inputs/fixture_e1_3d.npz; bad = badlat8192 (z = (1, 1, 1)), target gl64
Admitted iff finite inputs and evaluations, ||N_tgt|| > 0 and finite rho_tgt >= 1 at every state. Writes
checks/establish-e1.json.

    JAX_PLATFORMS=cpu python establish_e1.py
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
    n_ok = len(C) == len(Nb) == len(Nt) == 6                  # registered six states; never an empty reduction (code audit 2 #17)
    return dict(states=out, admitted=bool(n_ok and all(o['ok'] for o in out)))


def tested2d(X, w, r, kx, ky, C):
    t = NC.tables(X, w, r, kx, ky, 1024)
    return np.stack([NC.N(c, t) for c in C])


def tested3d(p, T, X, w, k, C, chunk=1 << 15):
    acc = np.zeros((len(C), len(k)))
    for s in range(0, len(X), chunk):
        G, D = N3.bank_np(p, T, X[s:s + chunk], deriv=True)
        acc += N3.offmesh_adv_np(G, D, X[s:s + chunk], w[s:s + chunk], 65, k, C)
    return acc



FIX2D = (HERE / 'inputs' / 'fixture_e1_2d.npz', '777df398a8ca7dbf0f2048ea906639805b999efe2c43c5653fd2bf9c6306b321')
FIX3D = (HERE / 'inputs' / 'fixture_e1_3d.npz', 'b791e71406ae3c30eddd77828b4bd15af44ce2164b2d4528232622697ab1bd02')


def panel2d(r=128, M=512, nbad=4181, ptgt=384):
    z = np.load(FIX2D[0])
    src = np.load(str(z['source']))
    assert sha_file(str(z['source'])) == str(z['source_sha256'])
    C = np.asarray(z['W'])[:, :r]
    assert C.shape == (6, r) and int(z['Rp']) == r
    kx, ky = np.asarray(src['kx'])[:M], np.asarray(src['ky'])[:M]
    Xb, wb = badlat(nbad, 2)
    Xt, wt = gauss(ptgt, 2)
    t0 = time.time()
    res = admit(C, tested2d(Xb, wb, r, kx, ky, C), tested2d(Xt, wt, r, kx, ky, C))
    res.update(panel=f'2d|R{r}|M{M}', bad=f'badlat{nbad}', target=f'gauss{ptgt}', fixture=str(FIX2D[0].relative_to(HERE)),
               fixture_sha256=FIX2D[1], case='dev6|0', seconds=time.time() - t0, mode_list_sha256=hashlib.sha256(
                   np.ascontiguousarray(np.stack([kx, ky]).astype(np.int64)).tobytes()).hexdigest())
    return res


def panel3d(r=256, Mreq=1024, nbad=8192, ptgt=64):
    z = np.load(FIX3D[0])
    C = np.asarray(z['W'])
    assert C.shape == (6, r)
    p, T = N3.load_bank(W1024)
    T = T[:, :r]
    k, _ = N3.modes(65, Mreq)
    Xb, wb = badlat(nbad, 3)
    Xt, wt = gauss(ptgt, 3)
    t0 = time.time()
    res = admit(C, tested3d(p, T, Xb, wb, k, C), tested3d(p, T, Xt, wt, k, C))
    res.update(panel=f'3d|W{r}|M{len(k)}', M=len(k), bad=f'badlat{nbad}', target=f'gl{ptgt}',
               fixture=str(FIX3D[0].relative_to(HERE)), fixture_sha256=FIX3D[1], case=0, seconds=time.time() - t0)
    return res


def main():
    for p, want in (FIX2D, FIX3D):
        assert sha_file(p) == want, p
    out = dict(rule='admitted iff finite inputs and evaluations, ||N_tgt|| > 0 and finite rho_tgt >= 1 at every state '
                    '(DESIGN 2.4, A1.3, A4.2)', panels=[])
    for f in (panel2d, panel3d):
        r = f()
        print(r['panel'], 'admitted', r['admitted'], [round(s['rho_tgt'], 3) for s in r['states']], flush=True)
        out['panels'].append(r)
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'establish-e1.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
