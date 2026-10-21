"""A3. Learned bank against plain sines at equal size, per case, against steepness (CPU, NumPy only).

Extends the improvement-ideation `analysis/bank_vs_sine.py` from the first-order `refdv` fields to the second-order
references (E0 2D dev6 + val32, and jcp-regimes N0, Am, C1, C2), and from four widths to every prefix width.

Bases on the 257^2 scoring lattice (zero walls), orthogonal projection in the lattice inner product:
  * bank: the deployed 2D bank (dn256b checkpoint + rotation_R512, via jcp-regimes' vendored NumPy evaluator `npcheck`,
    read-only); one unpivoted QR of its 512 columns gives nested orthonormal prefixes, so the floor of every prefix r is
    one cumulative sum;
  * sines, Laplacian order (k1^2 + k2^2, the order the ROM's test modes and the ideation used) and square blocks
    (modes 1..N-1 per axis, the spectral FOM's set, (N-1)^2 unknowns).
floor(r) = max_{k=1..5} ||u(t_k) - P_r u(t_k)|| / ||u(t_0)||.
Per case: r_bank(eps), K_lap(eps), K_block(eps) = smallest size reaching eps, and the compression ratio
rho(eps) = K_lap / r_bank (how many more sine unknowns than bank unknowns the same floor needs).
Writes analysis/a3_bank.json.
"""
import json
import sys

import numpy as np
from scipy.fft import dstn

import common as C

sys.path.insert(0, str(C.REG))
import npcheck as NC  # noqa: E402

EPS = (3e-2, 1e-2, 3e-3, 1e-3)


def first(fl, e):
    idx = np.nonzero(fl <= e)[0]
    return int(idx[0] + 1) if idx.size else None


def main():
    NC.decode257(np.zeros((1, 512)), 512)
    G = NC._G257[512]
    Q, R = np.linalg.qr(G)
    dg = np.abs(np.diag(R))
    assert dg.min() > 1e-8 * dg.max(), 'bank prefix columns nearly dependent'
    n = 255
    k = np.arange(1, n + 1)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    lap = np.argsort((KX ** 2 + KY ** 2).ravel(), kind='stable')
    blk = np.maximum(KX, KY).ravel()
    out = dict(eps=EPS, bank=C.rel(NC.CKPT), qr_diag_min_over_max=float(dg.min() / dg.max()), rows=[])
    sets = [('E0', coh, c) for coh, c in C.e0_cases(2)] + [(g, g, c) for g in C.REGIMES for c in C.regime_cases(g)]
    for src, coh, case in sets:
        if src == 'E0':
            f, meta = C.load_e0(2, coh, case)
        else:
            f, meta = C.load_regime(coh, case)
        n0 = meta['judge']['n0']
        U = f.reshape(6, -1)
        tot = (U[1:] ** 2).sum(1)
        cb = np.cumsum((U[1:] @ Q) ** 2, axis=1)
        fl_bank = np.sqrt(np.maximum(tot[:, None] - cb, 0)).max(0) / n0          # index r-1
        Cs = np.stack([dstn(f[t, 1:-1, 1:-1], type=1, norm='ortho').ravel() for t in range(1, 6)])
        totS = (Cs ** 2).sum(1)
        cl = np.cumsum(Cs[:, lap] ** 2, axis=1)
        fl_lap = np.sqrt(np.maximum(totS[:, None] - cl, 0)).max(0) / n0           # index K-1
        Eb = np.stack([np.bincount(blk - 1, weights=Cs[t] ** 2, minlength=n) for t in range(5)])
        cbk = np.cumsum(Eb, axis=1)                                                # cbk[:, j]: modes with max k <= j+1
        fl_blk = np.sqrt(np.maximum(totS[:, None] - cbk, 0)).max(0) / n0          # index j <-> N = j+2
        d = meta['desc']
        row = dict(src=src, case=f'{coh}|{case}', s=float(d['a']) / float(d['nu']), nu=float(d['nu']), a=float(d['a']),
                   floor_bank={str(r): float(fl_bank[r - 1]) for r in (64, 128, 256, 384, 512)},
                   floor_lap={str(K): float(fl_lap[K - 1]) for K in (128, 384, 512, 1024, 1536, 2304, 4096)},
                   r_bank={}, K_lap={}, N_block={}, rho={})
        for e in EPS:
            rb, kl = first(fl_bank, e), first(fl_lap, e)
            j = np.nonzero(fl_blk <= e)[0]
            nb = int(j[0] + 2) if j.size else None
            row['r_bank'][str(e)], row['K_lap'][str(e)], row['N_block'][str(e)] = rb, kl, nb
            row['rho'][str(e)] = (kl / rb) if (rb and kl) else None
        # the ideation's statistic: smallest Laplacian prefix matching the bank floor at r (ratio K/r)
        row['K_match_over_r'] = {}
        for r in (128, 384, 512):
            km = first(fl_lap, fl_bank[r - 1])
            row['K_match_over_r'][str(r)] = km / r if km else None
        out['rows'].append(row)
        print(row['case'], round(row['s'], 1), {e: (row['r_bank'][str(e)], row['K_lap'][str(e)]) for e in EPS},
              row['K_match_over_r'])
    (C.OUT / 'a3_bank.json').write_text(json.dumps(out, indent=0))


if __name__ == '__main__':
    main()
