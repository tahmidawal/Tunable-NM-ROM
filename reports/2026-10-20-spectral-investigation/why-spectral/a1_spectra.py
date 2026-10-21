"""A1. Sine-coefficient decay of the second-order reference solutions (CPU, NumPy/SciPy only).

For every reference field u(t_k) on the scoring lattice (2D: 255^2 interior nodes of x = i/256; 3D: 63^3 interior nodes
of x = k/64) take the orthonormal DST-I, which is the discrete sine series of the field on that lattice (Parseval).
Per case and output time:
  * tail(N) = ||u - P_N u|| / ||u(t0)||, P_N keeping modes 1..N-1 per axis (the spectral FOM's convention for N),
    i.e. the best possible error of an N-mode sine solver on that field (a projection floor, not a solver error);
  * N_proj(eps) = smallest N with max_{k>=1} tail_k(N) <= eps (project error convention: max over t_1..t_5);
  * sigma = exponential decay rate of tail(N) ~ C exp(-sigma N), least squares where 1e-6 <= tail <= 1e-2
    (upper end: past the initial transient; lower end: above the references' own error, eps_ref ~ 1e-6);
  * delta = max|u| / max|grad u| (central differences on the lattice): a measured front-width proxy.
Lattice limits: N <= 256 in 2D and N <= 64 in 3D; an N_proj beyond the lattice is reported as censored (None).
Cohorts: E0 2D dev6 + val32, E0 3D dev4 + val64, jcp-regimes N0, Am (A-), C1, C2 (16 each).
Writes analysis/a1_spectra.json.
"""
import json
import numpy as np
from scipy.fft import dstn

import common as C

EPS = (1e-2, 3e-3, 1e-3, 1e-4)


def shells(c, dim):
    """shell energies E[k], k = max_a k_a (1-based index k -> E[k-1])."""
    n = c.shape[0]
    k = np.arange(1, n + 1)
    if dim == 2:
        K = np.maximum(k[:, None], k[None, :])
    else:
        K = np.maximum(np.maximum(k[:, None, None], k[None, :, None]), k[None, None, :])
    return np.bincount(K.ravel() - 1, weights=(c ** 2).ravel(), minlength=n)


def analyse(fields, n0, dim, h):
    nt = fields.shape[0]
    if dim == 2:
        U = fields[:, 1:-1, 1:-1]
    else:
        m = round(fields.shape[1] ** (1 / 3))
        U = fields.reshape(nt, m, m, m)
    n = U.shape[1]
    tails, sig, dl, amp = [], [], [], []
    for t in range(nt):
        c = dstn(U[t], type=1, norm='ortho')
        E = shells(c, dim)
        rev = np.cumsum(E[::-1])[::-1]               # rev[j] = sum_{k >= j+1} E  -> tail of N = j+1
        tail = np.sqrt(np.maximum(np.r_[rev, 0.0], 0.0)) / n0   # index j <-> N = j+1 (N = 1 keeps nothing)
        tails.append(tail)
        Ns = np.arange(1, n + 2)
        msk = (tail >= 1e-6) & (tail <= 1e-2)
        if msk.sum() >= 4:
            p = np.polyfit(Ns[msk], np.log(tail[msk]), 1)
            sig.append(float(-p[0]))
        else:
            sig.append(None)
        g = np.gradient(U[t], h)
        gm = np.sqrt(sum(x ** 2 for x in g)).max()
        dl.append(float(np.abs(U[t]).max() / gm) if gm > 0 else None)
        amp.append(float(np.abs(U[t]).max()))
    tails = np.array(tails)
    worst = tails[1:].max(axis=0)
    Nproj = {}
    for e in EPS:
        idx = np.nonzero(worst <= e)[0]
        Nproj[str(e)] = int(idx[0] + 1) if idx.size else None
    return dict(sigma=sig, delta=dl, umax=amp, N_proj=Nproj, tail_worst=worst.tolist(),
                tail_t=[x.tolist() for x in tails], n_lattice_modes=n)


def main():
    out = dict(eps=EPS, cohorts={})
    for dim in (2, 3):
        rows = []
        for coh, case in C.e0_cases(dim):
            f, meta = C.load_e0(dim, coh, case)
            n0 = meta['judge']['n0']
            h = 1 / 256 if dim == 2 else 1 / 64
            r = analyse(f, n0, dim, h)
            d = meta['desc']
            r.update(case=f'{coh}|{case}', s=C.steepness(meta, dim), nu=float(d['nu']),
                     a=float(d['a'] if dim == 2 else d['A']), eps_ref=meta['judge']['eps_ref'],
                     w=(float(d['w']) if dim == 2 else [float(x) for x in d['w']]))
            rows.append(r)
            print(dim, r['case'], round(r['s'], 1), r['N_proj'], [None if x is None else round(x, 3) for x in r['sigma']])
        out['cohorts'][f'E0_{dim}d'] = rows
    for g in C.REGIMES:
        rows = []
        for case in C.regime_cases(g):
            f, meta = C.load_regime(g, case)
            n0 = meta['judge']['n0']
            r = analyse(f, n0, 2, 1 / 256)
            d = meta['desc']
            r.update(case=f'{g}|{case}', s=float(d['a']) / float(d['nu']), nu=float(d['nu']), a=float(d['a']),
                     w=float(d['w']), eps_ref=meta['judge']['eps_ref'], outcome=meta['outcome'])
            rows.append(r)
            print(g, r['case'], round(r['s'], 1), r['N_proj'])
        out['cohorts'][f'reg_{g}'] = rows
    (C.OUT / 'a1_spectra.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
