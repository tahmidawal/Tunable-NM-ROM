"""A2. Per-case accuracy of the sine pseudo-spectral FOM against the second-order references, as a function of N
(CPU, JAX x64; JAX_PLATFORMS=cpu).

Solver: E0's `pscore.make_ps` (byte-identical in jcp-references, jcp-fom2d and jcp-speed3d): continuous sine
coefficients, exact linear part, 3/2-padded collocation of -u.grad u, ETDRK4 (Kassam-Trefethen contour, 32 points), IC
by composite Gauss-Legendre sine projection of the continuous u0, outputs synthesised on the scoring lattice. Scored with
`refstore.score` (2D/3D E0) or the same metric against the jcp-regimes references.

Sets (argv[1]):
  gate2d   ps|N48|0.0125 and ps|N24|0.025 on the 38 2D validation cases; must reproduce jcp-fom2d's in-job
           median/worst (match-final.json pareto.PS) -- a reproduction gate for this CPU path.
  e02d     2D, N in NS2, dt = 0.0025 (time error small: fom2d's N=128 rows move by < 1e-5 between dt 0.0025 and 0.001).
  e03d     3D, N in NS3, dt = 0.01 (speed3d f1 rows at N16 differ by < 1e-6 between 0.01 and 0.005); gate: N16 dt 0.025
           reproduces speed3d f1 per case.
  e02d_0125 / e03d_025   the same at the step the cost model charges (2D dt 0.0125 = 20 steps; 3D dt 0.025 = 10
           steps), so accuracy and cost refer to one configuration (non-finite runs recorded as None).
  reg      jcp-regimes N0, Am, C1, C2 (2D), N in NSR, dt per regime below the measured stability limit dt ~ 1.4/s
           (A2b): 0.0025 (N0, Am; s <= 200), 0.002 (C1; s <= 495), 0.001 (C2; s <= 953).
  stab     E0 2D (38) and the regimes (64) at the step the steepness model charges: per case
           dt = min(0.0125, 0.05 / ceil(0.05 s / c)), c = the A2b minimum of dt_max*s (aligned to the 0.05 output
           interval, so the run is executable as is); N in NSS. Accuracy and charged cost then refer to one setting.
Writes analysis/a2_<set>.json.
"""
import json
import sys
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

import common as C
sys.path.insert(0, str(C.REF))
import pscore as PS  # noqa: E402
import verdict as V  # noqa: E402

NS2 = (8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128)
NS3 = (6, 8, 10, 12, 14, 16, 20, 24, 32)
NSR = (16, 24, 32, 48, 64, 96, 128, 192, 256)
NSS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
DT_REG = dict(N0=0.0025, Am=0.0025, C1=0.002, C2=0.001)

_cache = {}


def solver(N, d):
    if (N, d) not in _cache:
        _cache[(N, d)] = (PS.make_ps(N, d), PS.lattice_S(N, 256 if d == 2 else 64))
    return _cache[(N, d)]


def ic2(desc, N):
    return PS.ic_coefs_2d((desc['cx'], desc['cy'], desc['w'], desc['a'], desc['nu']), N)


def ic3(desc, N):
    tab = {k: [desc[k]] for k in ('B', 'c', 'w', 'rho', 'A', 'nu', 's_star')}
    return PS.ic_coefs_3d(tab, 0, N)


def run(desc, N, dt, d):
    run_, S = solver(N, d)
    b0 = ic2(desc, N) if d == 2 else ic3(desc, N)
    t = time.perf_counter()
    r = run_(np.asarray(b0), float(desc['nu']), dt, S)
    return r['lat'], r['finite'], time.perf_counter() - t


def score_e0(lat, d, coh, case):
    return float(C.RS.score(lat, d, coh, case)['e'])


def score_reg(lat, ref, n0):
    e, _ = V.rel_err(np.asarray(lat), ref, n0)
    return float(e)


def main(which):
    out = dict(set=which, rows={})
    if which == 'gate2d':
        mf = C.J(C.FOM / 'checks/match-final.json')
        pts = {p['key']: p for p in mf['pareto']['PS']['points']}
        for N, dt in ((48, 0.0125), (24, 0.025)):
            es = []
            for coh, case in C.e0_cases(2):
                _, meta = C.load_e0(2, coh, case)
                lat, fin, _ = run(meta['desc'], N, dt, 2)
                es.append(score_e0(lat, 2, coh, case) if fin else float('inf'))
            k = f'ps|N{N}|{dt}'
            out['rows'][k] = dict(per_case=es, median=float(np.median(es)), worst=float(np.max(es)),
                                  fom2d_median=pts[k]['median'], fom2d_worst=pts[k]['worst'],
                                  rel_diff_median=abs(np.median(es) - pts[k]['median']) / pts[k]['median'],
                                  rel_diff_worst=abs(np.max(es) - pts[k]['worst']) / pts[k]['worst'])
            print(k, out['rows'][k]['rel_diff_median'], out['rows'][k]['rel_diff_worst'])
    elif which in ('e02d', 'e03d', 'e02d_0125', 'e03d_025'):
        d = 2 if which.startswith('e02d') else 3
        Ns, dt = (NS2, 0.0025) if d == 2 else (NS3, 0.01)
        if which == 'e02d_0125':
            dt = 0.0125
        if which == 'e03d_025':
            dt = 0.025
        for coh, case in C.e0_cases(d):
            _, meta = C.load_e0(d, coh, case)
            row = dict(s=C.steepness(meta, d), nu=float(meta['desc']['nu']), e={}, secs={})
            for N in Ns:
                lat, fin, sec = run(meta['desc'], N, dt, d)
                row['e'][N] = score_e0(lat, d, coh, case) if fin else None
                row['secs'][N] = sec
            if which == 'e03d':      # reproduction gate against speed3d f1 (H200 in-job scores), N16 dt 0.025
                lat, fin, _ = run(meta['desc'], 16, 0.025, 3)
                row['gate_N16_0.025'] = score_e0(lat, 3, coh, case)
            out['rows'][f'{coh}|{case}'] = row
            print(which, coh, case, round(row['s'], 1), {N: (None if v is None else f'{v:.2e}') for N, v in row['e'].items()}, flush=True)
        out['dt'] = dt
    elif which == 'reg':
        for g in C.REGIMES:
            dt = DT_REG[g]
            for case in C.regime_cases(g):
                ref, meta = C.load_regime(g, case)
                n0 = meta['judge']['n0']
                row = dict(s=float(meta['desc']['a']) / float(meta['desc']['nu']), nu=float(meta['desc']['nu']), e={}, dt=dt)
                for N in NSR:
                    lat, fin, _ = run(meta['desc'], N, dt, 2)
                    row['e'][N] = score_reg(lat, ref, n0) if fin else None
                out['rows'][f'{g}|{case}'] = row
                print(g, case, round(row['s'], 1), {N: (None if v is None else f'{v:.2e}') for N, v in row['e'].items()}, flush=True)
        out['dt'] = DT_REG
    elif which == 'stab':
        import math
        st = C.J(C.OUT / 'a2b_stab.json')
        prods = []
        for r in st['rows'].values():
            m = min(max([dt for dt in st['DTS'] if r['res'][f'{N}|{dt}']['stable']] or [0]) for N in st['NS'])
            if 0 < m < max(st['DTS']):
                prods.append(m * r['s'])
        c = min(prods)
        out['c'] = c
        cases = [('E0', coh, k) for coh, k in C.e0_cases(2)] + [(g, g, k) for g in C.REGIMES for k in C.regime_cases(g)]
        for src, coh, case in cases:
            if src == 'E0':
                ref, meta = C.load_e0(2, coh, case)
            else:
                ref, meta = C.load_regime(coh, case)
            n0 = meta['judge']['n0']
            sv = float(meta['desc']['a']) / float(meta['desc']['nu'])
            dt = min(0.0125, 0.05 / math.ceil(0.05 * sv / c))
            row = dict(s=sv, nu=float(meta['desc']['nu']), dt=dt, steps=round(0.25 / dt), e={})
            for N in NSS:
                lat, fin, _ = run(meta['desc'], N, dt, 2)
                row['e'][N] = (score_e0(lat, 2, coh, case) if src == 'E0' else score_reg(lat, ref, n0)) if fin else None
            out['rows'][f'{coh}|{case}'] = row
            print(which, coh, case, round(sv, 1), row['steps'], {N: (None if v is None else f'{v:.2e}') for N, v in row['e'].items()}, flush=True)
    (C.OUT / f'a2_{which}.json').write_text(json.dumps(out, indent=0))


if __name__ == '__main__':
    main(sys.argv[1])
