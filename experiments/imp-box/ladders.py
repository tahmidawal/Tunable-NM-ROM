"""imp-box stage `ladder`: every arm's knob ladder on val32 (DESIGN §4, A1.3, A2.3-A2.4, A3.6). Rows go to ladder.jsonl
(`ladder.Ladder`, imp-geometry, unchanged except `stats_valid`).

Arms / families (cfg['fam']):
  spec    F-SPEC     N in SPEC_N (+ extensions 384, 512), ETDRK4 step ladder from n = 5
  specad  F-SPEC-AD  N in SPEC_N (+ ext.), tol in AD_TOLS (+ 1e-8)
  podg    R-PODG     r, rule q in {4..12} (point form) or 'T12' (tensor on q = 12), Galerkin ETDRK4, steps from 10
  trans   R-TRANS    (W40) the same on the centred POD, shifted decode
  ecsw    R-ECSW     r, kappa in {2, 4, 8} on the parent rule (k_q, 12), Galerkin ETDRK4 point form, steps from 10
  bank    R-BANK     (W40, W08) r, rule q / 'T12', steppers lspg_cnr (B-LSPG) and gal_etdrk4 (B-E4), steps from 10
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import geom as G
import ladder as LD
import podsine as PS
import rom as RM
import spec as SP

SPEC_N = (8, 10, 12, 14, 16, 24, 32, 48, 64, 96, 128, 192, 256)
EXCLUDED = {('spec', 'fft', 12)}      # A10: the H200 compiled-loop fault (fixed-step FFT N = 12 only); 10 and 14 added
SPEC_EXT = (384, 512)
DENSE_MAX_N = 128                     # A11: dense transforms only where (M-1)^2 (N-1) matmuls stay cheap
AD_TOLS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
AD_EXT = 1e-8
R_LADDER = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
RULES = (4, 6, 8, 10, 12)
SUCC = {4: 6, 6: 8, 8: 10, 10: 12, 12: 14}
TAUS = (0.01, 0.03, 0.10)
ECSW_ROWS, ECSW_SNAPS = tuple(range(12)), tuple(range(0, 21, 2))
R_POD_COLS = 384


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


from builders import (pod_spaces, bank_spaces, space_of, form_of, build_spec, build_reduced,  # noqa: E402,F401
                      build_ecsw)


# ------------------------------------------------------------------------------------------------ helpers ------

def rows_of(L, fam):
    return [r for r in L.done.values() if r['cfg'].get('fam') == fam and r.get('valid')]


def fastest_meeting(rows, U, tau, stat):
    ok = [r for r in rows if r.get('screen') is not None and r[stat] + U <= tau]
    return min(ok, key=lambda r: (r['screen'], r['key'])) if ok else None


def flop_skip(sp, q, r, st):
    """A5 flop screen: a point-form configuration is skipped when its per-evaluation flop count (Jacobian included for
    LSPG) exceeds 10x that of the tensor form at the same (r, M_t); it cannot be the cheaper charged form."""
    if q == 'T12':
        return False
    m = sp.P.shape[0]
    if st == 'lspg_cnr':
        return m > 10 * r
    return 4 * m > 10 * r * r


def ranks(R):
    return [r for r in R_LADDER if r <= R]


# -------------------------------------------------------------------------------------------- the stage --------

def stage_ladder(a, out, D):
    rj = json.loads((out / 'ref.json').read_text())
    U_allow = rj['allowance']['U']
    import cell as CL
    Dv, Uv = CL.eligible(D['val'], np.load(out / 'refs_val.npz'))
    L = LD.Ladder(out / 'ladder.jsonl', Uv, Dv, U_allow, tag='val')
    smoke = a.smoke
    cov = dict(spec_ext=[], specad_ext=[], rank_absent=[])

    # ---- F-SPEC and F-SPEC-AD
    Ns = SPEC_N[:3] if smoke else SPEC_N
    tols = AD_TOLS[:2] if smoke else AD_TOLS
    if smoke and os.environ.get('IMPBOX_TINY'):
        Ns, tols = SPEC_N[:2], AD_TOLS[:1]
    for N in Ns:
        if ('spec', 'fft', N) in EXCLUDED and not smoke:
            cov.setdefault('excluded', []).append(f'spec|N{N}|fft')
        else:
            L.step_ladder(f'spec|N{N}', dict(fam='spec', N=N), build_spec, 5)
        for tol in tols:
            L.run(f'specad|N{N}|tol{tol:.0e}', dict(fam='specad', N=N, tol=tol), build_spec)
        if N <= DENSE_MAX_N:                       # A11: dense-transform implementation of the same discretisation
            L.step_ladder(f'spec|N{N}|dense', dict(fam='spec', N=N, xform='dense'), build_spec, 5)
            for tol in tols:
                L.run(f'specad|N{N}|tol{tol:.0e}|dense', dict(fam='specad', N=N, tol=tol, xform='dense'), build_spec)
    for fam in ('spec', 'specad'):                                   # A2.4 / A3.6 extensions
        for ext in SPEC_EXT:
            need = False
            for tau in TAUS:
                for stat in ('median', 'worst'):
                    best = fastest_meeting(rows_of(L, fam), U_allow, tau, stat)
                    nmax = max(r['cfg']['N'] for r in rows_of(L, fam)) if rows_of(L, fam) else 0
                    if best is None or best['cfg']['N'] == nmax:
                        need = True
            if not need or smoke:
                break
            cov[f'{fam}_ext'].append(ext)
            if fam == 'spec':
                L.step_ladder(f'spec|N{ext}', dict(fam='spec', N=ext), build_spec, 5)
            else:
                for tol in tols:
                    L.run(f'specad|N{ext}|tol{tol:.0e}', dict(fam='specad', N=ext, tol=tol), build_spec)
    for tau in TAUS:                                                 # AD tolerance extension
        for stat in ('median', 'worst'):
            best = fastest_meeting(rows_of(L, 'specad'), U_allow, tau, stat)
            if best is not None and best['cfg']['tol'] == min(tols) and not smoke:
                c = dict(best['cfg'], tol=AD_EXT)
                xs = '|dense' if c.get('xform', 'fft') == 'dense' else ''
                L.run(f"specad|N{c['N']}|tol{AD_EXT:.0e}{xs}", c, build_spec)
                cov['specad_ext'].append(dict(tau=tau, stat=stat, N=c['N'], tol=AD_EXT))
    log('fom ladder rows', len(L.done))

    # ---- R-PODG and R-ECSW (shared POD)
    sps, Phi, kq = pod_spaces(out, 'pod', rules=(RULES[:1] + (12, 14)) if smoke else RULES + (14,))
    R = sps[12].R
    rl = ranks(R)[:2] if smoke else ranks(R)
    cov['rank_absent'] += [('pod', r) for r in R_LADDER if r > R]
    rules = (4, 'T12') if smoke else RULES + ('T12',)
    br = build_reduced(sps)
    for r in rl:
        for q in rules:
            if flop_skip(space_of(sps, q), q, r, 'gal_etdrk4'):
                cov.setdefault('flop_skipped', []).append(f'podg|r{r}|q{q}')
                continue
            L.step_ladder(f'podg|r{r}|q{q}', dict(fam='podg', r=r, q=q, st='gal_etdrk4'), br, 10)
    Z = np.load(out / 'pod_Z.npy')
    idx = [j * 21 + s for j in ECSW_ROWS for s in ECSW_SNAPS if j * 21 + s < len(Z)]
    parent = sps[12]
    subs, fits = {}, []
    for r in rl:
        t0 = time.perf_counter()
        fit, info = PS.ecsw_fit(parent, r, r, Z[idx])
        fits.append(dict(r=r, seconds=time.perf_counter() - t0, info=info,
                         sizes={str(k): (None if v is None else v['size']) for k, v in fit.items()},
                         relres={str(k): (None if v is None else v['relres']) for k, v in fit.items()}))
        for kap, x in fit.items():
            if x is not None:
                subs[(r, kap)] = parent.with_weights(x['nodes'], x['weights'])
                np.savez(out / f'ecsw_r{r}_k{kap}.npz', nodes=x['nodes'], weights=x['weights'])
    (out / 'ecsw_fits.json').write_text(json.dumps(fits, indent=1))
    be = build_ecsw(subs)
    for (r, kap) in sorted(subs):
        L.step_ladder(f'ecsw|r{r}|k{kap}', dict(fam='ecsw', r=r, kappa=kap, st='gal_etdrk4', q=12), be, 10)
    del sps, subs, parent, br, be
    RM.clear_cache()
    log('pod/ecsw ladder rows', len(L.done))

    # ---- R-TRANS (W40)
    if a.trans:
        sps, Phic, _ = pod_spaces(out, 'podc', rules=(RULES[:1] + (12, 14)) if smoke else RULES + (14,))
        bt = build_reduced(sps, Phic)
        for r in (ranks(sps[12].R)[:2] if smoke else ranks(sps[12].R)):
            for q in rules:
                if flop_skip(space_of(sps, q), q, r, 'gal_etdrk4'):
                    cov.setdefault('flop_skipped', []).append(f'trans|r{r}|q{q}')
                    continue
                L.step_ladder(f'trans|r{r}|q{q}', dict(fam='trans', r=r, q=q, st='gal_etdrk4'), bt, 10)
        del sps, bt
        RM.clear_cache()
        log('trans ladder rows', len(L.done))

    # ---- R-BANK
    if a.bank and (out / 'bank.pkl').exists():
        sps, b = bank_spaces(out, rules=(RULES[:1] + (12, 14)) if smoke else RULES + (14,))
        Rb = sps[12].R
        cov['rank_absent'] += [('bank', r) for r in R_LADDER if r > Rb]
        bb = build_reduced(sps)
        for r in (ranks(Rb)[:2] if smoke else ranks(Rb)):
            for st in ('lspg_cnr', 'gal_etdrk4'):
                for q in rules:
                    if flop_skip(space_of(sps, q), q, r, st):
                        cov.setdefault('flop_skipped', []).append(f'bank|r{r}|q{q}|{st}')
                        continue
                    L.step_ladder(f'bank|r{r}|q{q}|{st}', dict(fam='bank', r=r, q=q, st=st, tol='prod'), bb, 10)
        del sps, bb
        RM.clear_cache()
        log('bank ladder rows', len(L.done))
    (out / 'ladder_coverage.json').write_text(json.dumps(cov, indent=1, default=str))
