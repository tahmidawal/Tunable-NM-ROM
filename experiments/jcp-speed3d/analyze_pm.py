"""jcp-speed3d analysis of p1 and m1 (DESIGN H1.4/H2 final verdicts from the A1.5 confirmation panels, H3.1-H3.3, A1.13,
A2.7(d), H4.1-H4.3 with A4.4, A6, A9, A10; the deferred A4.5 items). Pure post-processing on CPU of the pulled jobs and
their E0 scores (score3.py -> checks/scores-p1.json, checks/scores-m1.json), the screening analysis
(checks/analysis-grid.json), the screening selection and the FOM matches (checks/SELECTION-SCREEN.json,
checks/fom-match.json) and the f1 FOM ladder. Every predicate is recomputed from raw records (rows, invocations,
distances, scores); nothing is copied from a summary field without its raw check.

    python analyze_pm.py [--p1 runs/p1/archive/output] [--m1 runs/m1/archive/output] [--out checks/analysis-pm.json]

Conventions. A timing panel's paired ratio is t_B / (t_A1 + t_A2)/2 per invocation; a panel's ratio for subject B is the
median over its invocations; a panel counts only if valid (hashes, finite ratios, subject drift in [1/1.1, 1.1]).
Confirmation panels: A = comparator (or FS-K2 / B*), B = selected configuration, so S = 1 / ratio. Route-4 panels:
A = ROM, B = FOM, so the ratio is the ROM speed-up t_FOM / t_ROM directly.
"""
import offline  # noqa: F401  (CPU, x64, highest; must be first)
import argparse
import json
from pathlib import Path

import numpy as np

import analyze as AN

HERE = Path(__file__).resolve().parent
BAR, BAR_ADT, POOL = AN.BAR, AN.BAR_ADT, AN.POOL
FAM_NAME = {'UB': 'upwind + BE (F-UB)', 'UC': 'upwind + CN-R (F-UC)', 'CC': 'central + CN-R (F-CC)', 'PS': 'sine pseudo-spectral ETDRK4 (F-PS)'}


def jl(p):
    return json.loads(Path(p).read_text())


def panel(t, name=None):
    """(median ratio, median t_B, median t_A, n, valid) for subject `name` (or all invocations) of one panel."""
    if not t or not t.get('invocations'):
        return dict(ratio=None, t_B=None, t_A=None, n=0, valid=False, reason='panel missing or skipped')
    inv = [x for x in t['invocations'] if name is None or x['B'] == name]
    if not inv:
        return dict(ratio=None, t_B=None, t_A=None, n=0, valid=False, reason=f'no invocations of {name}')
    import timing as TM
    v = TM.panel_validity(t['invocations'])                  # recomputed from the raw invocations
    return dict(ratio=float(np.median([x['ratio'] for x in inv])), t_B=float(np.median([x['tB'] for x in inv])),
                t_A=float(np.median([.5 * (x['tA1'] + x['tA2']) for x in inv])), n=len(inv), valid=bool(v['valid']),
                valid_job=bool(t.get('valid')), worst_subject_drift=v['worst_subject_drift'],
                cases=sorted({x['case'] for x in inv}), reps=sorted({x['rep'] for x in inv}))


def scores_of(sc, key, cases):
    """E0 errors of one saved trajectory key on the given cases: dict(e, eta, ranked_ids, median, worst, all_cases)."""
    s = sc['scores'].get(key)
    if s is None:
        return None
    have = [j for j in cases if str(j) in s]
    rk = [j for j in have if s[str(j)]['eligible']]
    e = [s[str(j)]['e'] for j in rk]
    return dict(e=e, eta=[s[str(j)]['eps_ref'] for j in rk], ranked_ids=rk, median=float(np.median(e)) if e else None,
                worst=float(np.max(e)) if e else None, all_cases=len(have) == len(cases),
                finite=all(np.isfinite(s[str(j)]['e']) for j in have))


def rows_elig(res, tag, spec, cases):
    """E-a (validity) and E-b (certified non-stationary steps <= 1 % pooled) from the raw rows of one trajectory key."""
    rs = [r for r in res['rows'] if r['tag'] == tag and r['spec'] == spec]
    got = sorted(r['case'] for r in rs)
    nst, nstep = sum(r['nonstationary'] for r in rs), sum(r['steps_n'] for r in rs)
    return dict(rows_complete=got == sorted(cases), Ea=bool(rs) and all(r['valid'] for r in rs),
                Eb=bool(rs) and nst <= POOL * nstep, Eb_frac=nst / max(nstep, 1), nonstationary=nst, steps=nstep)


def ec(dist, valid):
    d = np.asarray(dist, float)[:, 1:].max(1)
    return dict(Ec=bool(all(valid) and AN.ec_pass(d)), dist_worst=float(d.max()), cases_above=int((d > BAR).sum()))


def tob_or_within(a, m, rel=0.01):
    """H3.2 (b): a's median and worst tie-or-beat m's (Step 2 rule) or exceed them by at most 1 % relatively."""
    ea, em, eta, ids = AN.common(a, m)
    med, wst = AN.ties_or_beats(ea, em, eta)
    med = med or float(np.median(ea)) <= (1 + rel) * float(np.median(em))
    wst = wst or float(np.max(ea)) <= (1 + rel) * float(np.max(em))
    excess = ea - em
    lim = np.maximum(2 * eta, .1 * em)
    return dict(b=bool(med and wst), b_median=bool(med), b_worst=bool(wst), c=bool((excess <= lim).all()),
                c_cases_failing=[int(ids[i]) for i in np.where(excess > lim)[0]], ranked=len(ids),
                median=float(np.median(ea)), worst=float(np.max(ea)), median_mesh=float(np.median(em)), worst_mesh=float(np.max(em)))


# ------------------------------------------------------------------------------------------------------------- p1
def analyze_p1(rp, sel, grid, scp):
    res = jl(rp / 'result.json')
    sc = jl(scp)
    assert Path(sc['run']).resolve() == rp.resolve() and sc['job_id'] == res['job_id'], 'scores-p1.json is of another run'
    cfg, P, T = res['config'], res['prof'], res['timing']
    blk = cfg['prof']
    cases = res['cohort']['cases']
    lab = {c['label']: c for c in blk['configs']}
    out = dict(job=res['job_id'], gpu=res['gpu'], commit=res['commit'], complete=res.get('complete'),
               cohort=dict(seed=res['cohort']['seed'], n=len(cases)), K4=sc['controls'], coverage={})
    # coverage (A10: complete=true is not acceptance)
    cov = out['coverage']
    cov['confirm_panels'] = {f'{a}|{b}': bool(T.get(f'confirm|{a}|{b}', {}).get('invocations')) for a, b in blk['confirm']}
    cov['mf'] = {l_: l_ in P['mf'] for l_ in blk['mf']}
    cov['profile'] = {l_: f'profile|{l_}' in T for l_ in blk['profile']}
    cov['batch'] = {l_: {str(B): str(B) in {str(k) for k in P['batch'].get(l_, {})} for B in blk['batch']['sizes']} for l_ in blk['batch']['labels']}
    cov['conv'] = {l_: l_ in P['conv'] for l_ in blk['conv']}
    cov['reduced_adt'] = {l_: l_ in P['reduced_adt'] for l_ in blk['reduced_adt']}
    # mesh-path reproduction of the grid accuracy (p1 re-runs every configuration; must equal the grid by construction)
    rep = {}
    for l_, c in lab.items():
        key = f"{c['setting']}|{c['rule']}|{c['spec']}"
        a = scores_of(sc, key, cases)
        g = grid['configs'][key]
        if a is None:
            rep[l_] = None
            continue
        ge = dict(zip(g['ranked_ids'], g['e']))
        rep[l_] = float(max(abs(e_ - ge[j]) for j, e_ in zip(a['ranked_ids'], a['e'])))
    out['mesh_reproduction_max_abs_e'] = rep
    # ---- confirmation panels (A1.5): final H2 and H1.4 verdicts
    conf = {}
    for a, b in blk['confirm']:
        p = panel(T.get(f'confirm|{a}|{b}'), b)
        p['S'] = None if not p['ratio'] else 1. / p['ratio']
        conf[f'{a}|{b}'] = p
    out['confirm'] = conf
    H2 = {}
    for c, v in sel['comparators'].items():
        w, bs = v['winner_label'], v['B_star_label']
        pw = conf.get(f'{c}|{w}') if w and w != c else None
        pb = conf.get(f'{c}|{bs}') if bs and bs != c else None
        pbw = conf.get(f'{bs}|{w}') if (w and bs and w != bs) else None
        S = 1. if w == c else (pw['S'] if pw and pw['valid'] else None)
        H2[c] = dict(winner=v['winner'], winner_label=w, B_star=v['B_star'], B_star_label=bs, S_screen=v['S_screen'],
                     S_confirmed=S, panel_valid=None if w == c else (pw or {}).get('valid'),
                     verdict=('unavailable (no confirmed panel)' if S is None else ('pass' if S >= 1 / .6 else 'fail')),
                     S_BE_confirmed=1. if bs == c else (pb['S'] if pb and pb['valid'] else None),
                     S_scheme_confirmed=1. if w == bs else (pbw['S'] if pbw and pbw['valid'] else None),
                     S_BE_screen=v['S_BE_screen'])
    out['H2'] = H2
    H14 = {}
    for p_ in sel['h14_pairs']:
        k = f"{p_['fs2']}|{p_['qn']}"
        c_ = conf.get(k)
        r = c_['ratio'] if c_ and c_['valid'] else None
        H14[p_['cell']] = dict(fs2=sel['labels'][p_['fs2']], qn=sel['labels'][p_['qn']], ratio_screen=grid['H1']['H14'][p_['cell']]['ratio_screen'],
                               ratio_confirmed=r, verdict='unavailable (no valid panel)' if r is None else ('pass' if r <= .6 else 'fail'))
    for cell, v in grid['H1']['H14'].items():
        if cell not in H14:
            H14[cell] = dict(fs2=v['fs2'], qn=v['qn'], ratio_screen=None, ratio_confirmed=None, verdict=v['verdict'])
    out['H14'] = H14
    # ---- H3.2 mesh-free path + K5
    H32 = {}
    for l_ in blk['mf']:
        c = lab[l_]
        sn, rule, spec = c['setting'], c['rule'], c['spec']
        m = P['mf'].get(l_)
        if m is None or 'invalid' in m:
            H32[l_] = dict(qualified=False, reason=(m or {}).get('invalid', 'not run'))
            continue
        e_ab = rows_elig(res, f'{sn}|{rule}|mf', spec, cases)
        e_c = ec(m['dist_to_adt'], m['adt_valid'])
        a = scores_of(sc, f'{sn}|{rule}|mf|{spec}', cases)
        mm = scores_of(sc, f'{sn}|{rule}|{spec}', cases)
        q = tob_or_within(a, mm)
        elig = bool(e_ab['rows_complete'] and e_ab['Ea'] and e_ab['Eb'] and e_c['Ec'])
        k5 = P['k5'].get(l_, {})
        k5r = None
        if k5.get('rank_ok'):
            ak = scores_of(sc, f"{sn}|{rule}|mfproj_{k5['rule']}|{spec}", cases)
            k5r = None if ak is None else tob_or_within(ak, mm)
        H32[l_] = dict(key=f'{sn}|{rule}|{spec}', eligible=elig, **e_ab, **e_c, cond=m['cond'], **q,
                       qualified=bool(elig and q['b'] and q['c']),
                       K5=dict(rule=k5.get('rule'), cond=k5.get('cond'), rank_ok=k5.get('rank_ok'),
                               fails=None if k5r is None else bool(not (k5r['b'] and k5r['c'])), detail=k5r))
    out['H32'] = H32
    out['K5_fires'] = any(v.get('K5', {}).get('fails') for v in H32.values())
    # ---- H3.1 profile (indicative) and G3
    H31 = {}
    for l_ in blk['profile']:
        t = T.get(f'profile|{l_}')
        if not t or not t.get('invocations'):
            H31[l_] = dict(missing=True)
            continue
        import timing as TM
        v = TM.panel_validity(t['invocations'])
        comp = {s: float(np.median([x['tB'] for x in t['invocations'] if x['B'] == s])) for s in t['subjects']}
        fused = float(np.median([.5 * (x['tA1'] + x['tA2']) for x in t['invocations']]))
        parts = comp['project'] + comp['evolve'] + comp['decode']
        H31[l_] = dict(fused_seconds=fused, components_seconds=comp,
                       share={k: comp[k] / fused for k in ('project', 'evolve', 'decode')},
                       share_adaptive_start=(comp['adaptive_start'] / fused) if 'adaptive_start' in comp else None,
                       G3_sum_over_fused=parts / fused, G3=bool(abs(parts / fused - 1) <= .15), panel_valid=bool(v['valid']),
                       compile_estimate_seconds={k: x for k, x in res.get('compile_estimate_seconds', {}).items()
                                                 if k.startswith(f"{lab[l_]['setting']}|{lab[l_]['rule']}|") and k.endswith(lab[l_]['spec'])})
    out['H31'] = H31
    # ---- H3.3 batching
    H33 = {}
    for l_, rec in P['batch'].items():
        rr = {str(k): v for k, v in rec.items()}
        thr = {k: v.get('throughput') for k, v in rr.items()}
        H33[l_] = dict(per_B=rr, G4={k: v.get('G4') for k, v in rr.items()},
                       K6_fires={k: v.get('K6_fires') for k, v in rr.items() if v.get('K6_fires') is not None},
                       K6plus_pass={k: v.get('K6plus_pass') for k, v in rr.items()},
                       throughput_16_over_1=(thr['16'] / thr['1']) if thr.get('16') and thr.get('1') else None,
                       bar_2x=bool(thr.get('16') and thr.get('1') and thr['16'] >= 2 * thr['1']),
                       latency_B1=rr.get('1', {}).get('median_seconds'))
    out['H33'] = H33
    # ---- A1.13 converged-rule sensitivity, A2.7(d) reduced-rule ADT
    out['conv'] = {l_: dict(rule=v['rule'], delta_quad_max=float(np.asarray(v['delta_quad'])[:, 1:].max()),
                            delta_quad_median=float(np.median(np.asarray(v['delta_quad'])[:, 1:].max(1))))
                   for l_, v in P['conv'].items()}
    out['reduced_adt'] = {l_: dict(worst=v['worst'], ok=bool(v['worst'] <= BAR_ADT and all(v['tight_valid']) and all(v['tighter_valid'])))
                          for l_, v in P['reduced_adt'].items()}
    return out


# ------------------------------------------------------------------------------------------------------------- m1
def analyze_m1(rp, sel, grid, fm, f1, scp):
    res = jl(rp / 'result.json')
    sc = jl(scp)
    assert Path(sc['run']).resolve() == rp.resolve() and sc['job_id'] == res['job_id'], 'scores-m1.json is of another run'
    cfg, M_, T = res['config'], res['mesh'], res['timing']
    blk = cfg['mesh']
    cases = res['cohort']['cases']
    out = dict(job=res['job_id'], gpu=res['gpu'], commit=res['commit'], complete=res.get('complete'), K4=sc['controls'],
               G5=M_['G5'], G5_all=M_['G5_all'], label='mesh-free / scoring-lattice contract (not the full-field query)',
               per_L={}, H41={}, H43={}, panel_failures={})
    f1fd, f1ps = f1['fom']['fd'], f1['fom']['ps']

    def f1_rec(fk):
        for v in list(f1fd.values()) + list(f1ps.values()):
            if v.get('key') == fk:
                return v
        return None

    def fom_scores(fk):
        """f1 errors (64 cases) of a ladder FOM, aligned like the matcher."""
        v = f1_rec(fk)
        cs = v['cases']
        rk = [j for j in cases if cs[str(j)].get('eligible')]
        return dict(e=[cs[str(j)]['e'] for j in rk], eta=[cs[str(j)]['eps_ref'] for j in rk], ranked_ids=rk)
    roms = {r['label']: r for r in blk['roms']}
    for L in blk['Ls']:
        n = L + 1
        E = M_['per_L'].get(str(L))
        o = out['per_L'][str(L)] = dict(rom={}, fom={}, panels={})
        if E is None:
            o['missing'] = True
            continue
        for l_, r in roms.items():
            er = E['rom'].get(l_)
            if er is None or 'infeasible' in er:
                o['rom'][l_] = dict(feasible=False, detail=(er or {}).get('infeasible', 'not run'))
                continue
            tag = f"{r['setting']}|{r['rule']}|mf|n{n}"
            e_ab = rows_elig(res, tag, r['spec'], cases)
            e_c = ec(er['dist_to_adt'], er['adt_valid'])
            tt = bool(er['adt_tight_tighter'] <= BAR_ADT and all(er['adt_tighter_valid']))
            s = scores_of(sc, f"{tag}|{r['spec']}", cases)
            el = bool(e_ab['rows_complete'] and e_ab['Ea'] and e_ab['Eb'] and e_c['Ec'] and tt)
            o['rom'][l_] = dict(feasible=True, M=er['tables'].get('M'), table_seconds=er['tables'].get('seconds_A'), **e_ab, **e_c,
                                adt_tight_tighter=er['adt_tight_tighter'], adt_gate=tt, eligible=el,
                                median=s['median'], worst=s['worst'], ranked=len(s['ranked_ids']), e=s['e'], eta=s['eta'],
                                ranked_ids=s['ranked_ids'])
            p = T.get(f'mesh|L{L}|{l_}')
            if p and p.get('invocations'):
                import timing as TM
                v = TM.panel_validity(p['invocations'])
                o['panels'][l_] = dict(valid=bool(v['valid']), rom_seconds=float(np.median([.5 * (x['tA1'] + x['tA2']) for x in p['invocations']])),
                                       speedup={b: panel(p, b) for b in p['subjects']})
        for fk, fv in E['fom'].items():
            ent = dict(key=fv.get('key'), matched_for=fv.get('matched_for'), calibration=(fv.get('calibration') or {}).get('chosen'),
                       calibration_status=(fv.get('calibration') or {}).get('status'), infeasible=fv.get('infeasible'))
            cs = fv.get('cases') or {}
            if cs:
                e = [c['e'] for c in cs.values() if c.get('eligible')]
                ent.update(n_cases=len(cs), converged=all(c.get('converged') for c in cs.values()),
                           median=float(np.median(e)) if e else None, worst=float(np.max(e)) if e else None,
                           seconds_first_median=float(np.median([c['seconds_first'] for c in cs.values()])))
            o['fom'][fk] = ent
        if E.get('panel_failures'):
            out['panel_failures'][str(L)] = E['panel_failures']
    # H4.1 flat ROM cost: absolute warm query time over meshes (mesh-free path)
    for l_ in roms:
        ts = {L: out['per_L'][str(L)]['panels'].get(l_, {}).get('rom_seconds') for L in blk['Ls']}
        vals = [t for t in ts.values() if t]
        allL = len(vals) == len(blk['Ls'])
        out['H41'][l_] = dict(seconds=ts, max_over_min=(max(vals) / min(vals)) if vals else None,
                              verdict=('incomplete (not every mesh timed)' if not allL else
                                       ('pass' if max(vals) / min(vals) <= 1.25 else 'fail')))
    # H4.3 matched accuracy (A6): primary = match|<label> panel, mesh path n = 65; families and union
    for l_ in roms:
        mrec = fm['matches'].get(l_, {})
        p = T.get(f'match|{l_}')
        fam = {}
        for f, m in mrec.items():
            if m.get('matched'):
                pp = panel(p, m['matched']) if p else dict(valid=False, ratio=None)
                fam[f] = dict(matched=m['matched'], boundary=m['boundary'], fom_median=m['median'], fom_worst=m['worst'],
                              speedup=pp['ratio'] if pp.get('valid') else None, panel=pp,
                              reaches_2x=None if not pp.get('valid') else bool(pp['ratio'] >= 2),
                              label='upper bound (finite FOM ladder)' + ('; boundary (cheapest admissible member)' if m['boundary'] else ''))
            else:
                fam[f] = dict(matched=None, status=m['status'], speedup=None, reaches_2x=None,
                              label='no FOM configuration in the ladder ties-or-beats the ROM on median and worst')
        got = [v for v in fam.values() if v['speedup'] is not None]
        uni = min(got, key=lambda v: v['speedup']) if got else None
        # A6: mesh-free panels at n > 65 count as matched only if that mesh's ROM is still tie-or-beat by the FOM
        per_mesh = {}
        for L in blk['Ls']:
            rr = out['per_L'][str(L)]['rom'].get(l_, {})
            pl = out['per_L'][str(L)]['panels'].get(l_, {})
            row = {}
            for f, v in fam.items():
                if not v['matched'] or not rr.get('feasible'):
                    continue
                fs = fom_scores(v['matched'])
                ef, er_, eta, ids = AN.common(fs, rr)
                med, wst = AN.ties_or_beats(ef, er_, eta)
                sp = (pl.get('speedup') or {}).get(v['matched'], {})
                row[f] = dict(still_matched=bool(med and wst), speedup=sp.get('ratio') if pl.get('valid') else None,
                              label='matched' if (med and wst) else 'unmatched same-job timing')
            per_mesh[str(L)] = row
        out['H43'][l_] = dict(rom_key=f"{roms[l_]['setting']}|{roms[l_]['rule']}|{roms[l_]['spec']}", families=fam,
                              union=None if uni is None else dict(family=[f for f, v in fam.items() if v is uni][0], matched=uni['matched'],
                                                                  speedup=uni['speedup'], reaches_2x=uni['reaches_2x']),
                              per_mesh=per_mesh,
                              wording='a finite FOM ladder gives an upper bound on optimal FOM cost, hence an upper bound on the claimed ROM speed-up')
    # H-comp cells (V3-5) at the largest mesh and K2 (governing plan): speed-up >= 2x against the best validated FOM
    Lmax = str(max(blk['Ls']))
    cells = {}
    for l_ in roms:
        fam = out['H43'][l_]['families']
        pm = out['H43'][l_]['per_mesh'].get(Lmax, {})
        c = {}
        for f in AN_FAMS:
            v = pm.get(f)
            if fam.get(f, {}).get('matched') is None:
                c[f] = dict(status='no ladder match (ROM more accurate than every ladder member on median or worst)', reaches_2x=None)
            elif v is None or v['speedup'] is None:
                c[f] = dict(status='not timed / panel invalid at the largest mesh', reaches_2x=None)
            else:
                c[f] = dict(status=v['label'], speedup=v['speedup'], reaches_2x=bool(v['still_matched'] and v['speedup'] >= 2))
        timed = [x for x in c.values() if x.get('speedup') is not None]
        best = min(timed, key=lambda x: x['speedup']) if timed else None
        c['union'] = dict(speedup=None if best is None else best['speedup'], reaches_2x=None if best is None else best['reaches_2x'])
        cells[l_] = c
    out['Hcomp_largest_mesh'] = dict(L=int(Lmax), cells=cells)
    out['K2_fires'] = bool(all(c['union']['reaches_2x'] is False for c in cells.values()))
    # same-mesh H4.2 diagnostic
    H42 = {}
    for L in blk['Ls']:
        o = out['per_L'][str(L)]
        for l_, pl in o['panels'].items():
            for b, sp in pl['speedup'].items():
                H42.setdefault(str(L), {}).setdefault(l_, {})[b] = sp['ratio']
    out['H42'] = H42
    return out


AN_FAMS = ('UB', 'UC', 'CC', 'PS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1', default=str(HERE / 'runs' / 'p1' / 'archive' / 'output'))
    ap.add_argument('--m1', default=str(HERE / 'runs' / 'm1' / 'archive' / 'output'))
    ap.add_argument('--scores-p1', default=str(HERE / 'checks' / 'scores-p1.json'))
    ap.add_argument('--scores-m1', default=str(HERE / 'checks' / 'scores-m1.json'))
    ap.add_argument('--out', default=str(HERE / 'checks' / 'analysis-pm.json'))
    a = ap.parse_args()
    grid = jl(HERE / 'checks' / 'analysis-grid.json')
    sel = jl(HERE / 'checks' / 'SELECTION-SCREEN.json')
    fm = jl(HERE / 'checks' / 'fom-match.json')
    f1 = jl(HERE / 'runs' / 'f1' / 'archive' / 'output' / 'result.json')
    out = dict(label=AN.LABEL)
    if (Path(a.p1) / 'result.json').exists():
        out['p1'] = analyze_p1(Path(a.p1).resolve(), sel, grid, a.scores_p1)
    if (Path(a.m1) / 'result.json').exists():
        out['m1'] = analyze_m1(Path(a.m1).resolve(), sel, grid, fm, f1, a.scores_m1)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    if 'p1' in out:
        for c, v in out['p1']['H2'].items():
            print('H2', c, v['verdict'], v['S_confirmed'], 'screen', v['S_screen'])
        print('H14', {k: v['verdict'] for k, v in out['p1']['H14'].items()})
        print('H32', {k: v.get('qualified') for k, v in out['p1']['H32'].items()}, 'K5', out['p1']['K5_fires'])
    if 'm1' in out:
        print('H41', {k: v['verdict'] for k, v in out['m1']['H41'].items()})
        print('H43', {k: {f: x['speedup'] for f, x in v['families'].items()} for k, v in out['m1']['H43'].items()})
        print('K2 fires', out['m1']['K2_fires'])


if __name__ == '__main__':
    main()
