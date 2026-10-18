"""jcp-speed3d analysis of p1 and m1 (DESIGN H1.4/H2 final verdicts from the A1.5 confirmation panels, H3.1-H3.3, A1.13,
A2.7(d), H4.1-H4.3 with A4.4, A6, A9, A10; the deferred A4.5 items). Pure post-processing on CPU of the pulled jobs and
their E0 scores (score3.py -> checks/scores-p1.json, checks/scores-m1.json), the screening analysis
(checks/analysis-grid.json), the screening selection and the FOM matches (checks/SELECTION-SCREEN.json,
checks/fom-match.json) and the f1 FOM ladder.

    python analyze_pm.py [--p1 runs/p1/archive/output] [--m1 runs/m1/archive/output] [--out checks/analysis-pm.json]

Conventions. A paired ratio is recomputed per invocation from the raw times, t_B / ((t_A1 + t_A2)/2), and the drift as
t_A2 / t_A1; a subject's panel ratio is the median over its invocations. A panel counts only if (i) its A subject is the
expected one, (ii) the subject's invocations cover exactly the expected cases x repetitions, (iii) recomputed validity
holds (hashes, finite ratios, every subject's median drift in [1/1.1, 1.1]). Output hashes are compared in-job
(s3ctx.time_panel); the raw hashes are not retained, so `hashes_ok` is taken from the job (noted, A5).
Confirmation panels: A = comparator (or FS-K2 / B*), B = selected, so S = 1 / ratio. Route-4 panels: A = ROM, B = FOM, so
the ratio is the ROM speed-up t_FOM / t_ROM directly.
Missing evidence is never a pass or a fail: every verdict is three-state (pass / fail / unavailable-with-reason), and an
`acceptance` block lists every required item of the job with its presence (A10: complete = true is not acceptance).
"""
import offline  # noqa: F401  (CPU, x64, highest; must be first)
import argparse
import json
from pathlib import Path

import numpy as np

import analyze as AN
import timing as TM

HERE = Path(__file__).resolve().parent
BAR, BAR_ADT, POOL = AN.BAR, AN.BAR_ADT, AN.POOL
FAMS = ('UB', 'UC', 'CC', 'PS')


def jl(p):
    return json.loads(Path(p).read_text())


def recomputed(inv):
    return [dict(x, ratio=x['tB'] / (.5 * (x['tA1'] + x['tA2'])), drift=x['tA2'] / x['tA1']) for x in inv]


def panel(t, name, A=None, cases=None, reps=None):
    """Subject `name` of one panel: median recomputed ratio, t_B, t_A; validity per the module conventions."""
    if not t or not t.get('invocations'):
        return dict(ratio=None, valid=False, reason='panel missing or skipped' + (f" ({t.get('missing_expected')})" if t else ''))
    inv_all = recomputed(t['invocations'])
    inv = [x for x in inv_all if x['B'] == name]
    why = []
    if A is not None and t.get('A') != A:
        why.append(f"A is {t.get('A')}, expected {A}")
    if not inv:
        why.append(f'no invocations of {name}')
    if cases is not None and reps is not None:
        want = sorted((c, r) for c in cases for r in range(reps))
        if sorted((x['case'], x['rep']) for x in inv) != want:
            why.append('invocation population differs from cases x reps')
    v = TM.panel_validity(inv_all)
    if not v['valid']:
        why.append(f"panel invalid (hashes {v['hashes_ok']}, finite {v['ratios_finite']}, worst drift {v['worst_subject_drift']})")
    if not inv:
        return dict(ratio=None, valid=False, reason='; '.join(why))
    return dict(ratio=float(np.median([x['ratio'] for x in inv])), t_B=float(np.median([x['tB'] for x in inv])),
                t_A=float(np.median([.5 * (x['tA1'] + x['tA2']) for x in inv])), n=len(inv), valid=not why,
                reason='; '.join(why) or None, worst_subject_drift=v['worst_subject_drift'])


def scores_of(sc, key, cases):
    """E0 errors of one saved trajectory key on the given cases; ok only if every case is scored and finite."""
    s = sc['scores'].get(key)
    if s is None:
        return dict(ok=False, reason=f'no scores for {key}', e=[], eta=[], ranked_ids=[], median=None, worst=None)
    have = [j for j in cases if str(j) in s]
    fin = all(np.isfinite(s[str(j)]['e']) for j in have)
    rk = [j for j in have if s[str(j)]['eligible']]
    e = [s[str(j)]['e'] for j in rk]
    ok = len(have) == len(cases) and fin and len(rk) > 0
    return dict(ok=ok, reason=None if ok else f'scored {len(have)}/{len(cases)}, finite {fin}, ranked {len(rk)}',
                e=e, eta=[s[str(j)]['eps_ref'] for j in rk], ranked_ids=rk, median=float(np.median(e)) if e else None,
                worst=float(np.max(e)) if e else None)


def rows_elig(res, tag, spec, cases):
    """E-a (validity) and E-b (certified non-stationary steps <= 1 % pooled) from the raw rows of one trajectory key;
    the per-row count is recomputed from the certified ratios (gamma > 1e-3 or non-finite)."""
    rs = [r for r in res['rows'] if r['tag'] == tag and r['spec'] == spec]
    got = sorted(r['case'] for r in rs)
    nst = 0
    for r in rs:
        g = np.asarray(r.get('gam_cert', []), float)
        nst += int((~np.isfinite(g) | (g > 1e-3)).sum()) if len(g) else 0
    nstep = sum(r['steps_n'] for r in rs)
    cert_ok = all(len(r.get('gam_cert', [])) == r['steps_n'] for r in rs) if rs and spec.split('|')[0] != 'ad' else True
    return dict(rows_complete=got == sorted(cases) and cert_ok, Ea=bool(rs) and all(r['valid'] for r in rs),
                Eb=bool(rs) and nst <= POOL * nstep, Eb_frac=nst / max(nstep, 1), nonstationary=nst, steps=nstep)


def ec(dist, valid, ncases):
    d = np.asarray(dist, float)
    if d.ndim != 2 or d.shape[0] != ncases or not len(valid) == ncases:
        return dict(Ec=False, dist_worst=None, cases_above=None, reason='distance / validity population incomplete')
    d = d[:, 1:].max(1)
    return dict(Ec=bool(all(valid) and AN.ec_pass(d)), dist_worst=float(d.max()), cases_above=int((d > BAR).sum()))


def tob_or_within(a, m, rel=0.01):
    """H3.2 (b): a's median and worst tie-or-beat m's (Step 2 rule) or exceed them by at most 1 % relatively;
    (c): no case's error exceeds its mesh-path error by more than max(2 eta, 0.1 e_mesh). Same ranked cases required."""
    if not (a['ok'] and m['ok']) or a['ranked_ids'] != m['ranked_ids']:
        return dict(b=None, c=None, reason='score populations incomplete or differ')
    ea, em, eta, ids = AN.common(a, m)
    med, wst = AN.ties_or_beats(ea, em, eta)
    med = med or float(np.median(ea)) <= (1 + rel) * float(np.median(em))
    wst = wst or float(np.max(ea)) <= (1 + rel) * float(np.max(em))
    excess = ea - em
    lim = np.maximum(2 * eta, .1 * em)
    return dict(b=bool(med and wst), b_median=bool(med), b_worst=bool(wst), c=bool((excess <= lim).all()),
                c_cases_failing=[int(ids[i]) for i in np.where(excess > lim)[0]], ranked=len(ids),
                median=float(np.median(ea)), worst=float(np.max(ea)), median_mesh=float(np.median(em)), worst_mesh=float(np.max(em)))


def tri(xs):
    """All-of over every expected item: False if any measured False, True only if every item is measured True, else None."""
    if any(x is False for x in xs):
        return False
    return True if xs and all(x is True for x in xs) else None


def cond_ok(c, Rp):
    return bool(c and np.isfinite(c.get('cond', np.nan)) and c['cond'] < 1e10 and c.get('m', 0) >= Rp)


# ------------------------------------------------------------------------------------------------------------- p1
def analyze_p1(rp, sel, grid, scp):
    import make_configs as MC
    res = jl(rp / 'result.json')
    sc = jl(scp)
    assert Path(sc['run']).resolve() == rp.resolve() and sc['job_id'] == res['job_id'], 'scores-p1.json is of another run'
    cfg, P, T = res['config'], res['prof'], res['timing']
    blk, tc = cfg['prof'], cfg['timing']
    cases = res['cohort']['cases']
    lab = {c['label']: c for c in blk['configs']}
    Rp = {s: v['Rp'] for s, v in MC.SETTINGS.items()}
    out = dict(job=res['job_id'], gpu=res['gpu'], commit=res['commit'], complete=res.get('complete'),
               cohort=dict(seed=res['cohort']['seed'], n=len(cases)), K4=sc['controls'])
    ccases = cases[:blk.get('confirm_cases', 8)]
    # ---- confirmation panels (A1.5)
    conf = {}
    for a, b in blk['confirm']:
        p = panel(T.get(f'confirm|{a}|{b}'), b, A=a, cases=ccases, reps=blk.get('confirm_reps', 3))
        p['S'] = None if not p.get('ratio') else 1. / p['ratio']
        conf[f'{a}|{b}'] = p
    out['confirm'] = conf
    # ---- A2.7(d) reduced-rule ADT (finalists) and A1.13 converged-rule sensitivity
    red = {}
    for l_ in blk['reduced_adt']:
        v = P['reduced_adt'].get(l_)
        if v is None:
            red[l_] = dict(ok=None, reason='missing')
            continue
        n8 = min(8, len(cases))
        okv = len(v['tight_valid']) == n8 and len(v['tighter_valid']) == n8 and all(v['tight_valid']) and all(v['tighter_valid'])
        red[l_] = dict(worst=v['worst'], ok=bool(okv and np.isfinite(v['worst']) and v['worst'] <= BAR_ADT))
    out['reduced_adt'] = red
    conv = {}
    for l_, crule in blk['conv'].items():
        v = P['conv'].get(l_)
        if v is None or v.get('rule') != crule or np.asarray(v['delta_quad']).shape[0] != len(cases):
            conv[l_] = dict(rule=crule, available=False)
            continue
        c_ = lab[l_]
        rv = rows_elig(res, f"{c_['setting']}|{crule}", c_['spec'], cases)
        d = np.asarray(v['delta_quad'], float)[:, 1:].max(1)
        if not (rv['rows_complete'] and rv['Ea'] and np.isfinite(d).all()):
            conv[l_] = dict(rule=crule, available=False, reason='converged-rule rollout incomplete, invalid or non-finite', rows=rv)
            continue
        conv[l_] = dict(rule=crule, available=True, delta_quad_max=float(d.max()), delta_quad_median=float(np.median(d)),
                        converged_rule_Eb=rv['Eb'])
    out['conv'] = conv

    def red_gate(l_):
        """A1.4/A2.7(d): a finalist on a reduced rule is selectable only if its reduced-rule ADT tight/tighter check passed."""
        if l_ not in red:
            return True, None
        return red[l_]['ok'] is True, (None if red[l_]['ok'] else f"reduced-rule ADT gate {red[l_]['ok']}")
    H2 = {}
    for c, v in sel['comparators'].items():
        w, bs = v['winner_label'], v['B_star_label']
        pw = conf.get(f'{c}|{w}') if w and w != c else None
        pb = conf.get(f'{c}|{bs}') if bs and bs != c else None
        pbw = conf.get(f'{bs}|{w}') if (w and bs and w != bs) else None
        S = 1. if w == c else (pw['S'] if pw and pw['valid'] else None)
        gw, why = red_gate(w) if w else (False, 'no winner')
        if not gw:
            verdict = f'unavailable ({why})'
        elif S is None:
            verdict = 'unavailable (no valid confirmation panel' + (f": {pw.get('reason')}" if pw else '') + ')'
        else:
            verdict = 'pass' if S >= 1 / .6 else 'fail'
        H2[c] = dict(winner=v['winner'], winner_label=w, B_star=v['B_star'], B_star_label=bs, S_screen=v['S_screen'],
                     S_confirmed=S, verdict=verdict,
                     S_BE_confirmed=1. if bs == c else (pb['S'] if pb and pb['valid'] and red_gate(bs)[0] else None),
                     S_scheme_confirmed=1. if w == bs else (pbw['S'] if pbw and pbw['valid'] else None),
                     S_BE_screen=v['S_BE_screen'])
    out['H2'] = H2
    H14 = {}
    for p_ in sel['h14_pairs']:
        c_ = conf.get(f"{p_['fs2']}|{p_['qn']}")
        r = c_['ratio'] if c_ and c_['valid'] else None
        H14[p_['cell']] = dict(fs2=sel['labels'][p_['fs2']], qn=sel['labels'][p_['qn']],
                               ratio_screen=grid['H1']['H14'][p_['cell']]['ratio_screen'], ratio_confirmed=r,
                               verdict=(f"unavailable ({(c_ or {}).get('reason', 'panel missing')})" if r is None else ('pass' if r <= .6 else 'fail')))
    for cell, v in grid['H1']['H14'].items():
        if cell not in H14:
            H14[cell] = dict(fs2=v['fs2'], qn=v['qn'], ratio_screen=None, ratio_confirmed=None, verdict=v['verdict'])
    out['H14'] = H14
    # ---- H3.2 mesh-free path + K5 (K5 must fire for at least one setting, else H3.2 qualifications are blocked, A1.10)
    H32 = {}
    for l_ in blk['mf']:
        c = lab[l_]
        sn, rule, spec = c['setting'], c['rule'], c['spec']
        m = P['mf'].get(l_)
        if m is None or 'invalid' in m:
            H32[l_] = dict(qualified=None, status='unavailable', reason=(m or {}).get('invalid', 'not run'))
            continue
        e_ab = rows_elig(res, f'{sn}|{rule}|mf', spec, cases)
        e_c = ec(m['dist_to_adt'], m['adt_valid'], len(cases))
        a = scores_of(sc, f'{sn}|{rule}|mf|{spec}', cases)
        mm = scores_of(sc, f'{sn}|{rule}|{spec}', cases)
        q = tob_or_within(a, mm)
        elig = bool(e_ab['rows_complete'] and e_ab['Ea'] and e_ab['Eb'] and e_c['Ec'] and cond_ok(m.get('cond'), Rp[sn]))
        k5 = P['k5'].get(l_, {})
        k5_rank = bool(k5.get('rule') == blk.get('k5_rule', 'gl12') and cond_ok(k5.get('cond'), Rp[sn]))
        k5r = tob_or_within(scores_of(sc, f"{sn}|{rule}|mfproj_{k5.get('rule')}|{spec}", cases), mm) if k5_rank else None
        k5f = None if (k5r is None or k5r['b'] is None) else bool(not (k5r['b'] and k5r['c']))
        evid = bool(e_ab['rows_complete'] and e_c.get('dist_worst') is not None and q.get('b') is not None)
        H32[l_] = dict(key=f'{sn}|{rule}|{spec}', eligible=elig, evidence_complete=evid, **e_ab, **e_c, cond=m.get('cond'), **q,
                       measured_pass=bool(evid and elig and q.get('b') and q.get('c')),
                       K5=dict(rule=k5.get('rule'), cond=k5.get('cond'), rank_ok=k5_rank, fails=k5f, detail=k5r))
    # K5 three-state: fires if any measured control fails; does not fire only if every control is resolved and none fails
    k5s = [H32[l_].get('K5', {}).get('fails') for l_ in blk['mf']]       # every expected control; missing ones stay None
    out['K5_fires'] = True if any(x is True for x in k5s) else (False if k5s and all(x is False for x in k5s) else None)
    for v in H32.values():
        if 'K5' not in v:
            continue
        if not v['evidence_complete']:
            v['qualified'], v['status'] = None, 'unavailable (mesh-free rows, distances or scores incomplete)'
        elif not v['measured_pass']:
            v['qualified'], v['status'] = False, 'not qualified'
        elif out['K5_fires'] is True:
            v['qualified'], v['status'] = True, 'qualified'
        elif out['K5_fires'] is False:
            v['qualified'], v['status'] = False, 'blocked (K5 did not fire)'
        else:
            v['qualified'], v['status'] = None, 'unavailable (K5 evidence incomplete)'
    out['H32'] = H32
    # ---- H3.1 profile (indicative) and G3 (only on a valid, complete panel)
    H31 = {}
    need = ['project', 'evolve', 'decode', 'contraction', 'normal+cholesky', 'qn_iteration', 'qn_step_with_refresh', 'dispatch_floor']
    for l_ in blk['profile']:
        t = T.get(f'profile|{l_}')
        c = lab[l_]
        if not t or not t.get('invocations'):
            H31[l_] = dict(missing=True)
            continue
        subs = need + (['adaptive_start'] if c['spec'].split('|')[0] in ('fs', 'fsp') else [])
        pan = {s: panel(t, s, A=f"{c['setting']}|{c['rule']}|{c['spec']}", cases=cases[:tc['cases']], reps=tc['reps'])
               for s in subs}
        okp = all(p['valid'] for p in pan.values())
        comp = {s: p.get('t_B') for s, p in pan.items()}
        fused = float(np.median([.5 * (x['tA1'] + x['tA2']) for x in t['invocations']]))
        parts = comp['project'] + comp['evolve'] + comp['decode'] if okp else None
        H31[l_] = dict(fused_seconds=fused, components_seconds=comp, panel_valid=okp,
                       share={k: comp[k] / fused for k in ('project', 'evolve', 'decode')} if okp else None,
                       share_adaptive_start=(comp['adaptive_start'] / fused) if okp and 'adaptive_start' in comp else None,
                       G3_sum_over_fused=None if parts is None else parts / fused,
                       G3=None if parts is None else bool(abs(parts / fused - 1) <= .15),
                       compile_estimate_seconds={k: x for k, x in res.get('compile_estimate_seconds', {}).items()
                                                 if k.startswith(f"{c['setting']}|{c['rule']}|") and k.endswith(c['spec'])})
    out['H31'] = H31
    # ---- H3.3 batching: recomputed throughput; the 2x bar needs valid B = 1 and B = 16 records (hashes, G4 not fail)
    H33 = {}
    for l_ in blk['batch']['labels']:
        rec = {str(k): v for k, v in P['batch'].get(l_, {}).items()}
        per = {}
        for B in blk['batch']['sizes']:
            r = rec.get(str(B))
            if r is None or 'infeasible' in r:
                per[str(B)] = dict(available=False, reason=(r or {}).get('infeasible', 'missing'))
                continue
            med = float(np.median(r['seconds']))
            g4 = 'exact' if r['G4_max'] <= 1e-8 else ('within tolerance' if r['G4_max'] <= 2.5e-5 else 'fail')
            per[str(B)] = dict(available=True, median_seconds=med, throughput=B / med, hashes_ok=r['hashes_ok'], G4=g4, G4_max=r['G4_max'],
                               K6_max=r.get('K6_max'), K6_fires=None if r.get('K6_max') is None else bool(r['K6_max'] > 2.5e-5),
                               K6plus_max=r['K6plus_max'], K6plus_pass=bool(r['K6plus_max'] <= 2.5e-5),
                               usable=bool(r['hashes_ok'] and g4 != 'fail' and r['K6plus_max'] <= 2.5e-5
                                           and (B == 1 or (r.get('K6_max') is not None and r['K6_max'] > 2.5e-5))))
        b1, b16 = per.get('1', {}), per.get('16', {})
        bar = (None if not (b1.get('usable') and b16.get('usable')) else bool(b16['throughput'] >= 2 * b1['throughput']))
        H33[l_] = dict(per_B=per, throughput_16_over_1=(b16['throughput'] / b1['throughput']) if bar is not None else None,
                       bar_2x=bar,
                       K6_fires_all=tri([v.get('K6_fires') if v.get('available') else None for B, v in per.items() if B != '1']),
                       K6plus_all=tri([v.get('K6plus_pass') if v.get('available') else None for v in per.values()]),
                       latency_B1=b1.get('median_seconds'))
    out['H33'] = H33
    # ---- mesh-path reproduction of the grid accuracy (equal by construction)
    rep = {}
    for l_, c in lab.items():
        key = f"{c['setting']}|{c['rule']}|{c['spec']}"
        a = scores_of(sc, key, cases)
        g = grid['configs'][key]
        if not a['ok']:
            rep[l_] = None
            continue
        ge = dict(zip(g['ranked_ids'], g['e']))
        rep[l_] = float(max(abs(e_ - ge[j]) for j, e_ in zip(a['ranked_ids'], a['e'])))
    out['mesh_reproduction_max_abs_e'] = rep
    # ---- acceptance (A10)
    acc = dict(complete_flag=bool(res.get('complete')), cohort=res['cohort']['seed'] == 923801 and sorted(cases) == list(range(64)),
               K4=bool(sc['controls'].get('K4_fires')),
               confirm_all_valid=all(p['valid'] for p in conf.values()), mf_all_present=all(l_ in P['mf'] for l_ in blk['mf']),
               mf_evidence_valid=all(not str(v.get('status', '')).startswith('unavailable') for v in H32.values()),
               K5_resolved_present=out['K5_fires'] is not None,
               profile_all_present=all(not v.get('missing') for v in H31.values()),
               profile_all_valid=all(v.get('panel_valid') is True for v in H31.values()),
               batch_all_present=all(v['available'] for x in H33.values() for v in x['per_B'].values()),
               conv_all_present=all(v['available'] for v in conv.values()), reduced_adt_all_present=all(v['ok'] is not None for v in red.values()),
               mesh_reproduction=all(v is not None and v <= 1e-12 for v in rep.values()))
    # integrity gates every verdict of the job; coverage items make only their own verdicts unavailable (handled above)
    acc['integrity_ok'] = bool(acc['complete_flag'] and acc['cohort'] and acc['K4'] and acc['mesh_reproduction'])
    acc['coverage_complete'] = bool(all(v for k, v in acc.items() if k.endswith(('_valid', '_present'))))
    acc['accepted_for_claims'] = acc['integrity_ok']
    out['acceptance'] = acc
    if not acc['integrity_ok']:                            # A10: no verdict at all from a job without integrity
        suppress(out, 'p1 integrity failed')
    return out


def union_of(fam):
    """Union over families (A12, asymmetric): minimum over timed matched families; '< 2x' is established by any timed match
    below 2x, '>= 2x' only when every family is resolved (timed match or exhausted ladder)."""
    timed = [(f_, v) for f_, v in fam.items() if v.get('state') == 'matched' and v.get('speedup') is not None]
    best = min(timed, key=lambda z: z[1]['speedup']) if timed else None
    unresolved = [f_ for f_, v in fam.items() if not ((v.get('state') == 'matched' and v.get('speedup') is not None)
                                                       or v.get('state') == 'no ladder match')]
    if best is not None and best[1]['speedup'] < 2:
        r2 = False
    elif best is not None and not unresolved:
        r2 = True
    else:
        r2 = None
    return dict(family=None if best is None else best[0], matched=None if best is None else best[1].get('matched'),
                speedup=None if best is None else best[1]['speedup'],
                speedup_label=None if best is None else ('union' if not unresolved else 'upper bound on the union (families unresolved)'),
                reaches_2x=r2, unresolved_families=unresolved)


def suppress(out, why):
    """Replace every verdict-bearing field of one job's analysis by 'unavailable' (keeps the numbers for diagnosis)."""
    def walk(o):
        if isinstance(o, dict):
            for k in list(o):
                if k in ('verdict', 'status') and isinstance(o[k], str) and not o[k].startswith('unavailable'):
                    o[k] = f'unavailable ({why}; computed {o[k]})'
                elif k in ('qualified', 'measured_pass', 'evidence_complete', 'bar_2x', 'reaches_2x', 'G3', 'K5_fires', 'K6_fires', 'K6plus_pass',
                           'K6_fires_all', 'K6plus_all', 'ok', 'eligible', 'fails', 'usable', 'b', 'c', 'Ec', 'Eb', 'Ea', 'adt_gate',
                           'G4', 'K2_fires', 'still_matched', 'valid'):
                    o[k] = None
                else:
                    walk(o[k])
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(out)


# ------------------------------------------------------------------------------------------------------------- m1
def analyze_m1(rp, sel, grid, fm, f1, scp):
    res = jl(rp / 'result.json')
    sc = jl(scp)
    assert Path(sc['run']).resolve() == rp.resolve() and sc['job_id'] == res['job_id'], 'scores-m1.json is of another run'
    cfg, M_, T = res['config'], res['mesh'], res['timing']
    blk, tc = cfg['mesh'], cfg['timing']
    cases = res['cohort']['cases']
    tcases = cases[:tc['cases']]
    out = dict(job=res['job_id'], gpu=res['gpu'], commit=res['commit'], complete=res.get('complete'), K4=sc['controls'],
               G5=M_['G5'], G5_all=M_['G5_all'], label='mesh-free / scoring-lattice contract (not the full-field query)',
               per_L={}, H41={}, H43={}, panel_failures={})
    f1recs = {v.get('key'): v for v in list(f1['fom']['fd'].values()) + list(f1['fom']['ps'].values())}

    def fom_scores(fk):
        cs = f1recs[fk]['cases']
        rk = [j for j in cases if cs[str(j)].get('eligible')]
        return dict(ok=True, e=[cs[str(j)]['e'] for j in rk], eta=[cs[str(j)]['eps_ref'] for j in rk], ranked_ids=rk)
    roms = {r['label']: r for r in blk['roms']}
    for L in blk['Ls']:
        n = L + 1
        E = M_['per_L'].get(str(L))
        o = out['per_L'][str(L)] = dict(rom={}, fom={}, panels={})
        if E is None:
            o['missing'] = True
            continue
        same_keys = {fv.get('key') for fk, fv in E['fom'].items() if not fv.get('matched_for') and fv.get('key')}
        for l_, r in roms.items():
            er = E['rom'].get(l_)
            if er is None or 'infeasible' in er:
                o['rom'][l_] = dict(feasible=False, eligible=False, detail=(er or {}).get('infeasible', 'not run'))
                continue
            tag = f"{r['setting']}|{r['rule']}|mf|n{n}"
            e_ab = rows_elig(res, tag, r['spec'], cases)
            e_c = ec(er['dist_to_adt'], er['adt_valid'], len(cases))
            tt = bool(np.isfinite(er['adt_tight_tighter']) and er['adt_tight_tighter'] <= BAR_ADT and len(er['adt_tighter_valid']) == min(8, len(cases))
                      and all(er['adt_tighter_valid']))
            s = scores_of(sc, f"{tag}|{r['spec']}", cases)
            el = bool(e_ab['rows_complete'] and e_ab['Ea'] and e_ab['Eb'] and e_c['Ec'] and tt and s['ok'])
            o['rom'][l_] = dict(feasible=True, M=er['tables'].get('M'), **e_ab, **e_c, adt_tight_tighter=er['adt_tight_tighter'], adt_gate=tt,
                                eligible=el, scores_ok=s['ok'], median=s['median'], worst=s['worst'], ranked=len(s['ranked_ids']),
                                e=s['e'], eta=s['eta'], ranked_ids=s['ranked_ids'])
            t = T.get(f'mesh|L{L}|{l_}')
            if t and t.get('invocations'):
                A = f'{l_}|n{n}'
                sp = {b: panel(t, b, A=A, cases=tcases, reps=tc['reps']) for b in t['subjects']}
                valid = all(p['valid'] for p in sp.values())
                o['panels'][l_] = dict(valid=valid, A=t.get('A'),
                                       rom_seconds=float(np.median([.5 * (x['tA1'] + x['tA2']) for x in t['invocations']])) if valid else None,
                                       same_mesh={b: p for b, p in sp.items() if b in same_keys},
                                       cross={b: p for b, p in sp.items() if b not in same_keys})
            else:
                o['panels'][l_] = dict(valid=False, reason='panel missing (see panel_failures)')
        for fk, fv in E['fom'].items():
            ent = dict(key=fv.get('key'), matched_for=fv.get('matched_for'), calibration=(fv.get('calibration') or {}).get('chosen'),
                       calibration_status=(fv.get('calibration') or {}).get('status'), infeasible=fv.get('infeasible'))
            cs = fv.get('cases') or {}
            if cs:
                e = [c['e'] for c in cs.values() if c.get('eligible')]
                ent.update(n_cases=len(cs), converged=all(c.get('converged') for c in cs.values()),
                           median=float(np.median(e)) if e else None, worst=float(np.max(e)) if e else None)
            # absolute FOM warm time at this mesh: median t_B over every valid panel of this mesh that timed it
            tbs = [pl[grp][fv.get('key')]['t_B'] for pl in o['panels'].values() if pl.get('valid')
                   for grp in ('same_mesh', 'cross') if fv.get('key') in pl.get(grp, {})]
            ent['seconds_warm'] = float(np.median(tbs)) if tbs else None
            o['fom'][fk] = ent
        if E.get('panel_failures'):
            out['panel_failures'][str(L)] = E['panel_failures']
    # H4.1 flat ROM cost (valid panels only)
    for l_ in roms:
        ts = {str(L): out['per_L'][str(L)]['panels'].get(l_, {}).get('rom_seconds') for L in blk['Ls']}
        vals = [t for t in ts.values() if t]
        allL = len(vals) == len(blk['Ls'])
        out['H41'][l_] = dict(seconds=ts, max_over_min=(max(vals) / min(vals)) if vals else None,
                              verdict=('unavailable (not every mesh has a valid panel)' if not allL else
                                       ('pass' if max(vals) / min(vals) <= 1.25 else 'fail')))
    # H4.3 (A6) primary: match|<label> panel at n = 65, mesh path
    for l_, r in roms.items():
        mrec = fm['matches'].get(l_, {})
        t = T.get(f'match|{l_}')
        fam = {}
        for f_ in FAMS:
            m = mrec.get(f_, {})
            if m.get('matched'):
                pp = panel(t, m['matched'], A=f'{l_}|meshpath|n65', cases=tcases, reps=tc['reps'])
                fam[f_] = dict(state='matched', matched=m['matched'], boundary=m['boundary'], fom_median=m['median'], fom_worst=m['worst'],
                               speedup=pp['ratio'] if pp['valid'] else None, panel_reason=pp.get('reason'),
                               reaches_2x=None if not pp['valid'] else bool(pp['ratio'] >= 2),
                               label='upper bound (finite FOM ladder)' + ('; boundary (cheapest admissible member)' if m['boundary'] else ''))
            elif m.get('status', '').startswith('no joint'):
                fam[f_] = dict(state='no ladder match', matched=None, speedup=None, reaches_2x=None,
                               label='no FOM in the ladder ties-or-beats the ROM jointly on median and worst')
            else:
                fam[f_] = dict(state='unavailable', matched=None, speedup=None, reaches_2x=None, label=m.get('status', 'not matched'))
        timed = [(f_, v) for f_, v in fam.items() if v['state'] == 'matched' and v['speedup'] is not None]
        uni = min(timed, key=lambda z: z[1]['speedup']) if timed else None
        per_mesh = {}
        for L in blk['Ls']:
            rr = out['per_L'][str(L)]['rom'].get(l_, {})
            pl = out['per_L'][str(L)]['panels'].get(l_, {})
            row = {}
            for f_, v in fam.items():
                if v['state'] != 'matched':
                    row[f_] = dict(state=v['state'])
                    continue
                if not rr.get('eligible'):
                    row[f_] = dict(state='ROM not eligible / not scored at this mesh')
                    continue
                ef, er_, eta, ids = AN.common(fom_scores(v['matched']), rr)
                med, wst = AN.ties_or_beats(ef, er_, eta)
                p = (pl.get('cross') or {}).get(v['matched']) or (pl.get('same_mesh') or {}).get(v['matched']) or {}
                sp_ = p.get('ratio') if (pl.get('valid') and p.get('valid')) else None
                row[f_] = dict(state='matched' if (med and wst) else 'unmatched same-job timing', still_matched=bool(med and wst),
                               speedup=sp_, reaches_2x=None if sp_ is None or not (med and wst) else bool(sp_ >= 2))
            per_mesh[str(L)] = row
        out['H43'][l_] = dict(rom_key=f"{r['setting']}|{r['rule']}|{r['spec']}", families=fam,
                              union=union_of(fam),
                              per_mesh=per_mesh,
                              wording='a finite FOM ladder gives an upper bound on optimal FOM cost, hence an upper bound on the claimed ROM speed-up')
    # H-comp cells (V3-5) at the largest mesh; K2 three-state
    Lmax = str(max(blk['Ls']))
    cells = {}
    for l_ in roms:
        pm = out['H43'][l_]['per_mesh'][Lmax]
        c = {f_: pm[f_] for f_ in FAMS}
        c['union'] = union_of(c)
        cells[l_] = c
    out['Hcomp_largest_mesh'] = dict(L=int(Lmax), cells=cells)
    # K2: fires iff every selected ROM has a matched union speed-up below 2x at the largest mesh; any ROM with a matched
    # union >= 2x -> does not fire; otherwise (missing / undetermined cells) -> undetermined
    st = [c['union']['reaches_2x'] for c in cells.values()]
    out['K2'] = ('does not fire' if any(x is True for x in st) else ('fires' if st and all(x is False for x in st) else 'undetermined'))
    out['K2_fires'] = {'fires': True, 'does not fire': False}.get(out['K2'])       # None when undetermined
    # H4.2 same-mesh diagnostic (valid panels, same-mesh subjects only)
    out['H42'] = {str(L): {l_: {b: p['ratio'] for b, p in pl.get('same_mesh', {}).items()} for l_, pl in out['per_L'][str(L)]['panels'].items()
                           if pl.get('valid')} for L in blk['Ls']}
    acc = dict(complete_flag=bool(res.get('complete')), K4=bool(sc['controls'].get('K4_fires')), G5=bool(M_['G5_all']),
               every_rom_every_mesh=all(out['per_L'][str(L)]['rom'].get(l_, {}).get('feasible') for L in blk['Ls'] for l_ in roms),
               every_rom_scored=all(out['per_L'][str(L)]['rom'].get(l_, {}).get('scores_ok') for L in blk['Ls'] for l_ in roms),
               every_mesh_panel_valid=all(out['per_L'][str(L)]['panels'].get(l_, {}).get('valid') for L in blk['Ls'] for l_ in roms),
               match_panels_valid=all(v['speedup'] is not None for h in out['H43'].values() for v in h['families'].values()
                                      if v['state'] == 'matched'),
               no_panel_failures=not out['panel_failures'],
               primary_unions_resolved=all(not h['union']['unresolved_families'] for h in out['H43'].values()),
               cohort=res['cohort']['seed'] == 923801 and sorted(cases) == list(range(64)))
    acc['integrity_ok'] = bool(acc['complete_flag'] and acc['K4'] and acc['G5'] and acc['cohort'])
    acc['coverage_complete'] = bool(acc['every_rom_every_mesh'] and acc['every_rom_scored'] and acc['every_mesh_panel_valid'] and acc['match_panels_valid']
                                    and acc['no_panel_failures'] and acc['primary_unions_resolved'])
    acc['accepted_for_claims'] = acc['integrity_ok']
    out['acceptance'] = acc
    if not acc['integrity_ok']:
        suppress(out, 'm1 integrity failed')
        out['K2'] = f"undetermined (m1 integrity failed; computed {out['K2']})"
        out['K2_fires'] = None
    return out


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
        print('p1 acceptance', out['p1']['acceptance'])
        for c, v in out['p1']['H2'].items():
            print('H2', c, v['verdict'], v['S_confirmed'], 'screen', v['S_screen'])
        print('H14', {k: v['verdict'] for k, v in out['p1']['H14'].items()})
        print('H32', {k: v.get('status') for k, v in out['p1']['H32'].items()}, 'K5', out['p1']['K5_fires'])
    if 'm1' in out:
        print('m1 acceptance', out['m1']['acceptance'])
        print('H41', {k: v['verdict'] for k, v in out['m1']['H41'].items()})
        print('H43', {k: {f_: x['speedup'] for f_, x in v['families'].items()} for k, v in out['m1']['H43'].items()})
        print('K2', out['m1']['K2'])


if __name__ == '__main__':
    main()
