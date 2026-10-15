"""jcp-speed3d routes 1 and 2 analysis (DESIGN §5, H1.1-H1.4, H2; A1.4-A1.7, A1.10, A2.2, A2.6). Pure post-processing of
pulled grid jobs (result.json, W_grid.npz) and their scores (score3.py); CPU only.

Per configuration (setting|rule|spec): E-a validity (all cases), E-b certified non-stationary steps <= 1 % pooled,
E-c lattice distance to ADT tight <= 2.5e-4 on every case (ADT all steps verified; the (scheme, step) is unselectable if
the deployed-rule ADT tight/tighter check exceeds 2.5e-5); E0 median / worst (and first-order, PROVISIONAL); screening
cost = median paired ratio to C_old (A = C_old in every job's panel; panel validity required).
Selections (validation cohort only): H2 winner per comparator (cheapest eligible configuration whose median and worst
tie-or-beat the comparator's: median(e - eta) <= median(e_C + eta), same for max; 3 % cost ties -> lower median, then
lower worst), B* (cheapest eligible BE configuration that ties-or-beats), H1.2-H1.4 verdicts, H1.1 labels, controls.

    python analyze.py --runs runs/g1/archive/output runs/g2/archive/output ... [--out checks/analysis-grid.json]
"""
from __future__ import annotations

import offline  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAR, BAR_ADT, POOL = 2.5e-4, 2.5e-5, 0.01
COMPARATORS = {'C_old': 'M2_512|gl32|fs|BE|0.01|K1', 'C_W768': 'W768|lat32768|fs|BE|0.01|K1',
               'C_W1024': 'W1024|lat65536|fs|BE|0.01|K1'}
LABEL = ('E0 R_ST (conditional Richardson estimate under the asymptotic assumption, not a rigorous bound; the E0 '
         'cross-check rule was amended after data, E0 A10). Study prospectively specified conditional on Step 2 exploration.')


def ec_pass(dmax):
    """The E-c classifier (shared by eligibility and K9): every case's lattice distance (max over evolved times) <= BAR."""
    dmax = np.asarray(dmax, float)
    return bool(np.isfinite(dmax).all() and dmax.max() <= BAR)


def t1_ok(t):
    """T1 gate of one panel: median ratio of the 'C_old twice' subject in [1.8, 2.2]."""
    r = [x['ratio'] for x in t.get('invocations', []) if x['B'].startswith('T1')]
    return bool(r) and 1.8 <= float(np.median(r)) <= 2.2


def spec_parts(spec):
    p = spec.split('|')
    return p[0], p[1], float(p[2]), p[3]


def ties_or_beats(ea, ec, eta):
    """Step 2 rule (interval overlap with endpoint propagation) on median and worst."""
    ea, ec, eta = np.asarray(ea), np.asarray(ec), np.asarray(eta)
    med = bool(np.median(ea - eta) <= np.median(ec + eta))
    wst = bool(np.max(ea - eta) <= np.max(ec + eta))
    return med, wst


def load_job(rp):
    rp = Path(rp)
    res = json.loads((rp / 'result.json').read_text())
    sc = json.loads((HERE / 'checks' / f"scores-{res['config']['attempt']}.json").read_text())
    return res, sc


def configs_of(res, sc):
    """{config key: record} for every non-ADT, non-mutation grid configuration of one job."""
    cases = res['cohort']['cases']
    # job-level acceptance (audit code-1 #9): complete job, full validation cohort, K4 and the distance re-check pass
    job_ok = bool(res.get('complete') and res['cohort']['seed'] == 923801 and len(cases) == 64
                  and sc['controls'].get('K4_fires') and sc['gates'].get('dist_ok'))
    rows = {}
    for r in res['rows']:
        rows.setdefault(f"{r['tag']}|{r['spec']}", []).append(r)
    panel = {}
    for lab, t in res['timing'].items():
        if not lab.startswith('screen|'):
            continue
        for x in t['invocations']:
            panel.setdefault(x['B'], []).append(x)
        panel_valid = bool(t['valid'] and t1_ok(t))
        for s in t['subjects']:
            panel.setdefault(f'__valid__{s}', panel_valid)
    out = {}
    for sn, sv in res['settings'].items():
        dep_rule = next(iter(sv['rules']))
        adt_ok = {k: bool(v['worst'] <= BAR_ADT and v['tighter_valid'])            # A1.4: deployed-rule check, setting-wide
                  for k, v in sv['rules'][dep_rule].get('adt_tight_tighter', {}).items()}
        for rule, rr in sv['rules'].items():
            for spec, dist in rr['dist_to_adt'].items():
                if len(spec.split('|')) > 4:             # mutation controls: handled separately
                    continue
                key = f'{sn}|{rule}|{spec}'
                rs = sorted(rows[key], key=lambda r: cases.index(r['case']))
                kind, sch, dt, _ = spec_parts(spec)
                ak = f'ad|LSPG|{sch}|{dt:.8g}|tight'
                adt_valid = all(rr['adt_valid'][ak])
                dmax = np.asarray(dist)[:, 1:].max(1)
                nst = sum(r['nonstationary'] for r in rs)
                nstep = sum(r['steps_n'] for r in rs)
                sco = sc['scores'][key]
                rk = [j for j in cases if sco[str(j)]['eligible']]          # E0: ineligible case-kinds scored, not ranked
                e = np.array([sco[str(j)]['e'] for j in rk])
                eta = np.array([sco[str(j)]['eps_ref'] for j in rk])
                e1 = np.array([sco[str(j)]['first_order']['e'] for j in rk]) if 'first_order' in sco[str(cases[0])] else None
                inv = panel.get(key, [])
                ratio = float(np.median([x['ratio'] for x in inv])) if inv else None
                tB = float(np.median([x['tB'] for x in inv])) if inv else None
                tA = float(np.median([.5 * (x['tA1'] + x['tA2']) for x in inv])) if inv else None
                adt_gate = adt_ok.get(ak)
                rec = dict(setting=sn, rule=rule, deployed=rule == dep_rule, spec=spec, kind=kind, scheme=sch, dt=dt, solver=spec.split('|')[0] +
                           ('-' + spec.split('|')[3] if kind in ('fs', 'fsp', 'qn') else ''),
                           Ea=all(r['valid'] for r in rs), Eb_frac=nst / max(nstep, 1), Eb=nst <= POOL * nstep,
                           Ec=bool(adt_valid and ec_pass(dmax)), dist_worst=float(dmax.max()), ranked_cases=len(rk),
                           unranked_cases=len(cases) - len(rk),
                           adt_valid=adt_valid, adt_gate=adt_gate, median=float(np.median(e)), worst=float(e.max()),
                           e=e.tolist(), eta=eta.tolist(),
                           median_first_order=None if e1 is None else float(np.median(e1)),
                           worst_first_order=None if e1 is None else float(e1.max()),
                           ratio=ratio, t_B=tB, t_A=tA, panel_valid=panel.get(f'__valid__{key}'),
                           job=res.get('job_id'), gpu=res.get('gpu'), commit=res.get('commit'))
                rec['eligible'] = bool(rec['Ea'] and rec['Eb'] and rec['Ec'])
                rec['selectable'] = bool(rec['eligible'] and ratio is not None and rec['panel_valid'] and adt_gate is True
                                         and job_ok and len(rk) == len(cases))
                out[key] = rec
    return out


def controls(res, sc):
    """K1-K3 on one job (deployed rules)."""
    cases = res['cohort']['cases']
    rows = {}
    for r in res['rows']:
        rows.setdefault(f"{r['tag']}|{r['spec']}", []).append(r)
    out = {}
    for sn, sv in res['settings'].items():
        rule = next(iter(sv['rules']))
        rr = sv['rules'][rule]
        o = {}

        def elig(spec):
            rs = rows[f'{sn}|{rule}|{spec}']
            d = np.asarray(rr['dist_to_adt'][spec])[:, 1:].max(1)
            nst, nstep = sum(r['nonstationary'] for r in rs), sum(r['steps_n'] for r in rs)
            return dict(Ea=all(r['valid'] for r in rs), Eb=nst <= POOL * nstep, Ec=bool(d.max() <= BAR),
                        dist_cases_above=int((d > BAR).sum()), nonstationary=nst, steps=nstep)
        for name, spec in (('K1', 'fs|CN|0.025|K1'), ('K2', 'qn|CN|0.025|k1|q0'), ('K3_CN', 'qn|CN|0.025|k1|noJuT'),
                           ('K3_BE', 'qn|BE|0.01|k1|noJuT')):
            if spec in rr['dist_to_adt']:
                e = elig(spec)
                e['eligible'] = bool(e['Ea'] and e['Eb'] and e['Ec'])
                o[name] = e
        o['K1_fires'] = o.get('K1', {}).get('eligible') is False
        o['K2_fires'] = o.get('K2', {}).get('eligible') is False
        o['K3_fires'] = bool(o.get('K3_CN', {}).get('dist_cases_above', 0) >= 1 or o.get('K3_BE', {}).get('dist_cases_above', 0) >= 1)
        out[sn] = o
    return out


def k9(res, Wz, glat_fn):
    """A2.6 distance-classifier control on M2_512's ADT BE dt0 output, case 0 (needs the job holding M2_512)."""
    sn, rule = 'M2_512', 'gl32'
    key = f'{sn}|{rule}|ad|LSPG|BE|0.01|tight'
    if key not in Wz.files:
        return None
    import c3common as C3
    ids = [int(x) for x in Wz[f'cases|{key}']]
    W = Wz[key][ids.index(0)]
    G = glat_fn(sn)
    tab = C3.C.table(923801, 64)
    n0 = float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, 0))))
    v = np.random.default_rng(20261014).standard_normal(W.shape)
    v[0] = 0
    d1 = np.linalg.norm(v @ G, axis=1)[1:].max() / n0
    out = {}
    for f in (0.5, 2.0):
        s = f * BAR / d1
        Wp = W + s * v
        d = float((np.linalg.norm((Wp - W) @ G, axis=1) / n0)[1:].max())
        out[str(f)] = dict(constructed=d, within=ec_pass([d]))
    out['fires'] = bool(out['0.5']['within'] and not out['2.0']['within'] and abs(out['0.5']['constructed'] - .5 * BAR) < 1e-12
                        and abs(out['2.0']['constructed'] - 2 * BAR) < 1e-12)
    return out


def select(C, comp_key):
    c = C[comp_key]
    cand = [k for k, v in C.items() if v['selectable']]
    match = []
    for k in cand:
        med, wst = ties_or_beats(C[k]['e'], c['e'], C[k]['eta'])
        # eta per case is the same reference uncertainty for both arms (same case): endpoint propagation uses +/- eta
        if med and wst:
            match.append(k)

    def pick(keys):
        if not keys:
            return None
        keys = sorted(keys, key=lambda k: C[k]['ratio'])
        best = C[keys[0]]['ratio']
        tied = [k for k in keys if C[k]['ratio'] <= 1.03 * best]
        return sorted(tied, key=lambda k: (C[k]['median'], C[k]['worst']))[0]
    w = pick(match)
    b = pick([k for k in match if C[k]['scheme'] == 'BE'])
    return dict(comparator=comp_key, comparator_eligible=c['eligible'], comparator_median=c['median'],
                comparator_worst=c['worst'], comparator_ratio=c['ratio'], n_matching=len(match),
                winner=w, B_star=b,
                S_screen=None if w is None else c['ratio'] / C[w]['ratio'],
                S_BE_screen=None if b is None else c['ratio'] / C[b]['ratio'],
                status='selected' if w else 'unavailable')


def h1(C):
    out = dict(H12={}, H13={}, H14={})
    sets = sorted({v['setting'] for v in C.values()})
    for sn in sets:
        dep = [dict(v, eligible=v['selectable'], elig_raw=v['eligible']) for v in C.values() if v['setting'] == sn and v['deployed']]   # gates applied
        get = lambda solver, sch, dt: next((v for v in dep if v['solver'] == solver and v['scheme'] == sch
                                            and abs(v['dt'] - dt) < 1e-9), None)
        for sch in ('CN', 'BDF2'):
            x = get('fsp-K1', sch, .025)
            out['H12'][f'{sn}|{sch}'] = None if x is None else x['eligible']
            for dt in (.0125, .05 / 3, .025):
                for k in ('k1', 'k4', 'kinf'):
                    x = get(f'qn-{k}', sch, dt)
                    out['H13'][f'{sn}|{sch}|{dt:.8g}|{k}'] = None if x is None else x['eligible']
                fs2 = get('fs-K2', sch, dt)
                qs = [get(f'qn-{k}', sch, dt) for k in ('k1', 'k4', 'kinf')]
                qs = [q for q in qs if q is not None and q['eligible'] and q['ratio'] is not None]
                if not qs:
                    v = 'unavailable'
                    best = None
                else:
                    best = min(qs, key=lambda q: q['ratio'])
                    if fs2 is None or not fs2['elig_raw']:
                        v = 'rescue'
                    else:
                        v = 'pass (screening)' if best['ratio'] <= .6 * fs2['ratio'] else 'fail (screening)'
                out['H14'][f'{sn}|{sch}|{dt:.8g}'] = dict(verdict=v, qn=None if best is None else f"{best['setting']}|{best['rule']}|{best['spec']}",
                                                         fs2=None if fs2 is None else f"{fs2['setting']}|{fs2['rule']}|{fs2['spec']}",
                                                         ratio_screen=None if best is None or fs2 is None or fs2['ratio'] is None
                                                         else best['ratio'] / fs2['ratio'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', default=str(HERE / 'checks' / 'analysis-grid.json'))
    a = ap.parse_args()
    C, ctl, h11, jobs, k9r = {}, {}, {}, [], None
    import score3 as SC
    for rp in a.runs:
        res, sc = load_job(rp)
        jobs.append(dict(attempt=res['config']['attempt'], job=res.get('job_id'), gpu=res.get('gpu'), commit=res.get('commit'),
                         complete=res.get('complete'), K4=sc['controls'], dist_gate=sc['gates']))
        C.update(configs_of(res, sc))
        ctl.update(controls(res, sc))
        for sn, sv in res['settings'].items():
            rule = next(iter(sv['rules']))
            h11[sn] = {str(j): dict(label=v.get('label'), kind=v.get('kind'), detail=v) for j, v in sv['rules'][rule].get('h11_replay', {}).items()}
        Wz = np.load(Path(rp) / 'W_grid.npz')
        if k9r is None:
            k9r = k9(res, Wz, SC.glat)
    # controls block the claims they protect (A1.10): K1/K2 -> every H1 eligibility verdict of the setting; K3 -> QN configs
    for k, v in C.items():
        c = ctl.get(v['setting'], {})
        if not (c.get('K1_fires') and c.get('K2_fires')):
            v['selectable'] = False
            v['blocked_by'] = 'K1/K2 did not fire'
        if v['kind'] == 'qn' and not c.get('K3_fires'):
            v['selectable'] = False
            v['blocked_by'] = 'K3 did not fire'
    sel = {name: select(C, key) for name, key in COMPARATORS.items() if key in C}
    out = dict(label=LABEL, jobs=jobs, configs=C, controls=ctl, K9=k9r, H11=h11, H1=h1(C), H2=sel)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    for k, v in sel.items():
        print(k, v['status'], v['winner'], v['S_screen'], 'B*', v['B_star'], v['S_BE_screen'])


if __name__ == '__main__':
    main()
