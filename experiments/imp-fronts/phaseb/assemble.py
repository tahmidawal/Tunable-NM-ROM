"""imp-fronts Phase B: assemble the verdict.py input (arm b vs K1; K2 / K3 floors for the "passes against all three
incumbent retrains" report; K0 floors reported unmatched), run the B0 decision, and score E9 on K1. CPU, no JAX.

    python phaseb/assemble.py [--arm-b-label b --arm-b-selection <json>]     -> checks/phaseb_verdicts.json
    python phaseb/assemble.py --e9-only                                      -> checks/phaseb/e9.json

Inputs (all lane-relative): checks/phaseb/floors_<K0..K3|b>.json (floors_b.py), checks/phaseb/score_<K1|b>.json
(score_b.py), runs/refT/pulled/REF-LEDGER.json (eligibility = ledger 'accepted'), phaseb/e9_frozen.json.
Arms b and c are NOT RUN until both their floors and score files exist (arm c: floors_c / score_c, arch 'sine_even',
selection JSON for the m* route, Amendments 19-22); arm a is NOT RUN (schedule S3). E10 arms are reported beside the
verdict: E10-s from checks/phaseb/score_e10s.json (score_b.py --label e10s on the K1 bank) with its control evidence verified by
rom/evidence.verify_e10; E10-q = STOP by E10-DESIGN §10 only on validated, complete, hash-bound every-step evidence. K1's selection is gauss160 (25 600 nodes, jcp-robust E2); K2 / K3 enter with floors only (their ROM
scoring is deferred; their selections are not used by arm b's routes) -> their ROM routes are UNAVAILABLE.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
sys.path.insert(0, str(HERE))
import verdict as V  # noqa: E402

CHK = LANE / 'checks' / 'phaseb'
LEDGER = LANE / 'runs' / 'refT' / 'pulled' / 'REF-LEDGER.json'
K1_SELECTION = dict(certified=True, rule='gauss160', nodes=25600, source='jcp-robust E2 (K1)')


def _f(x):
    return math.inf if x == 'inf' else (None if x is None else float(x))


def load(name):
    p = CHK / name
    return json.loads(p.read_text()) if p.exists() else None


def eligible():
    led = json.loads(LEDGER.read_text())
    el = {}
    for r in led['cases']:
        if r['outcome'] == 'accepted':
            el.setdefault(r['cohort'], []).append(r['case'])
    return {c: sorted(v) for c, v in el.items()}, led


def bank_entry(floors, score, selection, trained=True):
    b = dict(trained=trained, rom_available=score is not None, selection=selection or dict(certified=False))
    if floors:
        b['floors'] = {c: {j: dict(f384=v.get('f384'), f512=v.get('f512')) for j, v in d['per_case'].items()
                           if v.get('eligible')} for c, d in floors['cohorts'].items()}
    if score:
        pol = {}
        for r in score['cases']:
            pol.setdefault(r['cohort'], {})[str(r['j'])] = dict(e=_f(r['e']), valid=r['valid'], qualified=r['qualified'],
                                                                tau=_f(r['tau4096']), terminal=r['terminal'])
        b['policy'] = pol
    return b


def eps_map(score):
    out = {}
    for r in (score or {}).get('cases', []):
        out.setdefault(r['cohort'], {})[str(r['j'])] = r['eps_ref']
    return out


def bank_ids(floors, score):
    ids = {}
    for name, d in (('floors', floors), ('score', score)):
        if d:
            ids[name] = (d['bank']['ckpt_sha256'], d['bank']['rotation_sha256'])
    if len(set(ids.values())) > 1:
        raise SystemExit(f'floors and score were computed with different banks: {ids}')
    return next(iter(ids.values()), None)


def assemble(arm_label='b', arm_selection=None, arm_c_label='c', arm_c_selection=None, e10s_controls=None):
    el, led = eligible()
    k1f, k1s = load('floors_K1.json'), load('score_K1.json')
    if not (k1f and k1s):
        raise SystemExit('K1 floors and score are required')
    k1id = bank_ids(k1f, k1s)
    controls = dict(K1=bank_entry(k1f, k1s, K1_SELECTION))
    for k in ('K2', 'K3'):
        f = load(f'floors_{k}.json')
        if f:
            controls[k] = bank_entry(f, None, None)
    arms = {}
    for arm, label, selp in (('b', arm_label, arm_selection), ('c', arm_c_label, arm_c_selection)):
        af, as_ = load(f'floors_{label}.json'), load(f'score_{label}.json')
        if not (af and as_):
            continue                                   # NOT RUN until both files exist
        aid = bank_ids(af, as_)
        if aid == k1id:
            raise SystemExit(f'arm {arm} bank is identical to K1')
        sel = json.loads(Path(selp).read_text()) if selp else dict(certified=False)
        arms[arm] = bank_entry(af, as_, {k: sel[k] for k in ('certified', 'rule', 'nodes') if k in sel})
        arms[arm]['_arch'] = af['bank'].get('arch')
    archs = {k: v.pop('_arch') for k, v in arms.items()}
    if 'c' in archs and archs['c'] != 'sine_even':
        raise SystemExit(f"arm c bank arch is {archs['c']!r}, not 'sine_even'")
    if 'b' in archs and archs['b'] == 'sine_even':
        raise SystemExit('arm b bank has the arm-(c) architecture')
    inp = dict(arms=arms, controls=controls, eligible={c: el.get(c, []) for c in ('FI', 'FN0s', 'FC1', 'FC2')},
               eps_ref=eps_map(k1s))
    out = V.verdict(inp)
    k0 = load('floors_K0.json')
    rep = dict(verdict=out, schedule=('S3C (arms b and c vs K1; arm a NOT RUN)' if out['arms']['c']['outcome'] != 'NOT RUN' else 'S3 (arm b + K1); arms a, c NOT RUN (Amendments 11-12)'),
               eligible=inp['eligible'], ledger_counts=led['counts'],
               k1=dict(selection=K1_SELECTION, summary=k1s['summary'], pooled=k1s['pooled_nonstationary'],
                       floors={c: d['summary'] for c, d in k1f['cohorts'].items()}),
               floors_unmatched_K0={c: d['summary'] for c, d in k0['cohorts'].items()} if k0 else None,
               floors_K2_K3={k: {c: d['summary'] for c, d in load(f'floors_{k}.json')['cohorts'].items()}
                             for k in ('K2', 'K3') if load(f'floors_{k}.json')},
               arms_present=sorted(arms), arch=archs, e10=e10(k1s, inp['eligible'], e10s_controls), note='eps_ref = conditional Richardson estimate (E0-style ST references)')
    return rep


E10_EVIDENCE = LANE / 'runs' / 'SM-e' / 'E10-EVIDENCE.json'
POL_K1 = LANE / 'runs' / 'rom' / 'POL-K1-out' / 'output' / 'fpolicy_pol1' / 'result.json'
RHO_TOL = 1e-6


def verify_e10s_evidence(path=None):
    """Audit codex-code-C-2 row 12: E10-s control evidence only through the hash-bound verifier rom/evidence.verify_e10."""
    p = Path(path) if path else E10_EVIDENCE
    if not p.exists():
        return False, f'{p} missing'
    sys.path.insert(0, str(LANE / 'rom'))
    import evidence as EV
    return EV.verify_e10(p)


def e10q_check(q, result_path=None):
    """Audit codex-code-C-2 row 13: the every-step rho evidence decides E10-q only if it is validated (passed, max relative
    error and tolerance <= 1e-6), bound to K1's policy result (sha256), complete (every '|cert|' record of that result
    recomputed with all states finite; validation covers every visited prod run) and self-consistent."""
    if not q:
        return False, 'e10q_every_step.json missing'
    rp = Path(result_path) if result_path else POL_K1
    v = q.get('validation') or {}
    if v.get('passed') is not True:
        return False, 'validation.passed is not true'
    try:
        if not (float(v['max_rel_err']) <= RHO_TOL and float(v['tolerance']) <= RHO_TOL):
            return False, f"validation relative error {v.get('max_rel_err')} / tolerance {v.get('tolerance')} > {RHO_TOL}"
    except (KeyError, TypeError, ValueError):
        return False, 'validation error fields missing'
    if not rp.exists():
        return False, 'K1 policy result missing'
    import hashlib
    if q.get('result_sha256') != hashlib.sha256(rp.read_bytes()).hexdigest():
        return False, 'evidence not bound to the K1 policy result (sha256)'
    res = json.loads(rp.read_text())
    cert = {k for k in res['records'] if '|cert|' in k}
    runs = {r['run'] for r in q.get('runs', [])}
    if runs != cert or q.get('n_runs') != len(cert) or len(q.get('runs', [])) != len(cert):
        return False, f'run coverage incomplete ({len(runs & cert)}/{len(cert)})'
    if any(r.get('finite_states') != r.get('states') for r in q['runs']):
        return False, 'non-finite states in the recompute'
    prods = {h['prod'] for c in res['policy'] for h in c['history'] if h.get('prod')}
    if {x['run'] for x in v.get('per_run', [])} != prods or v.get('n') != len(prods):
        return False, 'validation does not cover every visited prod run'
    if any(not (float(x['max_rel_err']) <= RHO_TOL) for x in v['per_run']):
        return False, 'a per-run validation error exceeds the tolerance'
    sm = q.get('summary') or {}
    if sm.get('states_over_bar') != sum(int(r['n_over']) for r in q['runs']) or \
            sm.get('runs_over_bar') != sum(int(r['n_over']) > 0 for r in q['runs']):
        return False, 'summary inconsistent with per-run records'
    return True, 'ok'


def e10(k1s, elig, controls_path=None):
    """E10 arms (E10-DESIGN §8-§10). E10-s: precedence NOT RUN -> INCONCLUSIVE (< 6 eligible FC1, K1 FC1 diagnostic
    median non-finite, must-fail control / prerequisite unavailable) -> STOP (not qualified on every eligible FC1 case, or
    E10-s FC1 median non-finite or > K1 median / 2) -> CONTINUE; step-converged FC1 count reported descriptively.
    E10-q: STOP without a job if the every-step evidence shows no accepted state over the rho bar (§10), else DEFERRED."""
    out = {}
    s = load('score_e10s.json')
    fc1 = [str(j) for j in elig.get('FC1', [])]
    k1e = [_f(r['e']) for r in k1s['cases'] if r['cohort'] == 'FC1' and str(r['j']) in fc1]
    k1med = float(np.median(k1e)) if k1e else math.inf
    if s is None:
        out['E10-s'] = dict(outcome='NOT RUN', why='no score_e10s.json')
    else:
        if (s['bank']['ckpt_sha256'], s['bank']['rotation_sha256']) != (k1s['bank']['ckpt_sha256'], k1s['bank']['rotation_sha256']):
            raise SystemExit('E10-s was not scored on the K1 bank')
        rows = {str(r['j']): r for r in s['cases'] if r['cohort'] == 'FC1'}
        es = [_f(rows[c]['e']) if c in rows else math.inf for c in fc1]
        med = float(np.median(es)) if es else math.inf
        qual = all(c in rows and rows[c]['qualified'] and rows[c]['valid'] and math.isfinite(_f(rows[c]['e'])) for c in fc1)
        ctl_ok, ctl_why = verify_e10s_evidence(controls_path)
        conv = sum(1 for c in fc1 if c in rows and rows[c]['terminal'] == 'CONVERGED')
        base = dict(fc1_median=med, k1_fc1_diagnostic_median=k1med, bar=k1med / 2, n_fc1=len(fc1),
                    qualified_all=qual, fc1_step_converged=conv, terminals={c: rows.get(c, {}).get('terminal') for c in fc1})
        if len(fc1) < 6 or not math.isfinite(k1med):
            out['E10-s'] = dict(base, outcome='INCONCLUSIVE', why='population or K1 median prerequisite')
        elif not ctl_ok:
            out['E10-s'] = dict(base, outcome='INCONCLUSIVE', why=f'must-fail control evidence not verified: {ctl_why}')
        elif not qual or not math.isfinite(med) or med > k1med / 2:
            out['E10-s'] = dict(base, outcome='STOP', why='not qualified on every eligible FC1 case' if not qual
                                else 'FC1 median not <= K1 diagnostic median / 2')
        else:
            out['E10-s'] = dict(base, outcome='CONTINUE')
    q = load('e10q_every_step.json')
    q_ok, q_why = e10q_check(q)
    if not q_ok:
        out['E10-q'] = dict(outcome='INCONCLUSIVE', why=f'every-step evidence not validated: {q_why}')
    elif q['summary'].get('states_over_bar') == 0 and q['summary'].get('runs_over_bar') == 0:
        out['E10-q'] = dict(outcome='STOP', why='E10-DESIGN §10: rho <= 0.116 on every accepted state of every K1 prod '
                            'run, so E10-q executes exactly K1 solves (FC1 Q2-qualified, not step-converged); no job, 0 GPU-h',
                            evidence=q['summary'])
    else:
        out['E10-q'] = dict(outcome='DEFERRED', why='every-step evidence missing or a state exceeds the rho bar')
    return out


def e9():
    import e9 as E9
    s = load('score_K1.json')
    pol = {r['case']: dict(e=_f(r['e']), valid=r['valid'], eligible=r['eligible']) for r in s['cases']}
    return E9.score(pol)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return 'inf' if x > 0 else 'nan'
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm-b-label', default='b')
    ap.add_argument('--arm-b-selection')
    ap.add_argument('--arm-c-label', default='c')
    ap.add_argument('--arm-c-selection')
    ap.add_argument('--e10s-controls', help='hash-bound E10 evidence (rom/evidence.py build-e10); default runs/SM-e/E10-EVIDENCE.json')
    ap.add_argument('--e9-only', action='store_true')
    a = ap.parse_args()
    r9 = e9()
    CHK.mkdir(parents=True, exist_ok=True)
    (CHK / 'e9.json').write_text(json.dumps(_clean(r9), indent=1) + '\n')
    print('E9', json.dumps({k: r9[k] for k in ('n', 'tau_s', 'flagged_fraction', 'input_flagged_fraction',
                                              'failure_fallback_fraction')}))
    for t, v in r9['targets'].items():
        print(f"  target {t}: bad {v['n_bad']} good {v['n_good']} AUC {v['auc']} det {v['detection']} "
              f"(lo {v['detection_cp95_lower']}) fa {v['false_alarm']} (hi {v['false_alarm_cp95_upper']}) {v['label']}")
    if a.e9_only:
        return
    rep = assemble(a.arm_b_label, a.arm_b_selection, a.arm_c_label, a.arm_c_selection, a.e10s_controls)
    (LANE / 'checks' / 'phaseb_verdicts.json').write_text(json.dumps(_clean(rep), indent=1) + '\n')
    for k, v in rep['verdict']['arms'].items():
        print(k, v['outcome'], {r: x['status'] for r, x in v['routes'].items()})
    for k, v in rep['e10'].items():
        print(k, v['outcome'], v.get('why', ''))


if __name__ == '__main__':
    main()
