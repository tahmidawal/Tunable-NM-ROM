"""poisson-lshape3d-test: generate every number of this lane from the audited result JSONs.

    python make_tables.py    -> reports/summary.json (+ prints its sha256)

For each (problem, mesh) it applies the FROZEN Table-1 rows (frozen-test-{lshape,cube}.json) to
  * the TEST job of this lane (runs/<attempt>/archive/output), and
  * the DEVELOPMENT job of the parent lane (../poisson-bank-knob-3d/runs/<attempt>/archive/output),
with the same code, the same scope (GPU query = fused_device_seconds; complete query also reported) and the same FOM
rule (fastest tested CG with worst error <= the accurate arm's worst error, same job). The development values are
checked against the numbers printed in the paper (PAPER below, copied from the lane brief; a mismatch is reported,
never hidden). The selection rule re-applied on test is descriptive only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'poisson-bank-knob-3d'
sys.path.insert(0, str(PARENT))
import pbk3_core as P3                                   # noqa: E402
import make_tables as PT                                 # noqa: E402  (parent helpers: subjects, fastest_cg, ...)

SCOPE, OTHER = 'fused_device_seconds', 'total_seconds'
ROM = PT.ROM
FROZEN = dict(lshape=json.loads((HERE / 'frozen-test-lshape.json').read_text()),
              cube=json.loads((HERE / 'frozen-test-cube.json').read_text()))
DEV = {('lshape', 256): 'l256b', ('lshape', 512): 'l512', ('lshape', 1024): 'l1024', ('lshape', 2048): 'l2048b',
       ('cube', 128): 'c128', ('cube', 256): 'c256'}
# the paper's printed development values (lane brief): accurate %, accurate x, fast %, fast x, FOM %
PAPER = {('lshape', 256): ('3.04', '47.1', '5.73', '53.4', '2.40'),
         ('lshape', 512): ('3.03', '39.4', '5.72', '45.5', '1.51'),
         ('lshape', 1024): ('3.03', '48.9', '5.72', '57.9', '1.05'),
         ('lshape', 2048): ('3.03', '84.2', '5.72', '104', '0.76'),
         ('cube', 128): ('0.14', '13.7', '4.50', '36.3', '0.075'),
         ('cube', 256): ('0.14', '23.2', '4.50', '83.5', '0.049')}
FAST_CRITERION_PCT = 5.0                                 # the paper's "fast = worst error < 5 %" wording


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def matches_printed(value, printed):
    """True iff `value` rounds to the printed string (same number of significant decimals, or 3 s.f. for >= 100)."""
    if '.' in printed:
        d = len(printed.split('.')[1])
        return abs(value - float(printed)) <= 0.5 * 10 ** (-d) + 1e-12
    return round(value) == int(printed) or abs(value - float(printed)) <= 0.5 + 1e-12


def per_case(R, name):
    errs = {}
    for x in R['invocations'] + R['slow_invocations']:
        if x['name'] == name:
            errs.setdefault(x['case'], []).append(x['same_grid_error'])
    return {c: 100 * max(v) for c, v in sorted(errs.items())}


def row(R, S, fz, extra=()):
    acc = S[fz['accurate']]
    fom = PT.fastest_cg(S, acc['worst_pct'])
    arms = {}
    for role, name in [('accurate', fz['accurate']), ('fast', fz['fast'])] + [('also:' + n, n) for n in extra]:
        s = S[name]
        own = PT.fastest_cg(S, s['worst_pct'])
        arms[role] = dict(name=name, worst_pct=s['worst_pct'], median_pct=s['median_pct'], ms=s['ms'],
                          speedup=(fom['ms'] / s['ms']) if fom else None,
                          own_fom=own['name'] if own else None,
                          own_speedup=(own['ms'] / s['ms']) if own else None,
                          stationary=s.get('stationary'), cases=s['cases'], invocations=s['invocations'],
                          below_fast_criterion=bool(s['worst_pct'] < FAST_CRITERION_PCT))
    return dict(arms=arms, fom=None if fom is None else dict(name=fom['name'], tolerance=fom['tolerance'],
                                                               worst_pct=fom['worst_pct'], median_pct=fom['median_pct'],
                                                               ms=fom['ms'], iterations_median=fom['iterations_median']))


def evaluate(base, prob, n, role):
    R = json.loads((base / 'result.json').read_text())
    A = json.loads((base / 'audit.json').read_text())
    assert R['problem'] == prob and R['intervals'] == n, (base, R['problem'], R['intervals'])
    fz = FROZEN[prob]
    extra = tuple(fz.get('also_reported', ()))
    gates = dict(R['gates'])
    og = {sc: P3.order_gate(R['neighbour'], R['invocations'], [a['name'] for a in R['arms']], sc)
          for sc in (SCOPE, OTHER)}
    gates_scope = dict(gates, neighbour=og[SCOPE]['passed'])
    gates_other = dict(gates, neighbour=og[OTHER]['passed'])
    S, S2 = PT.subjects(R, SCOPE), PT.subjects(R, OTHER)
    cohort = R['cohort']
    e = dict(role=role, problem=prob, intervals=n, job_id=R['job_id'], gpu=R['gpu'], commit=R['commit'],
             cohort_role=cohort.get('role', 'development'), cases=len(cohort['parameters']),
             cohort_seed=cohort.get('seed'), cohort_parameters_sha256=cohort.get('parameters_sha256', cohort.get('sha256')),
             result_sha256=sha_file(base / 'result.json'), audit_sha256=sha_file(base / 'audit.json'),
             audit=A['verdict'], audit_summary=A['summary'], driver_gates=R['gates'],
             order_gate={sc: {k: og[sc][k] for k in ('passed', 'pooled_paired_ratio', 'median_after_over_main',
                                                    'max_arm_paired_ratio', 'pairs', 'min_pairs_per_arm')}
                         for sc in og},
             usable=bool(A['verdict'] == 'PASS' and all(gates_scope.values())),
             usable_other_scope=bool(A['verdict'] == 'PASS' and all(gates_other.values())),
             parity=R['parity'], floors_pct={k: 100 * v['worst'] for k, v in R['floors'].items()},
             elapsed_seconds=R.get('elapsed_seconds'), device_memory=R.get('device_memory'),
             row=row(R, S, fz, extra), row_other_scope=row(R, S2, fz, extra),
             per_case_pct={name: per_case(R, name) for name in (fz['accurate'], fz['fast']) + extra})
    # descriptive only: the parent's selection rule and the 5 % fast wording re-applied on this cohort
    roms = [s for s in S.values() if s['family'] in ROM]
    acc_d, fast_d, bar = PT.select(S, R)
    under5 = [s for s in roms if s['worst_pct'] < FAST_CRITERION_PCT]
    e['descriptive_rule_on_this_cohort'] = dict(
        accurate=acc_d['name'], accurate_worst_pct=acc_d['worst_pct'], fast_le_orig_q0=fast_d['name'],
        orig_q0_worst_pct=bar, cheapest_under_5pct=min(under5, key=lambda s: s['ms'])['name'] if under5 else None,
        note='descriptive only; never used to choose a Table-1 setting')
    e['all_arms'] = PT.per_arm(S)
    e['all_arms_other_scope'] = PT.per_arm(S2)
    e['cg'] = sorted((s for s in S.values() if s['family'] == 'cg'), key=lambda s: -s['tolerance'])
    e['cg_other_scope'] = sorted((s for s in S2.values() if s['family'] == 'cg'), key=lambda s: -s['tolerance'])
    return e


def main():
    attempts = json.loads((HERE / 'attempts.json').read_text())
    tests = {}
    for att, role in attempts.items():
        if role != 'test':
            continue
        base = HERE / 'runs' / att / 'archive' / 'output'
        if not (base / 'audit.json').exists():
            print('not collected yet:', att)
            continue
        R = json.loads((base / 'result.json').read_text())
        key = (R['problem'], R['intervals'])
        assert key not in tests, ('two test attempts for one mesh', key, tests.get(key), att)
        tests[key] = att
    out = dict(generated_by='experiments/poisson-lshape3d-test/make_tables.py', scope=SCOPE, other_scope=OTHER,
               frozen=FROZEN, fast_criterion_pct=FAST_CRITERION_PCT, rows=[])
    for key in sorted(DEV, key=lambda k: (k[0] != 'lshape', k[1])):
        prob, n = key
        dev = evaluate(PARENT / 'runs' / DEV[key] / 'archive' / 'output', prob, n, 'development')
        dev['attempt'] = DEV[key]
        a, f = dev['row']['arms']['accurate'], dev['row']['arms']['fast']
        vals = (a['worst_pct'], a['speedup'], f['worst_pct'], f['speedup'], dev['row']['fom']['worst_pct'])
        dev['paper_check'] = dict(printed=PAPER[key], regenerated=vals,
                                  matches=[matches_printed(v, p) for v, p in zip(vals, PAPER[key])])
        entry = dict(problem=prob, intervals=n, development=dev, test=None)
        if key in tests:
            t = evaluate(HERE / 'runs' / tests[key] / 'archive' / 'output', prob, n, 'test')
            t['attempt'] = tests[key]
            assert t['cohort_role'] == 'test', t['cohort_role']
            dw = {r: dev['row']['arms'][r]['worst_pct'] for r in dev['row']['arms']}
            t['cases_above_development_worst'] = {
                r: int(sum(v > dw[r] for v in t['per_case_pct'][t['row']['arms'][r]['name']].values()))
                for r in t['row']['arms']}
            entry['test'] = t
        out['rows'].append(entry)
    (HERE / 'reports').mkdir(exist_ok=True)
    p = HERE / 'reports' / 'summary.json'
    p.write_text(json.dumps(out, indent=1) + '\n')
    print('wrote', p, sha_file(p))
    for r in out['rows']:
        d = r['development']
        print(r['problem'], r['intervals'], 'dev paper check', d['paper_check']['matches'],
              'test' if r['test'] else 'no test yet')


if __name__ == '__main__':
    main()
