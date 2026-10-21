"""S1 acceptance (DESIGN A1.6, A2.6, A3.5): collects controls 2-7 from smoke.json, runs controls 1 (CPU preflight exit
code from the job log), 8 (decision logic on fixtures derived from the smoke's measured records) and 9 (replication
fixture vs jcp-p1-scaling), writes ACCEPT.json and exits non-zero unless every control passed.

    python mmaccept.py <outdir> <joblog>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import cost as C  # noqa: E402

REQUIRED = ('2c_restriction', '2ab_prolongation', '3_mapping', '5_blocked', '4_q1_mutation', '6_case_identity',
            '7_timing')


def main():
    out, joblog = Path(sys.argv[1]), Path(sys.argv[2])
    acc = {}

    def part(name, fn):
        try:
            acc.update(fn())
        except Exception as ex:  # noqa: BLE001  (a failed prerequisite is a failed control, recorded)
            acc[name] = dict(passed=False, error=f'{type(ex).__name__}: {ex}')

    sm = {}
    try:
        sm = json.loads((out / 'smoke' / 'smoke.json').read_text())
    except Exception as ex:  # noqa: BLE001
        acc['smoke.json'] = dict(passed=False, error=str(ex))
    ctl = sm.get('controls', {})

    def p_smoke():
        r = {k: dict(passed=bool(ctl.get(k, {}).get('passed')), source='smoke.json') for k in REQUIRED}
        fx = ctl.get('4_q1_mutation', {}).get('fixture_q1', {})
        r['4_fixture_qualified'] = dict(passed=bool(fx.get('q1')), source='smoke.json')
        r['pilot_shape'] = dict(passed=bool(sm.get('pilot', {}).get('G_shape_ok')), G_shape=sm.get('pilot', {}).get('G_shape'))
        r['smoke_complete'] = dict(passed=bool(sm.get('complete')))
        return r

    def p_cpu():
        txt = joblog.read_text(errors='replace') if joblog.exists() else ''
        m = re.findall(r'cpu_preflight_rc=(\d+)', txt)
        return {'1_cpu_preflight': dict(passed=bool(m) and m[-1] == '42', rc=m[-1] if m else None)}

    def p_repl():
        p1 = json.loads((HERE.parent / 'jcp-p1-scaling' / 'checks' / 'analysis.json').read_text())['configurations']
        rq = json.loads((out / 'repl' / 'oursq.json').read_text())['configs']['R128|BE']['per_case']
        mine = {f"{x['cohort']}|{x['case']}": x['e'] for x in rq}
        ref128, ref256 = p1['1024|R128|BE']['e0_per_case'], p1['1024|R256|BE']['e0_per_case']
        d128 = max(abs(mine[k] - ref128[k]) for k in mine)
        d256 = min(abs(mine[k] - ref256[k]) for k in mine)
        return {'9_replication': dict(passed=bool(len(mine) == 6 and d128 <= 1e-8 and d256 > 1e-8),
                                      max_diff_R128=d128, min_diff_vs_R256=d256)}

    def p_logic():
        # control 8: fixture DERIVED from the smoke's measured competitor record (errors, query seconds, Q1 state);
        # ours' errors are the competitor errors x 1.5 and x 2 (two X), so an accuracy-admissible witness exists
        good = sm['fixture_records']
        cerr = {f"{x['cohort']}|{x['case']}": x['e'] for x in good['rows']}
        eps = {f"{x['cohort']}|{x['case']}": x['eps_ref'] for x in good['rows']}
        tq = float(np.mean([x['seconds'] for x in good['rows']]))
        src = sm['src1024']
        D = dict(eps=eps, ours_offline={'A_dep': 1000.}, ours={}, comp={},
                 art={'src': float(src['snap_seconds'] + src['pod_seconds'] + src['states_seconds'])})
        for X, f in (('X1.5', 1.5), ('X2', 2.0)):
            D['ours'][X] = dict(status={1024: 'QUALIFIED'}, err={1024: {c: f * e for c, e in cerr.items()}},
                                t={1024: 2 * tq}, setup={1024: 10.}, compile={1024: 10.})
        rec = dict(state='MEASURED-Q1' if good['q1']['q1'] else 'MEASURED-FAIL', err=cerr, t=tq, compile=5., armdata=1.,
                   needs=['src'], rollout_ok=None)
        for arm in C.CP_ARMS:
            D['comp'][arm] = {1024: {'r128|k4|CNR': dict(rec)}}
        return {f'8:{k}': v for k, v in C.selftest_strict(D).items()}

    part('smoke_controls', p_smoke)
    part('1_cpu_preflight', p_cpu)
    part('9_replication', p_repl)
    part('8_decision_logic', p_logic)
    allok = bool(acc) and all(v['passed'] for v in acc.values())
    (out / 'ACCEPT.json').write_text(json.dumps(dict(all_passed=allok, controls=acc), indent=1, default=str))
    for k, v in acc.items():
        print('ACCEPT', k, 'PASS' if v['passed'] else 'FAIL', {a: b for a, b in v.items() if a != 'passed'})
    print('ACCEPT ALL', allok)
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
