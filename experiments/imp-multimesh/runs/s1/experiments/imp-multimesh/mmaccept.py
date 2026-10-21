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
    sm = json.loads((out / 'smoke' / 'smoke.json').read_text())
    acc = {}
    ctl = sm.get('controls', {})
    for k in REQUIRED:
        acc[k] = dict(passed=bool(ctl.get(k, {}).get('passed')), source='smoke.json')
    # fixture of control 4 must itself be Q1-qualified for the mutation to be informative
    fx = ctl.get('4_q1_mutation', {}).get('fixture_q1', {})
    acc['4_fixture_qualified'] = dict(passed=bool(fx.get('q1')), source='smoke.json')
    # control 1
    txt = joblog.read_text(errors='replace') if joblog.exists() else ''
    m = re.findall(r'cpu_preflight_rc=(\d+)', txt)
    acc['1_cpu_preflight'] = dict(passed=bool(m) and m[-1] == '42', rc=m[-1] if m else None)
    # control 9
    p1 = json.loads((HERE.parent / 'jcp-p1-scaling' / 'checks' / 'analysis.json').read_text())['configurations']
    rq = json.loads((out / 'repl' / 'oursq.json').read_text())['configs']['R128|BE']['per_case']
    mine = {f"{x['cohort']}|{x['case']}": x['e'] for x in rq}
    ref128 = p1['1024|R128|BE']['e0_per_case']
    ref256 = p1['1024|R256|BE']['e0_per_case']
    d128 = max(abs(mine[k] - ref128[k]) for k in mine)
    d256 = min(abs(mine[k] - ref256[k]) for k in mine)
    acc['9_replication'] = dict(passed=bool(len(mine) == 6 and d128 <= 1e-8 and d256 > 1e-8), max_diff_R128=d128,
                                min_diff_vs_R256=d256)
    # control 8: fixture from measured records (smoke fixture configuration at 1024; ours from the replication run)
    good = sm['fixture_records']
    cerr = {f"{x['cohort']}|{x['case']}": x['e'] for x in good['rows']}
    eps = {f"{x['cohort']}|{x['case']}": x['eps_ref'] for x in good['rows']}
    tq = float(np.mean([x['seconds'] for x in good['rows']]))
    D = dict(eps=eps, ours_offline={'A_dep': float(sm['src1024']['snap_seconds'] * 10)}, art={'src': float(
        sm['src1024']['snap_seconds'] + sm['src1024']['pod_seconds'] + sm['src1024']['states_seconds'])}, ours={}, comp={})
    D['ours']['R128|BE'] = dict(status={1024: 'QUALIFIED'}, err={1024: mine}, t={1024: 0.02}, setup={1024: 10.},
                                compile={1024: 10.})
    D['ours']['R256|BE(p1 errors, fixture)'] = dict(status={1024: 'QUALIFIED'}, err={1024: dict(ref256)},
                                                   t={1024: 0.05}, setup={1024: 10.}, compile={1024: 10.})
    D['ours']['R256|BE(p1 errors, fixture)']['err'][1024] = {k: ref256[k] for k in mine}
    rec = dict(state='MEASURED-Q1' if good['q1']['q1'] else 'MEASURED-FAIL', err=cerr, t=tq, compile=5., armdata=1.,
               needs=['src'], rollout_ok=None)
    for arm in C.CP_ARMS:
        D['comp'][arm] = {1024: {'r128|k4|CNR': dict(rec)}}
    D['ours']['R128|BE']['status'] = {1024: 'QUALIFIED'}
    st = C.selftest_strict(D)
    acc.update({f'8:{k}': v for k, v in st.items()})
    allok = all(v['passed'] for v in acc.values())
    (out / 'ACCEPT.json').write_text(json.dumps(dict(all_passed=allok, controls=acc), indent=1, default=str))
    for k, v in acc.items():
        print('ACCEPT', k, 'PASS' if v['passed'] else 'FAIL', {a: b for a, b in v.items() if a != 'passed'})
    print('ACCEPT ALL', allok)
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
