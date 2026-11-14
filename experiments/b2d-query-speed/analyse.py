"""b2d-query-speed: per-arm statistics of one job and the PROTOCOL.md section 5 eligibility of every candidate
against the unchanged baseline of the same width in the same job.
    python analyse.py results/<attempt> [...]   ->  results/<attempt>/arms.json and a printed table
"""
import json
import sys
from pathlib import Path

import numpy as np


def arm_stats(r, name, ncase):
    q = sorted((x for x in r['quick'] if x['name'] == name), key=lambda x: x['case'])
    assert [x['case'] for x in q] == list(range(ncase)), (name, len(q))
    err = 100 * np.array([x['same_grid_evolved'] for x in q])
    inv = [x for x in r['invocations'] if x['name'] == name and x['phase'] in ('romA1', 'romA2', 'fomB')]
    t = 1e3 * np.array([x['gpu_seconds'] for x in inv])
    per_case = np.array([np.median([1e3 * x['gpu_seconds'] for x in inv if x['case'] == c]) for c in range(ncase)])
    out = dict(name=name, cases=ncase, finite=bool(np.isfinite(err).all()), median=float(np.median(err)),
               worst=float(err.max()), worst_case=int(np.nanargmax(err)), per_case=err.tolist(),
               invocations=len(t), ms_median=float(np.median(t)), ms_min=float(t.min()), ms_max=float(t.max()),
               ms_case_p05=float(np.quantile(per_case, .05)), ms_case_p95=float(np.quantile(per_case, .95)),
               ms_phase={ph: float(np.median([1e3 * x['gpu_seconds'] for x in inv if x['phase'] == ph]))
                         for ph in sorted({x['phase'] for x in inv})},
               identical_to_quick=bool(all(x['identical_to_quick'] for x in inv)))
    if 'total_iterations' in q[0]:
        out['iterations_median'] = float(np.median([x['total_iterations'] for x in q]))
    return out


def eligibility(c, b):
    """PROTOCOL.md section 5 against baseline b."""
    e, eb = np.array(c['per_case']), np.array(b['per_case'])
    fin = c['finite']
    acc = bool(fin and c['median'] <= 1.02 * b['median'] and c['worst'] <= 1.02 * b['worst']
               and (e <= 1.05 * eb + 1e-3).all())           # 1e-5 relative error = 1e-3 percent
    fast = bool(fin and c['median'] <= 1.10 * b['median'] and c['worst'] <= 1.10 * b['worst'])
    return dict(accurate_ok=acc, fast_ok=fast, median_ratio=c['median'] / b['median'], worst_ratio=c['worst'] / b['worst'],
                max_case_ratio=float(np.max(e / eb)) if fin else None, speedup=b['ms_median'] / c['ms_median'])


def analyse(d):
    r = json.loads((Path(d) / 'result.json').read_text())
    ncase = len(r['physical_cases'])
    setup = {s['arm']: s for s in r['arm_setup']}
    rows = []
    spec = {f"R{a['Rp']}_lin_M{a['M']}_{a['rule']}__qs_{a['tag']}" + ('_graphs' if a['graphs'] else ''): a
            for a in r['config']['arms'] if a.get('impl') == 'qs'}
    rejected = [x['name'] for x in r.get('rejected_nonfinite', [])]
    for name, s in setup.items():
        if name in rejected:
            continue
        if s['family'] != 'rom':
            a = arm_stats(r, name, ncase)
            a.update(width=None, role='fom')
            rows.append(a)
            continue
        a = arm_stats(r, name, ncase)
        a.update(width=s['R_prime'], M=s['M'], rule=s['rule'], role=s.get('role'), impl=s['impl'], graphs=s['graphs'],
                 gtol=s.get('gtol'), qs=spec.get(name, {}).get('qs'), tag=spec.get(name, {}).get('tag'))
        rows.append(a)
    base = {}
    for a in rows:
        if a['role'] == 'baseline' and a['M'] == 4 * a['width'] and a['rule'] == 'lat64' and a['gtol'] == 0.01:
            base[a['width']] = a
    for a in rows:
        if a['width'] in base and a is not base[a['width']]:
            a['vs_baseline'] = eligibility(a, base[a['width']])
    g = r['gates']
    out = dict(attempt=r['config']['attempt'], job_id=r['job_id'], host=r['host'], gpu=r['gpu'], backend=r['backend'],
               commit=r['commit'], complete=r['complete'], elapsed_seconds=r.get('elapsed_seconds'),
               cohort_sha256=r['physical_sha256'], cohort_name=r.get('cohort_name'), cases=ncase,
               drift=dict(passed=g['drift']['passed'], ratios=[min(x['ratio'] for x in g['drift']['rows']),
                                                               max(x['ratio'] for x in g['drift']['rows'])]),
               neighbour_passed=g['neighbour']['passed'], arms=rows, rejected_nonfinite=r.get('rejected_nonfinite', []), diagnostics=r.get('diagnostics'),
               batched=r.get('batched'))
    (Path(d) / 'arms.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


if __name__ == '__main__':
    for d in sys.argv[1:]:
        o = analyse(d)
        print(f"== {o['attempt']} job {o['job_id']} {o['host']} {o['gpu']} backend {o['backend']} cases {o['cases']} "
              f"drift {o['drift']} neighbour {o['neighbour_passed']}")
        for a in sorted(o['arms'], key=lambda a: (a['width'] or 0, a['ms_median'])):
            v = a.get('vs_baseline')
            tail = (f" | x{v['speedup']:.2f} med {v['median_ratio']:.3f} worst {v['worst_ratio']:.3f} case "
                    f"{v['max_case_ratio'] if v['max_case_ratio'] is None else round(v['max_case_ratio'], 3)} "
                    f"{'ACC' if v['accurate_ok'] else ''} {'FAST' if v['fast_ok'] else ''}") if v else ''
            print(f"{a['name'][:62]:62s} {a['median']:8.4f} {a['worst']:8.4f} {a['ms_median']:7.2f} "
                  f"[{a['ms_case_p05']:.2f},{a['ms_case_p95']:.2f}]{tail}")
