"""b2d-query-speed: apply PROTOCOL.md section 5 to the dev38 selection jobs and write frozen.json.
    python freeze.py results/sello results/selhi   ->  frozen.json
Reads only arms.json of the named dev38 jobs (written by analyse.py). No test64/fresh64 number exists when this runs.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT_WIDTHS = [96, 128, 192, 256, 384, 512]
EQ_WIDTHS = [128, 144, 160, 176, 192]
EQ_LEVEL = 0.195          # percent; FNO test64 median (burgers-ops-matched)

jobs = [json.loads((Path(d) / 'arms.json').read_text()) for d in sys.argv[1:]]
assert all(j['complete'] and j['backend'] == 'gpu' and j['cases'] == 38 and 'dev38' in j['cohort_name'] for j in jobs)
widths = {}
for j in jobs:
    for a in j['arms']:
        if a['width'] is None:
            continue
        w = widths.setdefault(a['width'], dict(job_id=j['job_id'], baseline=None, cands=[]))
        if a['role'] == 'baseline':
            w['baseline'] = a
        elif 'vs_baseline' in a:
            w['cands'].append(a)
    for x in j.get('rejected_nonfinite', []):
        print('rejected (non-finite):', x['name'])


def brief(a):
    v = a.get('vs_baseline', {})
    return dict(name=a['name'], tag=a.get('tag'), qs=a.get('qs'), graphs=a['graphs'], dev_median=a['median'],
                dev_worst=a['worst'], dev_ms=a['ms_median'], dev_speedup=v.get('speedup'),
                dev_median_ratio=v.get('median_ratio'), dev_worst_ratio=v.get('worst_ratio'),
                dev_max_case_ratio=v.get('max_case_ratio'))


out = dict(rule='PROTOCOL.md section 5', source_jobs=[j['job_id'] for j in jobs], cohort='dev38', widths={})
for Rp in sorted(widths):
    w = widths[Rp]
    assert w['baseline'] is not None and all(a['qs'] is not None for a in w['cands'] if a['impl'] == 'qs')
    w['cands'] = [a for a in w['cands'] if a['impl'] == 'qs' and a['M'] == 4 * Rp and a['rule'] == 'lat64']
    acc = sorted((a for a in w['cands'] if a['vs_baseline']['accurate_ok']), key=lambda a: a['ms_median'])
    fast = sorted((a for a in w['cands'] if a['vs_baseline']['fast_ok']), key=lambda a: a['ms_median'])
    A = acc[0] if acc else None
    ref_ms = A['ms_median'] if A else w['baseline']['ms_median']
    F = fast[0] if fast and fast[0]['ms_median'] <= 0.9 * ref_ms and fast[0] is not A else None
    out['widths'][str(Rp)] = dict(job_id=w['job_id'], baseline=brief(w['baseline']), accurate=A and brief(A),
                                  fast=F and brief(F), candidates=len(w['cands']),
                                  accurate_eligible=len(acc), fast_eligible=len(fast))
eq = [Rp for Rp in EQ_WIDTHS if str(Rp) in out['widths'] and out['widths'][str(Rp)]['accurate']
      and out['widths'][str(Rp)]['accurate']['dev_median'] <= EQ_LEVEL]
out['equal_error_width'] = min(eq) if eq else None
out['equal_error_level_percent'] = EQ_LEVEL
out['final_widths'] = sorted((set(REPORT_WIDTHS) | ({out['equal_error_width']} if eq else set())) & {int(k) for k in out['widths']})
(HERE / 'frozen.json').write_text(json.dumps(out, indent=1) + '\n')
for Rp, w in out['widths'].items():
    b = w['baseline']
    print(Rp, f"base {b['dev_median']:.4f}/{b['dev_worst']:.3f}/{b['dev_ms']:.2f}",
          *(f"| {k} {w[k]['tag']} {w[k]['dev_median']:.4f}/{w[k]['dev_worst']:.3f}/{w[k]['dev_ms']:.2f} x{w[k]['dev_speedup']:.2f}"
            if w[k] else f'| {k} none' for k in ('accurate', 'fast')))
print('equal-error width', out['equal_error_width'], 'final widths', out['final_widths'])
