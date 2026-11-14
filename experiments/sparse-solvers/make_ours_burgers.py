"""Collect the paper's Burgers numbers (our model's worst-case / median error and query time, and the paper's own
matrix-free full-order solver) from the archived result files of the paper's runs into ours_burgers.json.
Read-only on the other worktrees.   python make_ours_burgers.py <worktrees dir>
"""
import json
import sys
from pathlib import Path

W = Path(sys.argv[1])
out = dict(note='errors in percent of the initial-field norm (max over evolved output times), worst and median over the '
                'paper cohort (2D: 64 test cases; 3D: 32 held-out cases); ms = median online query time',
           burgers2d={}, burgers3d={})
B2 = W / '2026-09-25-burgers2d-test/experiments/burgers2d-test/checks'
# the paper's headline table uses the selection roles 'accurate' and 'fast' up to 1024^2, and 'accurate' and 'extra'
# (width 128) as its fast setting at 2048^2 and 4096^2
FAST2 = {256: 'fast', 512: 'fast', 1024: 'fast', 2048: 'extra', 4096: 'extra'}
for N in (256, 512, 1024, 2048, 4096):
    d = json.loads((B2 / f't{N}-summary.json').read_text())
    row = dict(source=f'burgers2d-test checks/t{N}-summary.json', gpu=d['gpu'], job_id=d['job_id'], fom={})
    for role, key in (('accurate', 'accurate'), ('fast', FAST2[N])):
        v = d['selection']['roles'][key]
        row[role] = dict(arm=v['name'], worst_percent=v['worst_evolved_percent'], median_percent=v['median_evolved_percent'],
                         ms=v['median_gpu_ms'])
    for k, v in d['table'].items():
        if v['family'] == 'rom':
            continue
        elif v['median_gpu_ms'] is not None:
            row['fom'][k] = dict(worst_percent=v['worst_evolved_percent'], median_percent=v['median_evolved_percent'],
                                 ms=v['median_gpu_ms'], dt=v['dt'], ntol=v['ntol'])
    out['burgers2d'][str(N)] = row
B3 = W / '2026-09-23-burgers3d-retry/experiments/burgers3d-retry/runs'
for L, h in ((64, 'ho65'), (128, 'ho129')):
    d = json.loads((B3 / h / 'archive/code/output/result.json').read_text())
    row = dict(source=f'burgers3d-retry runs/{h}/archive/code/output/result.json', gpu=d['gpu'], job_id=d['job_id'], fom={})
    for role, arm in (('accurate', 'span_R512_fsc_dt0.005'), ('fast', 'span_R192_fsc_dt0.01')):
        a = d['arms'][arm]
        row[role] = dict(arm=arm, worst_percent=100 * a['worst_evolved'], median_percent=100 * a['median_worst_evolved'],
                         ms=d['timing']['summary'][arm]['median_ms'])
    for k, v in d['timing']['summary'].items():
        if k.startswith('fom_'):
            row['fom'][k] = dict(ms=v['median_ms'])
    out['burgers3d'][str(L)] = row
Path(__file__).with_name('ours_burgers.json').write_text(json.dumps(out, indent=1))
for p in ('burgers2d', 'burgers3d'):
    for k, r in out[p].items():
        print(p, k, r['gpu'], {x: (round(r[x]['worst_percent'], 4), round(r[x]['ms'], 2)) for x in ('accurate', 'fast')})
