"""Freeze the selection before the test cohort is opened (PROTOCOL amendment A6). Validation only; reads no test file.

  freeze_selection.py --ns DIR --out FILE
Final run of each (arm, rung, seed) = its continuation (c*/runs) if one exists, else the first-round run (tr*/runs).
"""
import argparse, hashlib, json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


a = argparse.ArgumentParser(); a.add_argument('--ns', required=True); a.add_argument('--out', required=True)
a = a.parse_args()
ns = Path(a.ns)
final = {}
for d in sorted(ns.glob('tr*/runs/*')) + sorted(ns.glob('c*/runs/*')):      # continuations come last and replace
    if (d / 'result.json').exists() and (d / 'best.pt').exists():
        final[d.name] = d
runs = {}
for name, d in sorted(final.items()):
    r = json.loads((d / 'result.json').read_text())
    runs[f'{d.parent.parent.name}/{name}'] = dict(
        path=str(d), best_sha256=sha256(d / 'best.pt'), arm=r['arm'], rung=r['rung'], seed=r['seed'], steps=r['steps'],
        best_step=r['best_step'], stop_reason=r['stop_reason'], converged=r['converged'], job_id=r['job_id'], gpu=r['gpu'],
        continuation_of=(r.get('continuation') or {}).get('source'), validation=r['best_validation'])
selection = {}
for fam, letter in (('FNO', 'F'), ('U-Net', 'U')):
    cand = {v['arm']: v['validation']['mean'] for v in runs.values() if v['arm'][0] == letter and v['rung'] == 4608 and v['seed'] == 0}
    arm = min(cand, key=cand.get)
    selection[fam] = dict(rule='lowest validation mean at the 4,608 rung, seed 0', candidates=cand, selected_arm=arm,
                          seeds_available=sorted(v['seed'] for v in runs.values() if v['arm'] == arm and v['rung'] == 4608))
Path(a.out).write_text(json.dumps(dict(test_cohort_seen=False, runs=runs, selection=selection), indent=1) + '\n')
print(json.dumps(selection, indent=1)); print(len(runs), 'runs')
