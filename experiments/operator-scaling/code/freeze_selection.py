"""Freeze every final checkpoint and the per-family selection before the test cohort is opened (PROTOCOL.md section 6).
Validation only; reads no test file.
  freeze_selection.py --ns DIR --out FILE
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
runs, incomplete = {}, []
for d in sorted(Path(a.ns).glob('*/runs/*')):
    if not (d / 'result.json').exists():
        incomplete.append(str(d)); continue
    r = json.loads((d / 'result.json').read_text())
    hist = json.loads((d / 'history.json').read_text())
    base = dict(path=str(d), arm=r['arm'], config=r['config'], rung=r['rung'], train_cases=r['train_cases'], seed=r['seed'],
                steps=r['steps'], stop_reason=r['stop_reason'], converged=r['converged'], job_id=r['job_id'], gpu=r['gpu'],
                real_parameters=r['real_parameters'], final_lr=r['final_lr'], training_seconds=r['training_seconds'],
                continued=r.get('continued_from_last'))
    runs[f'{d.parent.parent.name}/{d.name}'] = dict(base, checkpoint='best.pt', best_sha256=sha256(d / 'best.pt'),
                                                    best_step=r['best_step'], validation=r['best_validation'], budget='final')
    snap = d / 'best_upto_200000.pt'
    if snap.exists():
        import torch
        ck = torch.load(snap, map_location='cpu', weights_only=False)
        runs[f'{d.parent.parent.name}/{d.name}@200k'] = dict(base, checkpoint=snap.name, best_sha256=sha256(snap),
                                                             best_step=ck['step'], validation=ck['validation'], budget='200k')
selection = {}
for fam, letter in (('FNO', 'F'), ('U-Net', 'U')):
    cand = {k: v['validation']['mean'] for k, v in runs.items() if v['arm'][0] == letter}
    best = min(cand, key=cand.get)
    selection[fam] = dict(rule='lowest validation mean over all runs of the family (any size, any rung)', candidates=cand,
                          selected=best)
Path(a.out).write_text(json.dumps(dict(test_cohort_seen=False, runs=runs, incomplete=incomplete, selection=selection), indent=1) + '\n')
print(json.dumps({k: v['selected'] for k, v in selection.items()}, indent=1)); print(len(runs), 'entries;', len(incomplete), 'incomplete')
