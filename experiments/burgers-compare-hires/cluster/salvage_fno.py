"""Add p2048b's in-job fno-large checkpoint to operators-2048.json (DESIGN A6). Every value is read from the files.

p2048b (4207177) trained fno-large at 2048^2 inside its H200 allocation (A4); the job was killed for exceeding its
240 GB host-memory request during epoch 19, after 19 completed epochs and 2897 s of the 3000 s budget, before
train.py's closing re-score and result.json. best.pt is the training loop's own best-validation checkpoint (saved at
epoch 18). The embedded epoch, the history and the SHA256 are checked here; the stop reason is recorded as the kill.
"""
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
import torch

ROOT = Path(__file__).resolve().parents[3]
LANE = ROOT / 'experiments/burgers-compare-hires'
d = LANE / 'runs/p2048b/archive/out/fno-large'
ck = d / 'best.pt'
sha = hashlib.sha256(ck.read_bytes()).hexdigest()
hist = json.loads((d / 'history.json').read_text())
c = torch.load(ck, map_location='cpu', weights_only=False)
best = min(hist, key=lambda h: h['validation']['mean_case_max'])
assert c['epoch'] == best['epoch'] and abs(c['best'] - best['validation']['mean_case_max']) < 1e-15, (c['epoch'], best['epoch'])
params = sum(v.numel() * (2 if v.is_complex() else 1) for k, v in c['model'].items()
             if not k.endswith('_metadata') and hasattr(v, 'numel'))
rec_path = LANE / 'operators-2048.json'
rec = json.loads(rec_path.read_text())
rec['operators'] = [o for o in rec['operators'] if o['name'] != 'fno-large']
rec['failed'] = [f for f in rec['failed'] if f['name'] != 'fno-large']
rec['operators'].insert(0, dict(
    name='fno-large', family='fno',
    role="the 256^2 panel's validation-selected fno configuration, retrained at 2048^2 (inside p2048b's H200 allocation)",
    local_path=str(ck.relative_to(ROOT)), sha256=sha, source_job='4207177 (p2048b, in-job training, DESIGN A4/A6)',
    trained_at='2048^2', epochs=len(hist), best_epoch=int(c['epoch']),
    stop_reason='killed (host memory) after %d completed epochs, %.0f s of the 3000 s budget' % (len(hist), hist[-1]['wall_seconds']),
    wall_budget_seconds=3000, training_seconds=hist[-1]['wall_seconds'], micro_batch_final=None,
    validation_mean_case_max=best['validation']['mean_case_max'],
    validation_worst_case_max=best['validation']['worst_case_max'], real_parameter_count=int(params),
    salvaged=True, history_sha256=hashlib.sha256((d / 'history.json').read_bytes()).hexdigest()))
rec_path.write_text(json.dumps(rec, indent=1) + '\n')
print('fno-large', sha, 'epochs', len(hist), 'best', c['epoch'], best['validation']['mean_case_max'], 'params', params)
