"""Pick up operator checkpoints from a training job that runs CONCURRENTLY with the panel job (DESIGN A3).

    python cluster/late_ops.py <committed operators-L.json> <training job dir> <arm,arm,...> <max wait s> <out json>

Waits (polling once a minute, at most <max wait> seconds) until the training job's worker.json lists every named
arm, then for each arm that trained: re-hashes best.pt against the SHA256 the TRAINING job recorded in its own
result.json, copies it to opckpt/<arm>.pt, and writes <out json> = the committed record plus these arms (with that
SHA256, epochs and stop reason). An arm that failed or never arrived is recorded under `failed` and gets no row.
The panel's audit then checks every timed checkpoint against this record, and the local audit re-checks it after
collection against the collected training archive.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

FAMILY = {'fno-large': 'fno', 'unet-refine': 'unet', 'tsol-refine': 'transolver', 'don-small': 'deeponet'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


rec = json.loads(Path(sys.argv[1]).read_text())
tdir = Path(sys.argv[2])
arms = sys.argv[3].split(',')
deadline = time.time() + float(sys.argv[4])
out = Path(sys.argv[5])
done = {}
while time.time() < deadline:
    w = tdir / 'out/worker.json'
    if w.exists():
        done = {a['arm']: a for a in json.loads(w.read_text())['arms']}
        if all(a in done for a in arms):
            break
    time.sleep(60)
rec['late_operators'] = dict(training_dir=str(tdir), requested=arms, waited_until=time.time())
L = rec['mesh']
for arm in arms:
    rec['failed'] = [f for f in rec['failed'] if f['name'] != arm]
    a = done.get(arm)
    res = tdir / 'out' / arm / 'result.json'
    if a is None or a['exit_code'] != 0 or not res.exists():
        rec['failed'].append(dict(name=arm, family=FAMILY[arm], failed=True,
                                  exit_code=None if a is None else a['exit_code'],
                                  note='late training arm did not arrive in time' if a is None else 'late training arm failed'))
        continue
    r = json.loads(res.read_text())
    ck = tdir / 'out' / arm / 'best.pt'
    got = sha(ck)
    assert got == r['best_checkpoint_sha256'], (arm, got)
    Path('opckpt').mkdir(exist_ok=True)
    shutil.copyfile(ck, f'opckpt/{arm}.pt')
    assert sha(f'opckpt/{arm}.pt') == got
    rec['operators'] = [o for o in rec['operators'] if o['name'] != arm] + [dict(
        name=arm, family=FAMILY[arm], role=f"the 256^2 panel's validation-selected {FAMILY[arm]} configuration, retrained at {L}^2",
        local_path=f'(late) {ck}', sha256=got, source_job=a['job_id'], trained_at=f'{L}^2', epochs=r['epochs_completed'],
        best_epoch=r['best_epoch'], stop_reason=r['stop_reason'], wall_budget_seconds=r['wall_budget_seconds'],
        micro_batch_final=r.get('micro_batch_final'), validation_mean_case_max=r['validation']['mean_case_max'],
        validation_worst_case_max=r['validation']['worst_case_max'], real_parameter_count=r['real_parameter_count'])]
out.write_text(json.dumps(rec, indent=1) + '\n')
print('LATE OPERATORS', [(o['name'], o.get('epochs')) for o in rec['operators']], 'failed', [f['name'] for f in rec['failed']])
