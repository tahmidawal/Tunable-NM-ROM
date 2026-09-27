"""Tiny generated fixture for exercising the complete training entrypoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
for split, count, code in [('train', 4, 100), ('validation', 2, 200)]:
    folder = args.out / split
    folder.mkdir()
    rows = []
    for i in range(count):
        coord = np.arange(17, dtype=np.float64) / 16
        field = np.sin(np.pi * coord[:, None]) * np.sin(np.pi * coord[None, :])
        field *= 1 + (code + i) / 1000
        field[[0,-1],:] = 0
        field[:,[0,-1]] = 0
        cid = f'smoke-{split}-{i}'
        path = folder / f'{cid}.npz'
        np.savez(path, input=field[None], target=(.2*field)[None,None],
                 parameters=np.empty(0, dtype=np.float64), times=np.array([0.], dtype=np.float64))
        rows.append(dict(case_id=cid, split=split, seed=code+i, mesh=16,
                         path=path.name, sha256=dataset.sha256(path)))
    (folder/'index.json').write_text(json.dumps(dict(pde='poisson', complete=True, records=rows,
        kind='manufactured implementation fixture; not PDE-family training evidence')))
config = dict(width=8, modes=6, layers=2, epochs=2, patience=2, batch_size=2,
              learning_rate=.001, weight_decay=.0001, seed=20260914)
(args.out/'config.json').write_text(json.dumps(config))
subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')), '--smoke',
    '--train-index', str(args.out/'train/index.json'), '--validation-index', str(args.out/'validation/index.json'),
    '--config', str(args.out/'config.json'), '--out', str(args.out/'run')], check=True)
result = json.loads((args.out/'run/result.json').read_text())
assert result['epochs_completed'] == 2 and result['complete']
print('training_entrypoint_smoke=passed', flush=True)
