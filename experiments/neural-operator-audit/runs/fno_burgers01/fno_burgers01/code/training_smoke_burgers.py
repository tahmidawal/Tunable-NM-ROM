"""Tiny generated Burgers-contract fixture for exercising the training entrypoint.

The fields are manufactured, not solutions of Burgers' equation. This checks the
four-key schema, the five evolved output channels, the exactly returned supplied
initial state and the complete training entrypoint; it is not physical evidence.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import dataset

TIMES = np.array([0., .05, .1, .15, .2, .25], dtype=np.float64)

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
for split, count, code in [('train', 4, 300), ('validation', 2, 400)]:
    folder = args.out / split
    folder.mkdir()
    rows = []
    for i in range(count):
        coord = np.arange(17, dtype=np.float64) / 16
        field = np.sin(np.pi * coord[:, None]) * np.sin(np.pi * coord[None, :])
        field = field * (1 + (code + i) / 1000)
        field[[0, -1], :] = 0
        field[:, [0, -1]] = 0
        nu = .02 + (code + i) / 10000
        trajectory = np.stack([field * np.exp(-2 * np.pi ** 2 * nu * t) for t in TIMES])[:, None]
        trajectory[0] = field[None]
        cid = f'smoke-burgers-{split}-{i}'
        path = folder / f'{cid}.npz'
        np.savez(path, input=field[None], target=trajectory,
                 parameters=np.array([nu], dtype=np.float64), times=TIMES)
        rows.append(dict(case_id=cid, split=split, case_index=i, seed=code + i, mesh=16,
                         path=path.name, sha256=dataset.sha256(path)))
    (folder / 'index.json').write_text(json.dumps(dict(pde='burgers', complete=True, records=rows,
        kind='manufactured implementation fixture; not PDE-family training evidence')))
config = dict(width=8, modes=6, layers=2, epochs=2, patience=2, batch_size=2,
              learning_rate=.001, weight_decay=.0001, seed=20260914)
(args.out / 'config.json').write_text(json.dumps(config))
subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')), '--smoke',
    '--train-index', str(args.out / 'train/index.json'),
    '--validation-index', str(args.out / 'validation/index.json'),
    '--config', str(args.out / 'config.json'), '--out', str(args.out / 'run')], check=True)
result = json.loads((args.out / 'run/result.json').read_text())
assert result['epochs_completed'] == 2 and result['complete'] and result['pde'] == 'burgers'
predictions = sorted(args.out.glob('run/*.prediction.npz'))
assert len(predictions) == 2
for path, row in zip(predictions, json.loads((args.out / 'validation/index.json').read_text())['records']):
    with np.load(path) as saved:
        prediction = saved['prediction']
    with np.load(args.out / 'validation' / row['path']) as truth:
        supplied = truth['input']
    assert prediction.shape == (6, 1, 17, 17) and prediction.dtype == np.float64
    assert np.array_equal(prediction[0], supplied), 'Supplied initial state must be returned exactly'
print('burgers_training_entrypoint_smoke=passed', flush=True)
