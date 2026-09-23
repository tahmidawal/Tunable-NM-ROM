"""Tiny manufactured Burgers-contract fixture at 64 intervals for one family.

The fields are manufactured, not Burgers solutions. This exercises the complete
`train.py` entrypoint for the requested family at the 65×65 nodal grid (so the
U-Net padding to a multiple of 16 and the Transolver patching are both non-trivial),
the four-key schema, the five evolved output channels and the exactly returned
supplied initial state. It is implementation evidence only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import dataset

TIMES = np.array([0., .05, .1, .15, .2, .25], dtype=np.float64)
TINY = {'unet': dict(family='unet', base=8, groups=8),
        'transolver': dict(family='transolver', dim=32, layers=2, heads=4, slices=8, mlp_ratio=2, patch=4, ref=8),
        'deeponet': dict(family='deeponet', width=8, rank=16, trunk_width=16, levels=2, pool_bins=2),
        # The FNO is float64 by construction (`model.to_f64`), so it is smoked at float64 only.
        'fno': dict(family='fno', width=8, modes=4, layers=2)}

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
parser.add_argument('--family', required=True, choices=sorted(TINY))
parser.add_argument('--dtype', default='float32', choices=('float32', 'float64'))
parser.add_argument('--mesh', type=int, default=64)
parser.add_argument('--pde', default='burgers', choices=('burgers', 'poisson'))
args = parser.parse_args()
# The FNO is float64 by construction (`model.to_f64` ignores a declared dtype), so record the
# dtype it actually runs at rather than the float32 default.
if args.family == 'fno':
    args.dtype = 'float64'
POISSON = args.pde == 'poisson'
if POISSON:
    TIMES = np.array([0.], dtype=np.float64)
args.out.mkdir(parents=True, exist_ok=False)
n = args.mesh
for split, count, code in [('train', 4, 300), ('validation', 2, 400)]:
    folder = args.out / split
    folder.mkdir()
    rows = []
    for i in range(count):
        coord = np.arange(n + 1, dtype=np.float64) / n
        field = np.sin(np.pi * coord[:, None]) * np.sin(np.pi * coord[None, :])
        field = field * (1 + (code + i) / 1000)
        field[[0, -1], :] = 0
        field[:, [0, -1]] = 0
        nu = .02 + (code + i) / 10000
        if POISSON:
            # Manufactured Poisson pair: source f = 2 pi^2 u for u = sin(pi x) sin(pi y) (scaled).
            trajectory = field[None, None].copy()
            field = 2 * np.pi ** 2 * field
            parameters = np.zeros((0,), dtype=np.float64)
        else:
            trajectory = np.stack([field * np.exp(-2 * np.pi ** 2 * nu * t) for t in TIMES])[:, None]
            trajectory[0] = field[None]
            parameters = np.array([nu], dtype=np.float64)
        cid = f'smoke-{args.family}-{split}-{i}'
        path = folder / f'{cid}.npz'
        np.savez(path, input=field[None], target=trajectory, parameters=parameters, times=TIMES)
        rows.append(dict(case_id=cid, split=split, case_index=i, seed=code + i, mesh=n,
                         path=path.name, sha256=dataset.sha256(path)))
    (folder / 'index.json').write_text(json.dumps(dict(pde=args.pde, complete=True, records=rows,
        kind='manufactured implementation fixture; not PDE-family training evidence')))
config = dict(TINY[args.family], dtype=args.dtype, epochs=2, patience=2, batch_size=2, warmup_epochs=1,
              learning_rate=.001, weight_decay=.0001, seed=20260914)
(args.out / 'config.json').write_text(json.dumps(config))
subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')), '--smoke',
    '--train-index', str(args.out / 'train/index.json'),
    '--validation-index', str(args.out / 'validation/index.json'),
    '--config', str(args.out / 'config.json'), '--out', str(args.out / 'run')], check=True)
result = json.loads((args.out / 'run/result.json').read_text())
assert result['epochs_completed'] == 2 and result['complete'] and result['pde'] == args.pde
assert result['family'] == args.family and result['parameter_dtype'] == f'torch.{args.dtype}'
assert result['stopped_by_epoch_cap'] and not result['stopped_by_early_stopping'] and not result['stopped_by_wall_budget']
predictions = sorted(args.out.glob('run/*.prediction.npz'))
assert len(predictions) == 2
for path, row in zip(predictions, json.loads((args.out / 'validation/index.json').read_text())['records']):
    with np.load(path) as saved:
        prediction = saved['prediction']
    with np.load(args.out / 'validation' / row['path']) as truth:
        supplied = truth['input']
    assert prediction.shape == ((1, 1, n + 1, n + 1) if POISSON else (6, 1, n + 1, n + 1)) and prediction.dtype == np.float64
    assert POISSON or np.array_equal(prediction[0], supplied), 'Supplied initial state must be returned exactly'
    for edge in (prediction[..., 0, :], prediction[..., -1, :], prediction[..., :, 0], prediction[..., :, -1]):
        assert not np.any(edge)
print(json.dumps(dict(family=args.family, pde=args.pde, dtype=args.dtype, mesh=n, real_parameters=result['real_parameter_count'],
                      training_seconds=result['training_seconds'])), flush=True)
print(f'{args.family}_training_entrypoint_smoke=passed', flush=True)
