"""Parity gate: swapping in the DeepONet lane's `families.py` / `model.py` must not move a
single bit of any arm `opt101` already measured.

    jaxrun /home/tahmid/Dev/.venv/bin/python checks/parity_families.py

`opt201` replaces `lib/families.py` and `lib/model.py` with the `ops-deeponet-b2d` lane's
versions, which add a `deeponet` branch. The diff reads as purely additive, but "reads as" is
not a gate. This loads each of `opt101`'s nine checkpoints twice -- once under the modules
`opt101` ran, once under the replacements -- and requires the predicted trajectories to be
bitwise equal. A single differing bit means the nine existing rows would not be re-measurements
of the same models and the job must not be submitted.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
WORKTREES = LANE.parents[2]
DEEPONET = WORKTREES / '2026-09-22-ops-deeponet-b2d/experiments/ops-deeponet-b2d'
L = 256


def load_modules(directory):
    """Import `dataset`/`model`/`families` out of `directory` in a fresh module namespace."""
    for name in ('model', 'families', 'dataset', 'spectral_conv_f64'):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module('model')
    finally:
        sys.path.remove(str(directory))


def predict(adapter, path, field, parameters):
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    net = adapter.make_model(ckpt['pde'], ckpt['config'])
    net.load_state_dict(ckpt['model'])
    adapter.check_dtypes(net)
    net.eval()
    norm = tuple(v.cuda() for v in ckpt['normalization'])
    with torch.no_grad():
        out = adapter.predict(net, torch.from_numpy(field).cuda(),
                              torch.from_numpy(parameters).cuda(), *norm, ckpt['pde'])
    got = out.cpu().numpy()[0, :, 0]
    del net, ckpt
    torch.cuda.empty_cache()
    return got


def main():
    # A scratch copy of lib/ with the two replacement files in place: the real lane directory is
    # never touched by this check.
    scratch = Path(tempfile.mkdtemp(prefix='parity-families-'))
    shutil.copytree(LANE / 'lib', scratch / 'new')
    for name, src in (('families.py', DEEPONET / 'families.py'), ('model.py', DEEPONET / 'model.py')):
        shutil.copy(src, scratch / 'new' / name)

    specs = json.loads((LANE / 'operators.json').read_text())['checkpoints']
    rng = np.random.default_rng(20260922)
    field = rng.standard_normal((1, 1, L + 1, L + 1))
    field[..., 0, :] = field[..., -1, :] = field[..., :, 0] = field[..., :, -1] = 0.
    parameters = np.array([[0.02]])

    ok = True
    for spec in specs:
        path = WORKTREES / spec['path'].removeprefix('worktrees/')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256'], spec['name']
        old = predict(load_modules(LANE / 'lib'), path, field, parameters)
        new = predict(load_modules(scratch / 'new'), path, field, parameters)
        same = bool(np.array_equal(old, new))
        ok &= same
        print(f"{spec['name']:12s} {spec['family']:10s} bitwise_identical={same} "
              f"max_abs_diff={float(np.max(np.abs(old - new))):.3e}", flush=True)
    shutil.rmtree(scratch)
    print('PARITY', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
