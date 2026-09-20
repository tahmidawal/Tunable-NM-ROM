"""Persist development-built POD assets and load them unchanged at final test."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np


def names(cfg):
    if not cfg.get('frozen_offline_assets', False):
        return []
    return [f'offline/pod_N{n}.{suffix}' for n in cfg['evaluation_intervals']
            for suffix in ('npy', 'json')]


def save(out, n, basis, info):
    folder=Path(out)/'offline';folder.mkdir(exist_ok=True)
    np.save(folder/f'pod_N{n}.npy',basis,allow_pickle=False)
    (folder/f'pod_N{n}.json').write_text(json.dumps(info,indent=2)+'\n')


def load(source, out, n):
    source=Path(source);out=Path(out);(out/'offline').mkdir(exist_ok=True)
    for suffix in ('npy','json'):
        name=f'offline/pod_N{n}.{suffix}'
        if (source/name).resolve()!=(out/name).resolve():shutil.copy2(source/name,out/name)
        with (source/name).open('rb') as left, (out/name).open('rb') as right:
            assert hashlib.file_digest(left,'sha256').hexdigest()==hashlib.file_digest(right,'sha256').hexdigest()
    basis=np.load(out/f'offline/pod_N{n}.npy',allow_pickle=False)
    info=json.loads((out/f'offline/pod_N{n}.json').read_text())
    assert basis.dtype==np.float64 and basis.shape==((n-1)**3,info['retained_rank'])
    return basis,info
