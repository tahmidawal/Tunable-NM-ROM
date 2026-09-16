"""Capture the parameter draws a cluster attempt actually used, and prove it.

The training job records only a SHA256 of each draw. Auditing that hash by
re-deriving the draw on this machine fails, not because the draw is wrong but
because `np.exp` differs by one unit in the last place between the local NumPy
and the cluster's: the viscosity column is the only one affected and the whole
discrepancy is 1 ULP (see DESIGN.md A5).

So the draws are regenerated in the cluster's own interpreter -- a CPU-only,
deterministic, seconds-long login-node command, not a GPU job -- and accepted
only if each one hashes to EXACTLY the value the job recorded. That equality is
the proof: the arrays written here are byte-identical to the ones the job used,
and every later check can then be made on values rather than on a hash that does
not survive a NumPy upgrade.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REMOTE_PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
# the draw, spelled exactly as `engines.params_draw` spells it
REMOTE = '''
import numpy as np, base64, json, sys
def pd(seed, count):
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count),
                     r.uniform(.05, .20, count), r.uniform(.5, 2., count),
                     np.exp(r.uniform(np.log(.01), np.log(.1), count))], axis=1)
spec = json.loads(%s)
out = {'numpy': np.__version__, 'python': sys.version.split()[0]}
for name, parts in spec.items():
    a = np.concatenate([pd(s, c) for s, c in parts])
    out[name] = base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    out[name + '_shape'] = list(a.shape)
sys.stdout.write(json.dumps(out))
'''


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result', help='the attempt result.json that recorded the hashes')
    p.add_argument('--out', required=True, help='npz to write')
    p.add_argument('--host', default='tufts-login')
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    cfg = r['config']
    spec = {
        'train': [[cfg['canonical_seed'], cfg['canonical_trajectories']],
                  [cfg['extra_seed'], cfg['extra_trajectories']]],
        'holdout': [[cfg['holdout_seed'], cfg['holdout_trajectories']]],
        'eval': [[cfg['eval_seed'], cfg['eval_cases']],
                 [cfg['eval_fresh_seed'], cfg['eval_fresh_cases']]],
    }
    want = {'train': r['data']['train_physical_sha256'],
            'holdout': r['data']['holdout_physical_sha256'],
            'eval': r['data']['eval_physical_sha256']}
    script = REMOTE % repr(json.dumps(spec))
    cp = subprocess.run(['ssh', a.host, f'{REMOTE_PY} -'], input=script, text=True,
                        capture_output=True, check=True)
    got = json.loads(cp.stdout)
    arrays, prov = {}, dict(host=a.host, remote_numpy=got['numpy'], remote_python=got['python'],
                            local_numpy=np.__version__, recorded_sha256=want, seeds=spec,
                            source_result=str(Path(a.result).resolve()))
    for name in want:
        arr = np.frombuffer(base64.b64decode(got[name]),
                            dtype=np.float64).reshape(*got[name + '_shape'])
        h = sha_array(arr)
        assert h == want[name], (name, h, want[name])
        arrays[name] = arr
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    Path(str(out) + '.provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(json.dumps({k: [v.shape, want[k]] for k, v in arrays.items()}, indent=1))
    print('EVERY DRAW HASHES TO THE VALUE THE JOB RECORDED ->', out)


if __name__ == '__main__':
    main()
