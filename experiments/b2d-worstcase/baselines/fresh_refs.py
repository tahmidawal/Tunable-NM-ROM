"""Reference solutions of a fresh cohort with the operator lane's generator (baselines/ops/gen_data.py, unchanged):
fft_tight, 256 intervals, float64 fields (n, 6, 257, 257) at t = 0, .05, ..., .25. Files <out>/<name>-L256.*
in the layout its evaluate.py and the CoLoRA lane's rom.py read.   fresh_refs.py --seed S --count N --name NAME --out DIR"""
import argparse, json, os
from pathlib import Path
import numpy as np
import jax
import gen_data as g

a = argparse.ArgumentParser()
a.add_argument('--seed', type=int, required=True); a.add_argument('--count', type=int, default=64)
a.add_argument('--name', required=True); a.add_argument('--out', required=True)
a = a.parse_args()
assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest' and jax.default_backend() == 'gpu'
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
name = f'{a.name}-L256'
assert not (out / f'{name}.fields.npy').exists(), 'cohort already generated'
phys = g.e.params_draw(a.seed, a.count)
seen = np.concatenate((g.pool_params(), g.e.params_draw(20260916, 64)))
assert not (phys[:, None, :] == seen[None, :, :]).all(-1).any(), 'cohort overlaps the pool or test64'
fields, n0, info = g.solve(256, phys, np.float64)
np.save(out / f'{name}.fields.npy', fields)
np.savez(out / f'{name}.meta.npz', params=phys, n0=n0)
info.update(name=name, seed=a.seed, cases=len(phys), physical_sha256=g.sha_array(phys), fields_sha256=g.sha_array(fields),
            job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'))
(out / f'{name}.json').write_text(json.dumps(info, indent=1) + '\n')
print('DONE', json.dumps(info), flush=True)
