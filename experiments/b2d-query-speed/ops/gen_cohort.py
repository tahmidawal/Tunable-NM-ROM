"""b2d-query-speed: reference fields of a 64-case cohort for the operator evaluation, in the file layout of
burgers-ops-matched gen_data.py `test` (same solver text: burgers_fom.make_fom(L, .005), ntol 1e-6, ltol 1e-8).
    gen_cohort.py --seed S --count N --name NAME --out DIR [--expect-sha SHA]
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import burgers_fom as e

NTOL, LTOL = 1e-6, 1e-8
a = argparse.ArgumentParser()
a.add_argument('--seed', type=int, required=True); a.add_argument('--count', type=int, default=64)
a.add_argument('--name', required=True); a.add_argument('--out', required=True); a.add_argument('--expect-sha')
a = a.parse_args()
assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest' and jax.default_backend() == 'gpu'
L = 256
phys = e.params_draw(a.seed, a.count)
sha = hashlib.sha256(np.ascontiguousarray(phys).tobytes()).hexdigest()
assert a.expect_sha in (None, sha), sha
fn = e.make_fom(L, .005)
out = np.empty((len(phys), 6, L + 1, L + 1))
n0, resid = [], []
for i, p in enumerate(phys):
    f, it, rn = fn(jnp.asarray(e.initial(L, p)), float(p[4]), NTOL, LTOL)
    f = np.asarray(f, dtype=np.float64); rn = np.asarray(rn)
    assert np.isfinite(f).all() and rn.max() <= NTOL * (1 + 1e-9), ('not converged', i, float(rn.max()))
    out[i] = f; n0.append(float(np.linalg.norm(f[0]))); resid.append(float(rn.max()))
o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
np.save(o / f'{a.name}-L{L}.fields.npy', out)
np.savez(o / f'{a.name}-L{L}.meta.npz', params=phys, n0=np.array(n0))
info = dict(name=a.name, seed=a.seed, cases=len(phys), physical_sha256=sha, max_relative_residual=max(resid),
            fields_sha256=hashlib.sha256(out.tobytes()).hexdigest(), job_id=os.environ.get('SLURM_JOB_ID'))
(o / f'{a.name}-L{L}.json').write_text(json.dumps(info, indent=1) + '\n')
print('COHORT DONE', json.dumps(info), flush=True)
