"""Generalized FOM parity plus independent fresh checkpoint evaluation."""
import json
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import core as c

assert jax.default_backend()=='gpu'
ck=pickle.loads(Path('runs/training-smoke/checkpoint.pkl').read_bytes())
params=jax.tree_util.tree_map(jnp.asarray,ck['params'])
gates=c.verify(params,ck['Z_tr'])
n=9;xyz=c.b3.grid_coords_3d(n)[c.b3.interior_indices_3d(n)]
u0=jnp.asarray(np.prod(np.sin(np.pi*xyz),axis=1))
old=c.b3.make_newton_tol_rollout(n,'fft')(u0,.02,1e-10,1e-11)
new=c.make_fom(n)(u0,.02,1e-10,1e-11)
delta=float(jnp.linalg.norm(old[0]-new[0])/jnp.linalg.norm(old[0]))
assert delta<1e-11
fine=c.make_fom(n,.0025,100)(u0,.02,1e-10,1e-11)
assert fine[0].shape==(101,(n-2)**3) and np.isfinite(fine[0]).all()
gates.update(generalized_fom_relative_parity=delta,half_dt_shape=list(fine[0].shape),backend=jax.default_backend())
Path('checks/training-smoke.json').write_text(json.dumps(gates,indent=2)+'\n');print(json.dumps(gates,indent=2))
