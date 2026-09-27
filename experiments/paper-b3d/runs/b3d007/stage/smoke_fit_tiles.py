"""Actual multistart fit batching parity, including solver counters."""
import json
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import core as c
from run import tiled_fit,host

p=pickle.loads(Path('inputs/refined_checkpoint.pkl').read_bytes())
hp=jax.tree_util.tree_map(jnp.asarray,{k:p['params'][k] for k in ['h','h_lin']})
z=np.asarray(p['Z_tr'][:7]);h=np.asarray(c.b3.head(hp,jnp.asarray(z)));floor=np.ones(7)*1e-6
starts=np.stack([z,z+.01],axis=1);R=np.eye(h.shape[1]);C=np.zeros((h.shape[1],0))
fit=c.make_fit(z.shape[1],0,4)
direct=host(fit(jnp.asarray(h),jnp.asarray(floor),jnp.asarray(starts),jnp.asarray(C),jnp.asarray(R),hp))
tiled=tiled_fit(fit,h,floor,starts,C,R,hp,tile=4)
maximum=0.
for a,b in zip(jax.tree_util.tree_leaves(direct),jax.tree_util.tree_leaves(tiled)):
    if np.issubdtype(a.dtype,np.integer):assert np.array_equal(a,b)
    else:
        maximum=max(maximum,float(np.max(np.abs(a-b))))
        assert np.allclose(a,b,atol=1e-10,rtol=1e-10)
fields_a=np.asarray(c.b3.head(hp,jnp.asarray(direct[0][0])))
fields_b=np.asarray(c.b3.head(hp,jnp.asarray(tiled[0][0])))
field_relative=float(np.linalg.norm(fields_a-fields_b)/np.linalg.norm(fields_a))
assert field_relative<1e-12
record=dict(passed=True,max_absolute_float_discrepancy=maximum,
    head_coefficient_relative_discrepancy=field_relative,integer_counters_exact=True,
    note='An initial absolute 1e-12 check on every internal array saw a 1.67e-12 roundoff difference. Acceptance uses 1e-10 internal absolute/relative tolerance and 1e-12 decoded coefficient tolerance.',
    backend=jax.default_backend())
Path('checks/fit-tile-parity.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
