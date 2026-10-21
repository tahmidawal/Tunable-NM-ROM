"""Tiny actual training, validation selection and saved-checkpoint replay."""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jnp
import common as C
from operators import training as T
from operators import models3d as M
from operators import heat_adapter as H
kind=sys.argv[1];out=Path(sys.argv[2]);assert jax.default_backend()=='gpu'
cfg=dict(steps=4,wall_seconds=40,batch_size=2,seed=920323,learning_rate=.001,validation_every=2,curve_every=1,components_per_output=1)
spec=dict(kind=kind,width=2,rank=4,trunk_width=8,levels=2,pool_bins=2,depth=1,heads=2,slices=4,patch=2,reference_grid=2,periodic=False)
rng=np.random.default_rng(920324);fields=rng.uniform(size=(4,3,7,7,7));x,y=H.arrays(fields,1.)
p,info=T.train(x[:3],y[:3],x[3:],y[3:],spec,cfg,out,C.dump,C.checkpoint)
loaded=pickle.loads((out/'best.pkl').read_bytes())
result=np.asarray(H.engine(p,spec,1.,8)(jnp.asarray(fields[3,0])))
replayed=np.asarray(H.engine(loaded['params'],loaded['spec'],1.,8)(jnp.asarray(fields[3,0])))
assert np.array_equal(result,replayed) and np.array_equal(result[0],fields[3,0])
e=np.linalg.norm((result[1:]-fields[3,1:]).reshape(2,-1),axis=1)/np.linalg.norm(fields[3,1:].reshape(2,-1),axis=1)
np.testing.assert_allclose(e,loaded['validation_error_by_case_time'][0],rtol=2e-13,atol=2e-13)
assert all(a.dtype==jnp.float64 for a in jax.tree_util.tree_leaves(loaded['params']))
fine=rng.uniform(size=(15,15,15));native=np.asarray(H.native_engine(p,spec,1.,8,16)(jnp.asarray(fine)))
assert native.shape==(3,15,15,15) and np.array_equal(native[0],fine)
metrics=dict(kind=kind,checkpoint_replay_exact=True,validation_independent_relative=e.tolist(),native_interpolation_shape=list(native.shape),dtype='float64',training=info)
C.dump(out/'audit.json',metrics);print(json.dumps(metrics),flush=True)
