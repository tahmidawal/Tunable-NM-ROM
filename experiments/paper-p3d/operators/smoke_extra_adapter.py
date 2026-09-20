"""Bounded actual forcing adapter, training, serialization and nodal transfer check."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import common as C
import poisson as P
from operators import poisson_adapter as A,models3d as M,training as T

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
p=C.family(123,4);fields=P.dataset(8,p);forcing=np.stack([np.asarray(P.source(8,a)) for a in p])
scales=dict(input=float(np.sqrt(np.mean(forcing[:3]**2))),output=float(np.sqrt(np.mean(fields[:3]**2))))
x,y=A.arrays(forcing,fields,scales);assert x.shape==(4,7,7,7,4) and y.shape==(4,7,7,7,1)
rows=[]
for spec in [dict(kind=sys.argv[2],width=2,rank=4,trunk_width=8,levels=2,pool_bins=2,depth=1,heads=2,slices=4,patch=2,reference_grid=2,periodic=False,coordinate_channels=[-3,-2,-1])]:
    cfg=dict(steps=2,wall_seconds=30,batch_size=1,seed=99,learning_rate=.001,validation_every=1,components_per_output=1)
    params,info=T.train(x[:3],y[:3],x[3:],y[3:],spec,cfg,out/spec['kind'],C.dump,C.checkpoint)
    pred=np.asarray(A.engine(params,spec,scales,8)(jnp.asarray(forcing[3])))
    actual=np.asarray(M.apply_model(params,jnp.asarray(x[3:]),spec))[0,...,0]*scales['output']
    assert np.max(np.abs(actual-pred))<1e-13
    saved=pickle.loads((out/spec['kind']/'best.pkl').read_bytes())
    replay=np.asarray(A.engine(saved['params'],spec,scales,8)(jnp.asarray(forcing[3])))
    assert np.array_equal(replay,pred)
    n=16;fine=np.zeros((15,15,15));fine[1::2,1::2,1::2]=forcing[3]
    mapped=np.asarray(A.native_interpolated(params,spec,scales,n,8)(jnp.asarray(fine)))
    assert np.max(np.abs(mapped[1::2,1::2,1::2]-pred))<1e-13
    # Independent SciPy interpolation on a grid that includes the zero boundary.
    from scipy.interpolate import RegularGridInterpolator
    old=np.linspace(0,1,9);f=RegularGridInterpolator((old,old,old),np.pad(pred,1))
    a=np.arange(1,16)/16;xx=np.stack(np.meshgrid(a,a,a,indexing='ij'),-1)
    defect=float(np.max(np.abs(mapped-f(xx))))
    assert defect<1e-13 and mapped.dtype==np.float64
    rows.append(dict(kind=spec['kind'],adapter_defect=float(np.max(np.abs(actual-pred))),independent_interpolation_defect=defect,pickle_replay=True,training=info))
C.dump(out/'audit.json',dict(passed=True,checks=rows));print(json.dumps(rows,indent=2))
