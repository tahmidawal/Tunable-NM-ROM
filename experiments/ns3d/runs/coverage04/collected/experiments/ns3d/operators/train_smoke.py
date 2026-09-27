import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pickle
import common as C
from operators import training as T
from operators import heat_adapter as H
from operators import models3d as M
import jax
import jax.numpy as jnp
cfg=dict(steps=3,wall_seconds=30,batch_size=2,seed=34,learning_rate=.001,validation_every=2,components_per_output=1)
rng=np.random.default_rng(2);fields=rng.uniform(size=(4,3,7,7,7));x,y=H.arrays(fields,1.)
for label in ('fno3d','unet3d','unet3d_periodic'):
 kind='unet3d' if label.endswith('periodic') else label
 spec=dict(kind=kind,width=2,modes=[2,2,2],depth=1,padding=1,levels=2,periodic=label.endswith('periodic'))
 p,info=T.train(x[:3],y[:3],x[3:],y[3:],spec,cfg,Path('/tmp/h3d-operator-smoke')/label,C.dump,C.checkpoint)
 fn=H.engine(p,spec,1.,8);value=np.asarray(fn(jnp.asarray(fields[0,0])))
 assert value.shape==(3,7,7,7) and np.array_equal(value[0],fields[0,0])
 loaded=pickle.loads((Path('/tmp/h3d-operator-smoke')/label/'best.pkl').read_bytes())
 replay=np.asarray(H.engine(loaded['params'],spec,1.,8)(jnp.asarray(fields[0,0])))
 assert np.array_equal(value,replay)
 print(label,info,flush=True)

# Vector/time packing and custom initial-norm denominator path.
vx=np.repeat(x[...,:1],3,axis=-1);vy=np.repeat(y,3,axis=-1)
den=np.repeat(np.sum(vx**2,axis=(1,2,3,4))[:,None],2,axis=1)
vcfg={**cfg,"components_per_output":3,"normalization":"initial vector norm"}
T.train(vx[:3],vy[:3],vx[3:],vy[3:],dict(kind="fno3d",width=2,modes=[2,2,2],depth=1,padding=0),vcfg,Path("/tmp/h3d-operator-smoke/vector"),C.dump,C.checkpoint,den[:3],den[3:])
print("vector custom-denominator smoke passed",flush=True)
