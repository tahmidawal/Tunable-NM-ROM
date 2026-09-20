"""Bounded GPU checks of same-architecture initialization and transfer identity."""
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import common as C
import poisson as P
from operators import poisson_adapter as O
from operators import models3d as M
from operators import extra_models3d as E
from operators import training as T
from operators.pretrained_deeponet import pretrain
from trunk_diagnostic import trunk as numpy_trunk
from audit_pretraining import audit as audit_teacher

start=time.perf_counter();root=Path(__file__).parent/'runs/pretraining-smoke2';root.mkdir(parents=True,exist_ok=False)
n=8;parameters=C.family(920701,12);solutions=P.dataset(n,parameters)
forcing=np.stack([np.asarray(P.source(n,p)) for p in parameters]);scales=dict(input=1.,output=1.)
x,y=O.arrays(forcing,solutions,scales)
spec=dict(kind='deeponet3d',width=2,levels=2,pool_bins=2,rank=4,trunk_width=8,periodic=False,coordinate_channels=[-3,-2,-1])
cfg=dict(seed=920702,steps=2,wall_seconds=30,batch_size=2,learning_rate=.0003,validation_every=1,curve_every=1)
pre=dict(trunk_steps=2,trunk_learning_rate=.001,trunk_batch_points=32,trunk_wall_seconds=15,
    branch_steps=2,branch_learning_rate=.001,branch_batch_size=2,branch_wall_seconds=15,checkpoint_every=1)
params,info=pretrain(x,y,spec,cfg,pre,root/'operators/test/pretraining',C.dump,C.checkpoint)
params,trained=T.train(x,y,x[:1],y[:1],spec,cfg,root/'operators/test',C.dump,C.checkpoint,initial_params=params)
native=O.engine(params,spec,scales,n)(jnp.asarray(forcing[0]));transfer=O.native_sensor_deeponet(params,spec,scales,n,n)(jnp.asarray(forcing[0]))
defect=float(np.max(np.abs(np.asarray(native)-np.asarray(transfer))));assert defect<1e-12
fine=P.source(16,parameters[0]);direct=O.native_sensor_deeponet(params,spec,scales,16,n)(fine)
coef=np.asarray(E.deeponet_coefficients(params,jnp.asarray(forcing[:1,...,None]),spec))
trunk=numpy_trunk(jax.device_get(params),np.asarray(O.coordinates(16)),spec)
expected=np.einsum('bcr,xyzr->bxyzc',coef,trunk)/np.sqrt(spec['rank'])+np.asarray(params['bias'])
fine_defect=float(np.max(np.abs(np.asarray(direct)-expected[0,...,0])));assert fine_defect<1e-12
result=dict(passed=True,native_transfer_maximum_defect=defect,independent_fine_trunk_maximum_defect=fine_defect,
    pretraining_used_no_development_data=not info['development_data_used'],seconds=time.perf_counter()-start,
    architecture=spec,backend=jax.default_backend(),x64=jax.config.jax_enable_x64)
result['independent_teacher_audit']=audit_teacher(root,dict(config=dict(train_intervals=n,train_seed=920701,train_count=12),
    operators=[dict(name='test',scales=scales,pretraining=info)]))
C.dump(root/'audit.json',result);print(json.dumps(result))
