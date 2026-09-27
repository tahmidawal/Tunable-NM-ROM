"""Bounded multi-output initial-norm and unchanged-architecture smoke."""
import json
import subprocess
import time
import types
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from operators import models3d as M
from operators import extra_models3d as E
from operators.pretrained_deeponet import pretrain
from operators.training import train
from run import dump
from train import checkpoint

start=time.perf_counter();root=Path(__file__).parent/'checks/conditioned';root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(920801);axis=np.linspace(-1,1,6);xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1)
x=np.concatenate((rng.normal(size=(6,6,6,6,1)),np.broadcast_to(xyz,(6,6,6,6,3)),rng.normal(size=(6,6,6,6,1))),axis=-1)
y=rng.normal(size=(6,6,6,6,3));den=np.repeat(np.sum(x[...,0]**2,axis=(1,2,3))[:,None],3,axis=1)
spec=dict(kind='deeponet3d',width=2,levels=2,pool_bins=2,rank=4,trunk_width=8,periodic=False,coordinate_channels=[1,2,3])
cfg=dict(seed=920802,steps=2,wall_seconds=25,batch_size=2,learning_rate=.0003,validation_every=1,curve_every=1)
pre=dict(trunk_steps=2,trunk_learning_rate=.001,trunk_batch_points=32,trunk_wall_seconds=15,
    branch_steps=2,branch_learning_rate=.001,branch_batch_size=2,branch_wall_seconds=15,checkpoint_every=1)
legacy=types.ModuleType('operators.legacy_extra');legacy.__package__='operators'
exec(subprocess.check_output(['git','show','178c0e83:experiments/paper-b3d/operators/extra_models3d.py']),legacy.__dict__)
p=M.init_model(jax.random.PRNGKey(cfg['seed']),spec,5,3)
before=np.asarray(legacy.apply_extra(p,jnp.asarray(x),spec));after=np.asarray(M.apply_model(p,jnp.asarray(x),spec))
assert np.array_equal(before,after)
p,info=pretrain(x,y,spec,cfg,pre,root/'pretraining',dump,checkpoint,den)
p,trained=train(x,y,x[:1],y[:1],spec,cfg,root/'joint',dump,checkpoint,den,den[:1],initial_params=p)
t=np.load(root/'pretraining/training_teacher.npz');b=t['spatial_basis'];snap=np.moveaxis(y,-1,1).reshape(18,-1)
err=np.linalg.norm(snap@b@b.T-snap,axis=1)/np.sqrt(den.ravel())
assert np.max(np.abs(err-t['projection_errors']))<1e-12
g=np.load(root/'pretraining/branch_teacher.npz');initial=__import__('pickle').loads((root/'pretraining/trunk_selected.pkl').read_bytes())['params']
matrix=np.asarray(E.deeponet_trunk(initial,jnp.asarray(xyz),spec)).reshape(-1,4)/2
assert np.max(np.abs(matrix.T@matrix-g['gram']))<1e-12
result=dict(passed=True,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,
    native_bitwise_identity=True,teacher_projection_maximum_defect=float(np.max(np.abs(err-t['projection_errors']))),
    explicit_initial_norm_multi_output=True,seconds=time.perf_counter()-start)
assert result['seconds']<60;dump(root/'audit.json',result);print(json.dumps(result))
