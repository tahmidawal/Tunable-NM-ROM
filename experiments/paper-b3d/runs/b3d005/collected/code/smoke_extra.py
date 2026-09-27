"""Bounded actual B3D adapter, output-contract, and head-retraining checks."""
import json,pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import core as c
import operator_panel as op
from operators import models3d as M
from run import host
from head_candidate import train_candidate
assert jax.default_backend()=='gpu'
cfg=json.loads(Path('config-extra.json').read_text());checks={}
n=cfg['nodes'];ni=n-2;xyz=c.b3.grid_coords_3d(n)[c.b3.interior_indices_3d(n)].reshape(ni,ni,ni,3)*2-1
u0=np.exp(-np.sum((xyz-.1)**2,axis=-1)*10)
data=jax.tree_util.tree_map(jnp.asarray,dict(xyz=xyz,scale=.2,nu_center=.02,nu_scale=.01))
for model in cfg['operators'][2:]:
    spec=model['spec'];p=M.init_model(jax.random.PRNGKey(7),spec,5,7)
    assert all(x.dtype==jnp.float64 for x in jax.tree_util.tree_leaves(p))
    x=op.input_fields(jnp.asarray(u0),.02,data['xyz'],.2,.02,.01)[None]
    value,grad=jax.jit(jax.value_and_grad(lambda p,x:jnp.mean(M.apply_model(p,x,spec)**2)))(p,x)
    assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(a)).all() for a in jax.tree_util.tree_leaves(grad))
    fields,knots=host(op.make_query(spec,n,cfg['steps'],cfg['train_steps'])(p,jnp.asarray(u0.ravel()),.02,data))
    assert fields.shape==(51,ni**3) and np.array_equal(fields[0],u0.ravel())
    discrepancy=float(np.max(np.abs(fields-op.interpolation_matrix(cfg['train_steps'],cfg['steps'])@knots)))
    assert discrepancy<1e-12
    checks[model['name']]=dict(f64=True,parameters=M.parameter_count(p),native_shape=list(fields.shape),interpolation_max=discrepancy,finite_backward=True,coordinate_channels=spec['coordinate_channels'])
    del p,grad,fields,knots;jax.clear_caches()
small=json.loads(Path('checks/config-operator-smoke.json').read_text());small['head_candidate']=dict(latent_dimension=4,head_width=24,head_steps=3,seed=920321,head_learning_rate=.0005,head_batch=4,checkpoint_every=3)
train_candidate(small,'runs/operator-smoke/training','runs/extra-smoke/head')
old=pickle.loads(Path('runs/operator-smoke/training/checkpoint.pkl').read_bytes());new=pickle.loads(Path('runs/extra-smoke/head/checkpoint.pkl').read_bytes())
for key in ['B','g','out_scale']:
    assert all(np.array_equal(a,b) for a,b in zip(jax.tree_util.tree_leaves(old['params'][key]),jax.tree_util.tree_leaves(new['params'][key])))
assert new['Z_tr'].shape==(8,4)
checks['head_candidate']=dict(bank_unchanged=True,code_shape=list(new['Z_tr'].shape),coefficient_conversion_relative=new['coefficient_conversion_relative'],finite_training=True)
checks['backend']=jax.default_backend();Path('checks/extra-smoke.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks,flush=True)
