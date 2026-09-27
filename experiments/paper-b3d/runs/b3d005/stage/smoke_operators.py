"""Bounded training refresh / sparse-output contract verification, never evidence."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import core as c
import train as trainer
import operator_panel as op
from run import tiled_fit, host

assert jax.default_backend()=='gpu'
cfg=json.loads(Path('checks/config-training-smoke.json').read_text())
cfg['training']['coefficient_refresh_every']=2
cfg['operators']=[dict(name='fno3d',spec=dict(kind='fno3d',width=2,modes=[2,2,2],depth=1,padding=1),
    training=dict(steps=2,wall_seconds=20,batch_size=1,seed=7,learning_rate=.001,validation_every=1,curve_every=1,components_per_output=1)),
    dict(name='unet3d',spec=dict(kind='unet3d',width=2,levels=2,periodic=False),
    training=dict(steps=2,wall_seconds=20,batch_size=1,seed=8,learning_rate=.001,validation_every=1,curve_every=1,components_per_output=1))]
path=Path('checks/config-operator-smoke.json');path.write_text(json.dumps(cfg,indent=2)+'\n')
c.b3.N_REF=17
sys.argv=['train','--config',str(path),'--out','runs/operator-smoke/training'];trainer.main()
sys.argv=['operator','--config',str(path),'--training','runs/operator-smoke/training','--out','runs/operator-smoke/out','--mode','train'];op.main()
metadata=json.loads(Path('runs/operator-smoke/out/operator_metadata.json').read_text())
n=cfg['nodes'];xyz=c.b3.grid_coords_3d(n)[c.b3.interior_indices_3d(n)].reshape((n-2,n-2,n-2,3))*2-1
raw=np.load('runs/operator-smoke/training/training_fields.npy');u0=jnp.asarray(raw[0])
data=jax.tree_util.tree_map(jnp.asarray,dict(xyz=xyz,**{k:metadata[k] for k in ['scale','nu_center','nu_scale']}))
checks={}
for model in cfg['operators']:
    ck=pickle.loads(Path('runs/operator-smoke/out',model['name'],'best.pkl').read_bytes())
    query=op.make_query(model['spec'],n,cfg['steps'],cfg['train_steps'])
    fields,knots=host(query(jax.tree_util.tree_map(jnp.asarray,ck['params']),u0,.02,data))
    interpolated=op.interpolation_matrix(cfg['train_steps'],cfg['steps'])@knots
    assert fields.shape==(51,(n-2)**3) and np.array_equal(fields[0],np.asarray(u0))
    assert np.max(np.abs(interpolated-fields))<1e-12 and np.isfinite(fields).all()
    checks[model['name']]=dict(dtype=str(fields.dtype),shape=list(fields.shape),interpolation_max=float(np.max(np.abs(interpolated-fields))))
# The tile pad must preserve an uneven batch exactly.
fake=jax.jit(lambda target,floor,starts,C,R,hp:((target,floor), (target[:,None],floor[:,None])))
t=np.arange(21.).reshape(7,3);f=np.arange(7.)
a=tiled_fit(fake,t,f,np.zeros((7,1,2)),np.zeros((3,0)),np.eye(3),{},tile=4)
assert np.array_equal(a[0][0],t) and np.array_equal(a[0][1],f)
checks['tile_padding']=True
checks['backend']=jax.default_backend();checks['precision']='f64/highest'
Path('checks/operator-smoke.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks,flush=True)
