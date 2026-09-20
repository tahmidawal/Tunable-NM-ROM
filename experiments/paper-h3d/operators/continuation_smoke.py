"""Check warm-restart checkpoint selection and independent model replay."""
from pathlib import Path
import json,pickle
import numpy as np
import jax
from . import models3d as M
from . import training as T
root=Path(__file__).resolve().parents[1];out=root/'smokes/operator_continuation';out.mkdir(exist_ok=True)
x=np.random.default_rng(17).normal(size=(4,7,7,7,4));y=x[...,:1]*.3
spec=dict(kind='unet3d',width=2,levels=2,periodic=False)
cfg=dict(steps=3,wall_seconds=30,batch_size=2,seed=73,learning_rate=.001,validation_every=1,curve_every=1)
def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def checkpoint(path,data):path.write_bytes(pickle.dumps(jax.tree_util.tree_map(lambda a:np.asarray(a) if isinstance(a,jax.Array) else a,data)))
p=M.init_model(jax.random.PRNGKey(23),spec,4,1)
pred=np.asarray(M.apply_model(p,x,spec));initial=float(np.max(np.sqrt(np.sum((pred-y)**2,axis=(1,2,3))/np.sum(y*y,axis=(1,2,3)))))
selected,info=T.train(x,y,x,y,spec,cfg,out,dump,checkpoint,initial_params=p,initial_source={'test':'retained initial parameters'})
curve=json.loads((out/'curve.json').read_text());assert curve[0]['step']==0
np.testing.assert_allclose(curve[0]['validation_worst'],initial,rtol=1e-12)
assert info['best_validation_worst']<=initial
saved=pickle.loads((out/'best.pkl').read_bytes());pred=np.asarray(M.apply_model(jax.device_put(saved['params']),x,spec))
replay=float(np.max(np.sqrt(np.sum((pred-y)**2,axis=(1,2,3))/np.sum(y*y,axis=(1,2,3)))))
np.testing.assert_allclose(replay,info['best_validation_worst'],rtol=1e-12)
dump(out/'audit.json',dict(passed=True,initial_checkpoint_included=True,selected_checkpoint_replayed=True,selected_step=info['best_step']))
print('CONTINUATION_SMOKE_PASS',flush=True)
