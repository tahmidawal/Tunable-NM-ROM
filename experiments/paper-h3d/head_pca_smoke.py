"""Check weighted-PCA initialization and full-field validation selection."""
import json,pickle
from pathlib import Path
import numpy as np
import jax
import common as C
import train as T
import rom as R
from audit_head import head as numpy_head
root=Path(__file__).resolve().parent;source=root/'smokes/tune02';out=root/'smokes/head_pca';out.mkdir(exist_ok=True)
cfg=dict(json.loads((source/'result.json').read_text())['config'])
cfg.update(head_steps=3,checkpoint_every=1,head_validation_fit_budget=30,head_validation_fit_tolerance=1e-6,head_validation_starts=4,
    representation_fit_budget=30,representation_fit_tolerance=1e-6,head_initialization='weighted_pca_linear_skip',head_include_initial_checkpoint=True)
b=pickle.loads((source/'bank.pkl').read_bytes());p=jax.device_put(b['params']);g=C.bank_at(p,cfg['train_intervals'],cfg['field_chunk'])@b['rotation']
training=C.dataset(cfg['train_intervals'],C.family(cfg['train_seed'],cfg['train_count']),cfg)
valid=C.dataset(cfg['train_intervals'],C.family(cfg['validation_seed'],cfg['validation_count']),cfg)
u=training.reshape(-1,len(g));target=u@g;norm=np.sum(u*u,axis=1);perp=np.maximum(norm-np.sum(target*target,axis=1),0.)
q,r=np.linalg.qr(g,mode='reduced');v=valid.reshape(-1,len(g));vt=v@q;vn=np.sum(v*v,axis=1)
validation=dict(matrix=r,target=vt,norm2=vn,perpendicular2=np.maximum(vn-np.sum(vt*vt,axis=1),0.),shape=valid.shape[:2])
model=T.train_head(target,norm,perp,3,cfg,out,validation)
for z in np.asarray(model['codes'])[:3]:
    value,derivative=numpy_head(model['params'],z)
    np.testing.assert_allclose(value,np.asarray(C.head(model['params'],jax.numpy.asarray(z))),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(derivative,np.asarray(jax.jacfwd(C.head,argnums=1)(model['params'],jax.numpy.asarray(z))),rtol=1e-12,atol=1e-12)
initial=pickle.loads((out/'head_K3_initial.pkl').read_bytes());info=initial['initialization']
independent=np.asarray(info['mean'])+np.asarray(initial['codes'])@(np.asarray(info['code_scale'])[:,None]*np.asarray(info['axes']).T)
np.testing.assert_allclose(np.asarray(C.head(jax.device_put(initial['params']),jax.device_put(initial['codes']))),independent,rtol=1e-12,atol=1e-12)
observed=[]
for starts in (4,8):
    prediction,stats,latents=R.best_found_fields(model,g,valid,{**cfg,'representation_fit_starts':starts},return_latents=True)
    for z,s,target in zip(latents.reshape(-1,3),stats.reshape(-1,5),vt):
        value,derivative=numpy_head(model['params'],z);scale=max(np.linalg.norm(target),1e-14)
        residual=(r@value-target)/scale;jac=r@derivative/scale
        gradient=np.linalg.norm(jac.T@residual)/max(np.linalg.norm(jac),1e-30)
        np.testing.assert_allclose(gradient,s[4],rtol=2e-6,atol=2e-11)
    error=max(C.metrics(a,b)['current_evolved'] for a,b in zip(prediction,valid));observed.append(error)
np.testing.assert_allclose(observed[0],model['info']['validation_evolved_worst'],rtol=2e-11,atol=2e-13)
assert observed[1]<=observed[0]+1e-10
curve=json.loads((out/'head_K3_curve.json').read_text());assert curve[0]['step']==0
assert model['info']['validation_evolved_worst']==min(row['validation_evolved_worst'] for row in curve if 'validation_evolved_worst' in row)
C.dump(out/'audit.json',dict(passed=True,pca_initialization_independently_replayed=True,initial_checkpoint_considered=True,
    analytic_numpy_head_jacobian_and_fit_gradient_passed=True,
    full_field_validation_matches_coordinate_selection=True,selected_step=model['info']['selected_step'],worst_errors_four_eight=observed))
print('HEAD_PCA_SMOKE_PASS',flush=True)
