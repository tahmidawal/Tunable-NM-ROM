"""Bounded head checkpoint-selection check against full reconstructed fields."""
import json,pickle
from pathlib import Path
import numpy as np
import jax
import common as C
import train as T
import rom as R
root=Path(__file__).resolve().parent;source=root/'smokes/tune02';out=root/'smokes/head_validation';out.mkdir(exist_ok=True)
original=json.loads((source/'result.json').read_text());cfg=dict(original['config'])
cfg.update(head_steps=3,checkpoint_every=1,head_validation_fit_budget=30,head_validation_fit_tolerance=1e-6,head_validation_starts=4,representation_fit_budget=30,representation_fit_tolerance=1e-6)
b=pickle.loads((source/'bank.pkl').read_bytes());p=jax.device_put(b['params'])
g=C.bank_at(p,cfg['train_intervals'],cfg['field_chunk'])@b['rotation']
training=C.dataset(cfg['train_intervals'],C.family(cfg['train_seed'],cfg['train_count']),cfg)
valid=C.dataset(cfg['train_intervals'],C.family(cfg['validation_seed'],cfg['validation_count']),cfg)
u=training.reshape(-1,len(g));target=u@g;norm=np.sum(u*u,axis=1);perp=np.maximum(norm-np.sum(target*target,axis=1),0.)
q,r=np.linalg.qr(g,mode='reduced');v=valid.reshape(-1,len(g));vt=v@q;vn=np.sum(v*v,axis=1)
validation=dict(matrix=r,target=vt,norm2=vn,perpendicular2=np.maximum(vn-np.sum(vt*vt,axis=1),0.),shape=valid.shape[:2])
model=T.train_head(target,norm,perp,3,cfg,out,validation)
prediction,stats=R.best_found_fields(model,g,valid,cfg)
errors=[C.metrics(a,b)['current_evolved'] for a,b in zip(prediction,valid)]
np.testing.assert_allclose(max(errors),model['info']['validation_evolved_worst'],rtol=2e-11,atol=2e-13)
curve=json.loads((out/'head_K3_curve.json').read_text());assert model['info']['validation_evolved_worst']==min(row['validation_evolved_worst'] for row in curve if 'validation_evolved_worst' in row)
C.dump(out/'audit.json',dict(passed=True,full_field_validation_matches_coordinate_selection=True,selected_step=model['info']['selected_step'],worst_error=max(errors),nonstationary_fits=int(np.count_nonzero(stats[...,2]!=1))))
print('HEAD_VALIDATION_PASS',model['info'],flush=True)
