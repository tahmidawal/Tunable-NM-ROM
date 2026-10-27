"""Tiny checks of subject adapters, fresh certificates and coefficient field distances."""
from pathlib import Path
import json
import numpy as np
import pod_subjects as P
from fit import conv

jax,jnp,H=P.jax,P.jnp,P.H
rng=np.random.default_rng(20261026);Phi=jnp.asarray(np.linalg.qr(rng.normal(size=(64,4)))[0])
data=dict(Phi=Phi,DPhi=H.dsum_cols(Phi,3,5),LPhi=H.lap_cols(Phi,3,5))
subs=P._subjects_from_data(data,.5,dict(test_only=True),L=5)
u=jnp.asarray(rng.normal(size=64)*.02);nu=jnp.float64(.025)
out=[]
for s in [subs[3],subs[2]]:
    q=jax.block_until_ready(s['run'](u,nu));c0,scale=jax.block_until_ready(s['initialize'](u))
    W,C,stats=jax.block_until_ready(s['evolve'](c0,nu,scale));F=jax.block_until_ready(s['decode'](W))
    diff=float(np.max(np.abs(np.asarray(F)-np.asarray(q['fields']))))
    v=s['validate'](u,nu,prod=q,need_tighter=True)
    full=float(np.linalg.norm(np.asarray(u)))
    fields_t=v['Wtight']@np.asarray(Phi).T
    direct=float(np.linalg.norm((np.asarray(q['fields'])-fields_t)[1:],axis=1).max()/full)
    err=abs(direct-v['prod_to_tight_distance'])
    assert diff<=1e-12 and err<=1e-12 and v['production_certificate']['steps']==s['steps']
    assert q['C'].shape[0]==s['steps']+1 and v['tight_tighter_distance'] is not None
    assert v['production_certificate']['valid'] and v['tight_certificate']['anchor_ok'] and v['tighter_certificate']['anchor_ok']
    out.append(dict(name=s['name'],steps=s['steps'],split_difference=diff,distance_difference=err,
                    prod_valid=v['production_certificate']['valid'],tight_anchor=v['tight_certificate']['anchor_ok'],
                    tighter_anchor=v['tighter_certificate']['anchor_ok']))
result=dict(backend=jax.default_backend(),test_grid_L=5,rank=4,subjects=out,pass_=True)
p=Path(__file__).parent/'checks/subjects-controls.json';p.write_text(json.dumps(result,indent=2,default=conv)+'\n')
print(p.read_text(),flush=True)
