"""Small controls for explicit larger stepping buffers, with unchanged no-HR equations."""
from pathlib import Path
import json
import numpy as np
import interface as I

jax,jnp,H,R=I.jax,I.jnp,I.H,I.R
rng=np.random.default_rng(20261026)
Phi=jnp.asarray(np.linalg.qr(rng.normal(size=(64,4)))[0])
data=dict(Phi=Phi,DPhi=H.dsum_cols(Phi,3,5),LPhi=H.lap_cols(Phi,3,5))
u=jnp.asarray(rng.normal(size=64)*.02);nu=jnp.float64(.025)
q,_,_,_,cert,sch=I.functions_for(.5,25,L=5)
new=jax.block_until_ready(q(u,nu,data,sch('CN',.01)))
old=jax.block_until_ready(R.make_query('none',3,5,'common',trust=.5)(u,nu,data,R.schedule(3,'CN',.01,'prod')))
diff=float(np.max(np.abs(np.asarray(new['W'])-np.asarray(old['W']))))
q,_,_,_,cert,sch=I.functions_for(.5,40,L=5)
sf=sch('BDF2',.00625);fine=jax.block_until_ready(q(u,nu,data,sf))
gamma=np.asarray(jax.block_until_ready(cert(fine['C'],nu,data,sf)))
slots=np.flatnonzero(np.asarray(sf['slot'])>=0)
slotdiff=float(np.max(np.abs(np.asarray(fine['W'])-np.asarray(fine['C'])[slots])))
counts=[]
for dt in [.025,1/60,.0125,.01,.00625]:
    n=int(round(.25/dt));_,_,_,_,_,s=I.functions_for(.5,n,L=5);ss=s('BDF2',dt)
    counts.append(dict(dt=dt,steps=int(ss['steps']),buffer=len(ss['slot'])-1))
out=dict(backend=jax.default_backend(),old_new_CN_W_maxabs=diff,fine_steps=int(sf['steps']),
         fine_buffer=int(fine['C'].shape[0]-1),fine_output_slot_difference=slotdiff,
         fine_finite=bool(np.isfinite(np.asarray(fine['C'])).all() and np.isfinite(gamma).all()),
         steps=counts,H_NMAX_unchanged=H.NMAX[3]==25)
out['pass']=bool(diff<=1e-12 and slotdiff<=1e-12 and out['fine_finite'] and out['fine_buffer']==40
                 and out['H_NMAX_unchanged'] and all(v['steps']==v['buffer'] for v in counts))
assert out['pass']
p=Path(__file__).parent/'checks/interface-controls.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(p.read_text(),flush=True)
