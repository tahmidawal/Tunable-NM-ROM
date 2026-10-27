"""One predeclared case5 adaptive check of the accepted finer quadrature witness; not a new sweep."""
import hashlib
import json
import numpy as np
import jax
import jax.numpy as jnp
import qn3
from time_run import adaptive,fielddist,historical,HERE

def run(cx):
    assert cx.cases==[5] and cx.cfg['scheme']=='CN' and cx.cfg['dt']==.01
    st=cx.setting('W1024');d=st.data('gl80');j=5;L=st.lean['L'];n0=float(jnp.linalg.norm(cx.U[j]))
    fs_pin=cx.cfg['fs80'];p=HERE/fs_pin['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==fs_pin['sha256']
    fs=np.load(p,allow_pickle=False)['W'];old=historical(cx,'W1024|gl64|ad|LSPG|CN|0.01|tight',j)
    assert old is not None,'existing gl64 tight anchor must be independently verified'
    old64,oldrecord=old
    outputs={};cert=qn3.make_certify('CN',.01,25)
    for level,gt in [('tight',1e-7),('tighter',1e-8)]:
        q,sch=adaptive(st,.01,gt,scheme='CN');o=jax.block_until_ready(q(cx.U[j],cx.nu[j],d,sch))
        W=np.asarray(o['W']);states=np.asarray(o['stats']['states']);g=np.asarray(cert(jnp.asarray(states),cx.nu[j],d))
        valid=bool(np.isfinite(W).all() and np.isfinite(states).all() and int(o['stats']['nfail'])==0 and np.isfinite(g).all() and np.all(g<=gt))
        outputs[level]=W
        cx.rep['runs'][level]=dict(valid=valid,gtol=gt,fresh_gamma=g.tolist(),nfail=int(o['stats']['nfail']),
            score=cx.score(W,st,j),FS80_to_AD80=fielddist(fs,W,L,n0),AD64_to_AD80=fielddist(old64,W,L,n0),
            iterations=int(o['stats']['it_sum']),existing_gl64_anchor=oldrecord,FS80_source=fs_pin)
        np.savez_compressed(cx.out/f'CN_gl80_{level}_c005.npz',W=W,states=states,fresh_gamma=g,
            step_it=np.asarray(o['stats']['step_it'])[:25],step_reason=np.asarray(o['stats']['step_reason'])[:25])
        cx.save()
    tt=fielddist(outputs['tight'],outputs['tighter'],L,n0)
    cx.rep.update(tight_tighter_distance=tt,anchor_usable=bool(tt<=2.5e-5 and all(r['valid'] for r in cx.rep['runs'].values())),
                  scope='case5-only adaptive check of an already admitted finer witness; not gl80 qualification')
    cx.save()
