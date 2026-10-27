"""Bounded time-error controls with original recording adaptive LM and own-state quadrature checks."""
from pathlib import Path
import gc
import hashlib
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3
import qn3
import p1core as P1
import s3common as S

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def adaptive(st,dt,gtol):
    steps=int(round(.25/dt))
    ev=P1.make_evolve(C3.nl3,C3.nlJ3,'LSPG',st.Rp,st.trust,nrec=steps+1)
    sch=P1.sched('BDF2',dt,gtol,0.)
    @jax.jit
    def query(u,nu,d,sch):
        w0,s0=C3.TB.project(u,d)
        W,stats=ev(w0,nu,jnp.linalg.norm(s0),d,sch)
        return dict(W=W,stats=stats,fields=C3.TB.decode(W,d))
    return query,sch

def candidate(st,dt,policy):
    steps=int(round(.25/dt));keep=int(round(.05/dt))
    assert policy['qmax']==12
    ev=qn3.make_qn_evolve(st.Rp,'BDF2',dt,steps,st.trust*policy['trust_factor'],policy['refresh'],
                         q_max=policy['qmax'],g_in=policy['g_in'])
    @jax.jit
    def query(u,nu,d):
        w0,_=C3.TB.project(u,d)
        o=ev(w0[None],jnp.asarray([nu]),d)
        W=o['W_all'][::keep,0]
        return dict(o,W=W,fields=C3.TB.decode(W,d))
    return query

def fielddist(a,b,L,n0):
    return float(jnp.linalg.norm((jnp.asarray(a)-jnp.asarray(b))@L,axis=1)[1:].max()/n0)

def historical(cx,key,j):
    if key not in cx.history['entries'] or j not in cx.history['entries'][key]['cases']:return None
    rows=[r for r in cx.history['entries'][key]['rows'] if r['case']==j]
    assert len(rows)==1 and rows[0]['valid'] and rows[0]['steps'].get('nfail',1)==0
    gt=1e-8 if key.endswith('|tighter') else 1e-7
    gam=np.asarray(rows[0]['steps'].get('step_gn',[]),float)
    if gam.size==0 or not np.isfinite(gam).all() or not np.all(gam<=gt):return None
    return cx.historical(key,j),dict(reused=True,source_key=key,case=j,source=cx.history['sources'][cx.history['entries'][key]['source']],
        valid=True,states_available=False,recorded_step_gamma=rows[0]['steps'].get('step_gn'),iterations=rows[0]['steps']['it_sum'])

def run(cx):
    cfg=cx.cfg;st=cx.setting('W1024');D={rl:st.data(rl) for rl in cfg['rules']}
    assert cfg['rules']==['gl40','gl48','gl56']
    L=st.lean['L'];n0={j:float(jnp.linalg.norm(cx.U[j])) for j in cx.cases}
    np.savez_compressed(cx.out/'metric_W1024.npz',L=np.asarray(L),case_ids=np.asarray(cx.cases),n0=np.asarray(list(n0.values())))
    outputs={}
    for dt in cfg['dts']:
        steps=int(round(.25/dt));keep=int(round(.05/dt));cert=qn3.make_certify('BDF2',dt,steps)
        ad={lev:adaptive(st,dt,gt) for lev,gt in cfg['gtols'].items()};prod=candidate(st,dt,cfg['candidate_policy'])
        for rl,d in D.items():
            for j in cx.cases:
                saved={}
                for lev,gt in cfg['gtols'].items():
                    old=historical(cx,f'W1024|{rl}|ad|LSPG|BDF2|{dt:.8g}|{lev}',j)
                    if old:
                        W,rec=old
                    else:
                        q,sch=ad[lev];t=time.perf_counter();o=jax.block_until_ready(q(cx.U[j],cx.nu[j],d,sch));elapsed=time.perf_counter()-t
                        W=np.asarray(o['W']);stats=o['stats'];states=np.asarray(stats['states'])
                        g=np.asarray(cert(jnp.asarray(states),cx.nu[j],d))
                        valid=bool(np.isfinite(W).all() and np.isfinite(states).all() and int(stats['nfail'])==0
                                   and np.isfinite(g).all() and np.all(g<=gt))
                        rec=dict(reused=False,valid=valid,states_available=True,elapsed_with_first_compile=elapsed,
                                 fresh_gamma=g.tolist(),iterations=int(stats['it_sum']),exits=np.asarray(stats['exits']).tolist())
                        np.savez_compressed(cx.out/f'ad_{rl}_{dt:.8g}_{lev}_c{j:03d}.npz',W=W,states=states,fresh_gamma=g,
                                            step_it=np.asarray(stats['step_it'])[:steps],step_reason=np.asarray(stats['step_reason'])[:steps])
                    rec['score']=cx.score(W,st,j)
                    saved[lev]=W;outputs[(dt,rl,lev,j)]=W
                    cx.rep['runs'][f'W1024|{rl}|BDF2|{dt:.8g}|{lev}|case{j}']=rec;cx.save()
                tt=fielddist(saved['tight'],saved['tighter'],L,n0[j])
                if dt==.025 and rl=='gl40':
                    rp=HERE/'results/sdiag01/raw/output'/f'W1024_refresh_c{j:03d}.npz'
                    z=np.load(rp,allow_pickle=False);Wall=np.asarray(z['W_all']);W=Wall[::keep]
                    old=json.loads((rp.parent/'result.json').read_text())['runs'][f'W1024|refresh|case{j}']
                    assert hashlib.sha256(np.ascontiguousarray(Wall).tobytes()).hexdigest()==old['coefficient_sha256']
                    g=np.asarray(old['fresh_gamma']);qwork=old['nonlinear_iterations'];jwork=old['jacobian_contractions']
                    valid=old['valid'];reused=dict(path=str(rp.relative_to(HERE)),sha256=sha(rp))
                else:
                    o=jax.block_until_ready(prod(cx.U[j],cx.nu[j],d));Wall=np.asarray(o['W_all'][:,0]);W=np.asarray(o['W'])
                    g=np.asarray(cert(o['W_all'][:,0],cx.nu[j],d));valid=bool(np.isfinite(Wall).all() and not np.any(np.asarray(o['reason'])==3))
                    qwork=int(np.asarray(o['q']).sum());jwork=int(np.asarray(o['n_exec']).sum());reused=None
                dist=fielddist(W,saved['tight'],L,n0[j]);freshbad=int(np.sum(~(g<=1e-3)))
                rec=dict(valid=valid,reused=reused,fresh_gamma=g.tolist(),nonstationary=freshbad,steps=steps,
                         distance_tight=dist,tight_tighter_distance=tt,anchor_usable=bool(tt<=2.5e-5 and all(cx.rep['runs'][f'W1024|{rl}|BDF2|{dt:.8g}|{lev}|case{j}']['valid'] for lev in saved)),
                         iterations=qwork,jacobian_contractions=jwork,score=cx.score(W,st,j))
                outputs[(dt,rl,'candidate',j)]=W
                cx.rep['runs'][f'W1024|{rl}|BDF2|{dt:.8g}|candidate|case{j}']=rec
                np.savez_compressed(cx.out/f'candidate_{rl}_{dt:.8g}_c{j:03d}.npz',W=W,W_all=Wall,fresh_gamma=g)
                cx.save()
    rho=jax.jit(qn3.adv);cx.rep['quadrature']={}
    for dt in cfg['dts']:
        for lev in ['tight','tighter','candidate']:
            for j in cx.cases:
                W40,W48,W56=[outputs[(dt,rl,lev,j)] for rl in cfg['rules']]
                v40=np.asarray(rho(jnp.asarray(W40),D['gl40']));v48=np.asarray(rho(jnp.asarray(W40),D['gl48']))
                den=np.linalg.norm(v48,axis=1);num=np.linalg.norm(v40-v48,axis=1)
                rr=np.divide(num,den,out=np.full_like(num,np.inf),where=den>0)
                rr=np.where((den==0)&(np.linalg.norm(v40,axis=1)==0),0.,rr)
                dist=fielddist(W40,W48,L,n0[j]);finer=fielddist(W48,W56,L,n0[j])
                valid=all(cx.rep['runs'][f'W1024|{rl}|BDF2|{dt:.8g}|{lev}|case{j}']['valid'] for rl in cfg['rules'])
                cx.rep['quadrature'][f'{dt:.8g}|{lev}|case{j}']=dict(target='gl40',successors=['gl48','gl56'],distance=dist,finer_distance=finer,
                    rho=rr.tolist(),all_rungs_valid=valid,
                    quadrature_subset_pass=bool(valid and np.isfinite(rr).all() and rr.max()<=.116 and dist<=2.5e-4 and finer<=2.5e-4))
                np.savez_compressed(cx.out/f'rho_gl40_{dt:.8g}_{lev}_c{j:03d}.npz',W=W40,adv_gl40=v40,adv_gl48=v48,rho=rr)
                cx.save()
    cx.rep['scope']='fixed four-case time diagnostics; not full-population qualification'
    cx.rep['execution_complete']=True;cx.rep['complete']=True;cx.save()
