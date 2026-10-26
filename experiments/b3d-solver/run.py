"""Bounded 3D solver/quadrature diagnostics; no full-population qualification claim."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'/'speed3d'))
sys.path.insert(0,str(HERE/'vendor'/'e0'))
import c3common as C3
import s3common as S
import qn3
import qn_policy
import t2core as T2
import refstore as RS
import diagcore as DC
import jax
import jax.numpy as jnp

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ah(a):
    a=np.ascontiguousarray(np.asarray(a));return hashlib.sha256(a.tobytes()).hexdigest()

def dump(p,x):
    C3.C.dump(p,C3.C.clean(x))

def provenance():
    p=json.loads((HERE/'PROVENANCE.json').read_text())
    for v in p['files']+p.get('derived_files',[]):
        rel=Path(v['destination']).relative_to('experiments/b3d-solver')
        assert sha(HERE/rel)==v['sha256'],rel
    assert RS.store_version()=='3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0'
    assert sha(HERE/'vendor/e0/store/STATUS.json')=='af7d824e94b42afb83753d6b11a0c4d71407cff0923ebee9c245eb8d51f81005'
    return sha(HERE/'PROVENANCE.json')

class Context:
    def __init__(self,cfg,out):
        self.cfg,self.out=cfg,Path(out);self.out.mkdir(parents=True,exist_ok=True)
        assert jax.default_backend()=='gpu'
        assert jax.config.x64_enabled and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
        print('jax_backend=gpu x64=True precision=highest',flush=True)
        self.t0=time.perf_counter(); self.prov=provenance()
        self.history=json.loads((HERE/'inputs/historical.json').read_text())
        assert sha(HERE/'inputs/historical.npz')==self.history['subset_sha256']
        self.hz=np.load(HERE/'inputs/historical.npz',allow_pickle=False)
        self.tab=C3.C.table(923801,64)
        assert self.tab['sha256']==next(iter(self.history['sources'].values()))['cohort']['table_sha256']
        self.cases=cfg['cases']; assert all(j in range(64) for j in self.cases)
        self.U={j:jax.block_until_ready(jnp.asarray(C3.C.initial_interior(65,self.tab,j))) for j in self.cases}
        self.nu={j:float(self.tab['nu'][j]) for j in self.cases}
        self.refs={};self.rep=dict(config=cfg,config_sha256=sha(cfg['_config_path']),source_commit=os.environ['SOURCE_COMMIT'],
            job_id=os.environ['SLURM_JOB_ID'],backend=jax.default_backend(),jax=jax.__version__,gpu=C3.gpu_info(),
            provenance_sha256=self.prov,cohort_sha256=self.tab['sha256'],complete=False,scope='diagnostic subset',
            setup={},runs={},restarts={},timing={},archive_identity={},gates={})
        self.banks={};self.tables={};self.settings={}

    def save(self):
        self.rep['elapsed_seconds']=time.perf_counter()-self.t0
        self.rep['memory']=jax.devices()[0].memory_stats()
        dump(self.out/'result.json',self.rep)

    def setting(self,name):
        if name not in self.settings:
            bname='M2' if name=='M2_512' else 'W1024'
            r=512 if name in ('M2_512','W512k8') else int(name[1:])
            t=time.perf_counter()
            if bname not in self.banks:
                self.banks[bname]=C3.load_bank(bname)
                # Prefixes must be explicit edges because TB.decode walks stored row blocks.
                self.tables[bname]=C3.Tables(65,self.banks[bname],[256,512,768,1024] if bname=='W1024' else [512])
            bk,tb=self.banks[bname],self.tables[bname]
            rules=S.load_rules(['gl40','gl56','gl64','gl80'])
            if name=='W512k8':
                # Only used for archived decode; identical bank prefix and existing Gram suffice.
                rules=S.load_rules(['gl40'])
            st=S.Setting(dict(name=name,bank=bname,Rp=r,a=3 if name=='M2_512' else 6),self.banks,self.tables,rules)
            self.settings[name]=st
            self.rep['setup'][name]=dict(seconds=time.perf_counter()-t,r=r,M=st.M,bank_sha256=bk['sha256'],trust=st.trust)
        return self.settings[name]

    def ref(self,j):
        if j not in self.refs:
            R,meta=RS.load(3,'val64',j,'ST',field='finest');status=RS.reference_status(3,'val64',j,'ST')
            assert R.shape==(6,63**3)
            assert np.allclose(R[0],np.asarray(self.U[j]),rtol=1e-12,atol=1e-14)
            self.refs[j]=(R,meta,status)
        return self.refs[j]

    def score(self,W,st,j):
        F=np.asarray(jax.block_until_ready(C3.TB.decode(jnp.asarray(W),st.lean)))
        R,meta,status=self.ref(j)
        n0=float(meta['judge']['n0']); e=np.linalg.norm(F-R,axis=1)/n0
        return dict(e_t=e.tolist(),e=float(e[1:].max()),eps_ref=status['eps_ref'],eligible=status['eligible'],
                    reference=f'3d/val64/ST/c{j:03d}',store_version=RS.store_version(),normalisation=n0)

    def historical(self,key,j):
        ids=list(self.hz['cases|'+key]);return self.hz[key][ids.index(j)]

    def identities(self,keys):
        for key in keys:
            name=key.split('|')[0];st=self.setting(name)
            for j in self.cases:
                if j not in self.history['entries'][key]['cases']:continue
                score=self.score(self.historical(key,j),st,j)
                old=self.history['entries'][key]['scores'][str(j)]
                delta=abs(score['e']-old['e'])
                rec=dict(score=score,archived_e=old['e'],delta=delta,passed=delta<=1e-11)
                self.rep['archive_identity'][f'{key}|case{j}']=rec
                assert rec['passed'],rec
                self.save()

def result_summary(o):
    out={}
    for k,v in o.items():
        if k!='W_all':out[k]=np.asarray(v)[:,0].tolist()
    out['finite']=bool(np.isfinite(np.asarray(o['W_all'])).all())
    return out

def make_q(st,policy):
    dt=.025;steps=10
    factory=qn3.make_qn_evolve if policy['qmax']==12 else qn_policy.make_qn_evolve
    assert policy['slow_threshold']==6
    extra={} if policy['qmax']==12 else dict(slow_threshold=policy['slow_threshold'])
    ev=factory(st.Rp,'BDF2',dt,steps,st.trust*policy['trust_factor'],policy['refresh'],
               g_in=policy['g_in'],q_max=policy['qmax'],**extra)
    @jax.jit
    def query(u0,nu,d):
        w0,_=C3.TB.project(u0,d)
        o=ev(w0[None],jnp.asarray([nu]),d)
        W=o['W_all'][::2,0]
        return dict(o,fields=C3.TB.decode(W,d))
    return query,ev

def dsolver(cx):
    cfg=cx.cfg;rule='gl40';Q={}
    for width in cfg['widths']:
        sn=f'W{width}';st=cx.setting(sn);d=st.data(rule)
        np.savez_compressed(cx.out/f'metric_{sn}.npz',L=np.asarray(st.lean['L']),
                            case_ids=np.asarray(cx.cases),n0=np.asarray([float(jnp.linalg.norm(cx.U[j])) for j in cx.cases]))
        for pol in cfg['policies']:
            key=pol['name'];q,ev=make_q(st,pol);Q[key]=q
            for j in cx.cases:
                rid=f'{sn}|{key}|case{j}'
                t=time.perf_counter();o=jax.block_until_ready(q(cx.U[j],cx.nu[j],d));elapsed=time.perf_counter()-t
                W=np.asarray(o['W_all'][:,0]);rec=result_summary({k:v for k,v in o.items() if k!='fields'})
                g=np.asarray(qn3.make_certify('BDF2',.025,10)(o['W_all'][:,0],cx.nu[j],d))
                rec.update(fresh_gamma=g.tolist(),nonstationary=int(np.sum(~(g<=1e-3))),first_seconds=elapsed,
                           score=cx.score(W[::2],st,j),coefficient_sha256=ah(W),
                           nonlinear_iterations=int(np.asarray(o['q']).sum()),jacobian_contractions=int(np.asarray(o['n_exec']).sum()),
                           residual_evaluations=None,gradient_evaluations=None,
                           unavailable_counts='original byte-identical QN kernel does not record residual/gradient call counts')
                rec['valid']=bool(rec['finite'] and not np.any(np.asarray(o['reason'])==3))
                akey=f'{sn}|gl40|ad|LSPG|BDF2|0.025|tight'
                if akey in cx.history['entries'] and j in cx.history['entries'][akey]['cases']:
                    anchor=cx.historical(akey,j)
                    diff=(jnp.asarray(W[::2])-jnp.asarray(anchor))@st.lean['L']
                    rec['distance_historical_tight']=float(jnp.linalg.norm(diff,axis=1)[1:].max()/jnp.linalg.norm(cx.U[j]))
                    rows=[rr for rr in cx.history['entries'][akey]['rows'] if rr['case']==j]
                    rec['historical_anchor_valid']=bool(len(rows)==1 and rows[0]['valid'] and rows[0]['steps'].get('nfail',1)==0)
                    rec['historical_anchor_source']=dict(key=akey,entry=cx.history['entries'][akey]['source'],
                        source=cx.history['sources'][cx.history['entries'][akey]['source']])
                    rec['distance_gate_diagnostic']=bool(rec['historical_anchor_valid'] and rec['distance_historical_tight']<=2.5e-4)
                if key=='baseline':
                    histkey=f'{sn}|gl40|qn|BDF2|0.025|k4'
                    target=cx.historical(histkey,j)
                    delta=np.linalg.norm(W[::2]-target)/(np.linalg.norm(target)+1e-300)
                    rec['historical_reproduction_relative']=float(delta)
                    assert delta<=1e-9,rec
                cx.rep['runs'][rid]=rec
                np.savez_compressed(cx.out/f'{sn}_{key}_c{j:03d}.npz',W_all=W,**{k:np.asarray(v) for k,v in o.items() if k not in ('W_all','fields')})
                cx.save()
                if key=='baseline':
                    bad=np.where(g>1e-3)[0].tolist()
                    if not bad:bad=[int(np.argmax(g))]
                    for k in bad[:cfg['max_restart_steps']]:
                        runs,diag=DC.check_step(jnp.asarray(W),int(k),cx.nu[j],d,int(np.asarray(o['pred'])[k,0]),st.trust)
                        rrec=dict(step=k,diagnostics={a:float(v) for a,v in diag.items()},kind='fresh restart; original cached carry not restored',levels={})
                        for level,rr in runs.items():
                            rc=np.asarray(rr['c']); delta=rc-W[k+1]
                            field=C3.TB.decode(jnp.asarray(delta)[None],st.lean)
                            dist=float(jnp.linalg.norm(field)/jnp.linalg.norm(cx.U[j]))
                            rrec['levels'][level]={a:np.asarray(v).tolist() for a,v in rr.items() if a not in ('c','trace')}
                            rrec['levels'][level]['baseline_distance']=dist
                            rr_c,rr_J=DC.step_functions()[1](rr['c'],(jnp.asarray(W),jnp.int32(k),jnp.float64(cx.nu[j]),d))
                            fresh=float(jnp.linalg.norm(rr_J.T@rr_c)/(jnp.linalg.norm(rr_J)*jnp.linalg.norm(rr_c)+1e-300))
                            gt=1e-7 if level=='tight' else 1e-8
                            verified=bool(np.isfinite(rc).all() and np.isfinite(fresh) and fresh<=gt and int(rr['reason'])==4)
                            rrec['levels'][level].update(fresh_gamma=fresh,verified=verified,gtol=gt)
                            np.savez_compressed(cx.out/f'{sn}_restart_c{j:03d}_k{k}_{level}.npz',c=rc,trace=np.asarray(rr['trace']))
                        diff=runs['tight']['c']-runs['tighter']['c']
                        rrec['tight_tighter_distance']=float(jnp.linalg.norm(C3.TB.decode(diff[None],st.lean))/jnp.linalg.norm(cx.U[j]))
                        rrec['anchor_usable']=bool(all(x['verified'] for x in rrec['levels'].values()) and rrec['tight_tighter_distance']<=2.5e-5)
                        rrec['causal_interpretation_allowed']=rrec['anchor_usable']
                        assert float(diag['residual_agreement'])<1e-10 and float(diag['jvp_agreement'])<1e-10,rrec
                        cx.rep['restarts'][f'{sn}|case{j}|step{k}']=rrec;cx.save()
                del o
        # Paired timing: the coefficient-bearing full query output used for error is the timed output.
        rng=np.random.default_rng(20261026)
        for j in cx.cases:
            for pname in rng.permutation([p['name'] for p in cfg['policies'] if p['name']!='baseline']):
                vals=[]
                for rep in range(cfg['timing_reps']):
                    trip=[]
                    for arm in ['baseline',pname,'baseline']:
                        C3.burn(cfg['burn_seconds']);t=time.perf_counter()
                        o=jax.block_until_ready(Q[arm](cx.U[j],cx.nu[j],d));sec=time.perf_counter()-t
                        W=np.asarray(o['W_all'][:,0]);ref=cx.rep['runs'][f'{sn}|{arm}|case{j}']['coefficient_sha256']
                        assert ah(W)==ref
                        trip.append(dict(arm=arm,seconds=sec,sha256=ah(W),score=cx.score(W[::2],st,j)))
                    drift=trip[2]['seconds']/trip[0]['seconds']
                    for record in trip:
                        record['anchor_drift_ratio']=drift
                        record['timing_admitted']=bool(1/1.1<=drift<=1.1)
                    vals.append(trip)
                cx.rep['timing'][f'{sn}|{pname}|case{j}']=vals;cx.save()
        st.drop(rule);gc.collect();Q.clear();jax.clear_caches()
    cx.identities(cfg.get('archive_identity_keys',[]))

def dquad(cx):
    st=cx.setting('W1024');rule='gl80'
    # Raw point tables alone occupy ~25 GB; refuse devices without suitable total capacity.
    ms=jax.devices()[0].memory_stats();limit=ms.get('bytes_limit',0)
    assert limit>=100e9,('gl80 requires >=100 GB device memory',ms)
    d=st.data(rule)
    fine_values={}
    advbatch=jax.jit(qn3.adv)
    sp=('fs','CN',.01,1);q=S.make_query(sp,st)
    for j in cx.cases:
        o=jax.block_until_ready(q(cx.U[j][None],jnp.asarray([cx.nu[j]]),d))
        W=np.asarray(o['W'][:,0]);wall=np.asarray(o['W_all'][:,0])
        g=np.asarray(qn3.make_certify('CN',.01,25)(o['W_all'][:,0],cx.nu[j],d))
        old56=cx.historical('W1024|gl56|fs|CN|0.01|K1',j)
        old64=cx.historical('W1024|gl64|fs|CN|0.01|K1',j)
        metric=lambda x:float(jnp.linalg.norm(C3.TB.decode(jnp.asarray(x)[None],st.lean))/jnp.linalg.norm(cx.U[j]))
        dist=max(metric(x) for x in (W-old64)[1:])
        # Keep only small tested advection outputs before releasing the gl80 point tables.
        fine_values[j]=np.asarray(jax.block_until_ready(advbatch(jnp.asarray(old64),d)))
        rec=dict(fresh_gamma=g.tolist(),valid=S.run_valid(sp,o,0),dist_gl64_gl80=dist,
                 dist_gl56_gl80=max(metric(x) for x in (W-old56)[1:]),score=cx.score(W,st,j),
                 own_state_rho=None,note='diagnostic subset; no full-population qualification claim')
        cx.rep['runs'][f'W1024|gl80|case{j}']=rec
        np.savez_compressed(cx.out/f'W1024_gl80_c{j:03d}.npz',W_all=wall,W=W)
        cx.save()
    del o,d
    st.drop('gl80');gc.collect()
    d64=st.data('gl64')
    for j in cx.cases:
        old64=cx.historical('W1024|gl64|fs|CN|0.01|K1',j)
        coarse=np.asarray(jax.block_until_ready(advbatch(jnp.asarray(old64),d64)))
        fine=fine_values[j]
        norm=np.linalg.norm(fine,axis=1)
        num=np.linalg.norm(coarse-fine,axis=1)
        rho=np.divide(num,norm,out=np.full_like(num,np.inf),where=norm>0)
        rec=cx.rep['runs'][f'W1024|gl80|case{j}']
        rec.update(own_state_rho=rho.tolist(),rho_finite=bool(np.isfinite(rho).all()),
                   rho_gl64_own_states=True,zero_denominator_times=np.flatnonzero(norm==0).tolist())
        np.savez_compressed(cx.out/f'W1024_gl64_gl80_adv_c{j:03d}.npz',W=old64,adv_gl64=coarse,adv_gl80=fine,rho=rho)
        cx.save()
    st.drop('gl64');del d64;gc.collect()
    cx.identities(['W1024|gl56|fs|CN|0.01|K1','W1024|gl64|fs|CN|0.01|K1'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();p=Path(args.config).resolve();cfg=json.loads(p.read_text());cfg['_config_path']=str(p)
    assert cfg['seed']==923801
    assert cfg['mode'] in ('solver','quadrature')
    cx=Context(cfg,args.out);cx.save()
    try:
        (dsolver if cfg['mode']=='solver' else dquad)(cx)
        cx.rep['execution_complete']=True
        cx.rep['diagnostic_anchors_usable']=bool(cx.rep['restarts']) and all(x['anchor_usable'] for x in cx.rep['restarts'].values())
        cx.rep['complete']=True;cx.save()
    except BaseException as e:
        cx.rep['failure']=repr(e);cx.save();raise

if __name__=='__main__':main()
