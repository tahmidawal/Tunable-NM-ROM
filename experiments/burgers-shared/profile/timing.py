"""Synchronized stage diagnostic, NOT a decomposition of fused execution.
Same LM functions and mathematical trajectory, less monitor bookkeeping in split path.
Every timing has retained states and decisions. Full fused runs remain cost authority.
"""
import sys,os,time,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/burgers-shared-step'))
import pilot as B
from pilot import P,HC,H,C,Q,QS,T2,jax,jnp,make_active_residual,make_candidate

OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
started=time.perf_counter();P.check_env({})
model=ROOT/'model';manifest=json.loads((model/'MODEL.json').read_text())
assert manifest['complete'] and manifest['training']['steps']==200000
for k in (16,32,64):assert HC.sha_file(model/f'h{k}.npz')==manifest['heads'][str(k)]['sha256']
bank=P.Bank(1024,[0,128,256,384,512])
f=HC.Head(model/'h64.npz',bank.mdl);c=HC.Head(model/'h16.npz',bank.mdl)
fine=HC.HeadSetting(bank,'shared64',f,'gauss128',M=256,nrec=51,nstep=50)
fine.check('main',bank);d=fine.data['main'];cold=fine.cold
hf=H.head_fn(f.pf);nl,nlJ=Q.tested_value('point',1024),Q.tested_jac('point',1024)
seed=make_candidate(hf,nl,nlJ,c.trust)
qr={'direct':fine.query(record=True),'affine':C.make_head_query(f.pf,64,1024,f.trust,nrec=51,nstep=50,seed=seed)}
qt={'direct':fine.query(record=False),'affine':C.make_head_query(f.pf,64,1024,f.trust,nrec=0,nstep=50,seed=seed)}
sch=T2.sched('BE',.005,1e-3,1e-9)
cases=Q.cohort('dev6');u0=[jnp.asarray(Q.e.initial(1024,ph)) for ph in cases]
old=json.loads((ROOT/'expected/result.json').read_text())
assert B.sha(cases)==B.sha(old['cases'])
assert [B.sha(u) for u in u0]==old['initial_hashes']
assert HC.sha_file(model/'MODEL.json')==old['model_manifest_sha256']
assert {k:B.sha(d[k]) for k in old['table_hashes']}==old['table_hashes']
_,res,ev=C.make_head_evolve(hf,nl,nlJ,64,f.trust)
fine_lm=jax.jit(T2.make_lm(ev,600,f.trust,1e-10,'lu',True))
_,active_ev=make_active_residual(hf,nl,nlJ)
aux_lm=T2.make_lm(active_ev,600,c.trust,1e-10,'lu',True)

@jax.jit
def prep(zv,zp,zp2,k,nu,data,sch,scale):
    co=jnp.where(k<sch['nstart'],jnp.asarray(T2.BE_CO,jnp.float64),sch['co'])
    cv,cp=hf(zv),hf(zp);acn=data['A']@cv;acm=data['A']@cp
    fn=jax.lax.cond(co[4]!=0.,lambda:nl(cv,data)+nu*data['lam']*(data['A']@cv),lambda:jnp.zeros(256,jnp.float64))
    hist=(acn,acm,fn);args=(hist,nu,data,co,sch['dt'])
    ze=zv+(zv-zp);zqd=jnp.where(k>=2,3.*zv-3.*zp+zp2,ze)
    cand=jnp.stack((zv,ze,zqd));rs=jax.vmap(lambda zz:jnp.linalg.norm(res(zz,args)))(cand)
    zi=cand[jnp.argmin(jnp.where(jnp.isfinite(rs),rs,jnp.inf))]
    return zi,hist,co,sch['tolf']*scale

@jax.jit
def coarse(warm,args,tol):
    return aux_lm(warm[:16],(args,warm[16:]),tol,1e-3,jnp.asarray(1e-6))

@jax.jit
def guard(warm,args,tol,aux):
    active,rn,it,reason,gn,rej,damping=aux
    proposed=jnp.concatenate((active,warm[16:]))
    finite=jnp.all(jnp.isfinite(proposed))&jnp.isfinite(rn)&jnp.isfinite(gn)
    reason=jnp.where(finite,reason,jnp.int32(3))
    hist,nu,data,co,dt=args
    rc=T2.lspg_residual(nl,co,dt,nu,data,hist)
    r0,r1=rc(hf(warm)),rc(hf(proposed));f0,f1=jnp.vdot(r0,r0),jnp.vdot(r1,r1)
    solved=((reason==1)|(reason==4))&((gn<=1e-3)|(rn<=tol))
    take=(finite&solved&jnp.isfinite(f0)&jnp.isfinite(f1)&(f1<f0)&(f1<=(1-1e-8)*f0))
    return jnp.where(take,proposed,warm),jnp.array([take,it,reason,gn,rej,rn,damping,f0,f1])

@jax.jit
def bundle(states,outputs,infos,auxes):
    return jnp.stack(states),jnp.stack(outputs),jnp.stack(infos),jnp.stack(auxes)

PHASES=['initial_fit','predictor_history','coarse16','guard_transfer','fine64','output_decode','host_other']
def split(ci,arm):
    ticks={k:0. for k in PHASES};events=[]
    def run(name,fn):
        t=time.perf_counter();v=fn();jax.block_until_ready(v);elapsed=time.perf_counter()-t
        ticks[name]+=elapsed;events.append((name,elapsed));return v
    start=time.perf_counter()
    z,scale,ic=run('initial_fit',lambda:qr[arm]['init'](u0[ci],cold))
    zp=zp2=z;lam=jnp.asarray(1e-6,jnp.float64);states=[z];outputs=[];infos=[];auxes=[]
    nu=float(cases[ci,4])
    for k in range(50):
        warm,hist,co,tol=run('predictor_history',lambda:prep(z,zp,zp2,jnp.int32(k),nu,d,sch,scale))
        args=(hist,nu,d,co,sch['dt'])
        if arm=='affine':
            aux=run('coarse16',lambda:coarse(warm,args,tol))
            zi,info=run('guard_transfer',lambda:guard(warm,args,tol,aux))
        else:zi=warm;info=jnp.zeros(9)
        result=run('fine64',lambda:fine_lm(zi,args,tol,sch['gtol'],lam))
        zn,rn,it,reason,gn,rej,lam=result
        zp2,zp,z=zp,z,zn;states.append(z);infos.append(info)
        auxes.append(jnp.array([it,reason,gn,rej,rn]))
        if (k+1)%10==0:outputs.append(z)
    def finish():
        ss,ww,ii,aa=bundle(states,[states[0]]+outputs,infos,auxes)
        fields=qr[arm]['decode'](ww,d)
        return ss,ww,ii,aa,fields
    ss,ww,ii,aa,fields=run('output_decode',finish)
    elapsed=time.perf_counter()-start
    ticks['host_other']=elapsed-sum(ticks.values())
    return dict(seconds=elapsed,phases=ticks,events=events),dict(states=ss,W=ww,info=ii,step=aa,fields=fields,ic=ic,scale=scale)

rep=dict(complete=False,job_id=os.environ.get('SLURM_JOB_ID'),commit=os.environ.get('SOURCE_COMMIT'),
    backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),gpu=P.gpu_info(),
    scope='Synchronized split-stage diagnostic; no attribution of fused kernel durations.',
    caveats=['Each phase includes dispatch and synchronization. host_other contains Python/bookkeeping and untimed small device operations.',
    'Split path omits original alternation-monitor bookkeeping. Solver paths/decisions are checked; split times do not replace fused production times.'],
    model_sha256=HC.sha_file(model/'MODEL.json'),settings=old['settings'],cases=np.asarray(cases).tolist(),
    initial_hashes=old['initial_hashes'],table_hashes=old['table_hashes'],parity=[],timings=[],phase_names=PHASES)
def save():
    (OUT/'result.json').write_text(json.dumps(B.clean(rep),indent=2,allow_nan=False)+'\n')

def verify(v,ref):
    st=ref['stats'];maxgap=float(np.max(np.abs(np.asarray(v['states'])-np.asarray(st['states']))))
    np.testing.assert_allclose(v['states'],st['states'],rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(v['fields'],ref['fields'],rtol=1e-9,atol=1e-11)
    for idx,key in [(0,'step_it'),(1,'step_reason')]:np.testing.assert_array_equal(np.asarray(v['step'])[:,idx],st[key])
    assert int(np.asarray(v['step'])[:,3].sum())==int(st['rej'])
    np.testing.assert_allclose(np.asarray(v['step'])[:,2],st['step_gn'],rtol=1e-6,atol=1e-12)
    for key in ['ic_it','ic_reason','ic_idx']:np.testing.assert_array_equal(v['ic'][key],st[key])
    if 'seed_info' in st:
        info=np.asarray(st['seed_info'])[:,:9]
        for idx in [0,1,2,4]:np.testing.assert_array_equal(np.asarray(v['info'])[:,idx],info[:,idx])
        np.testing.assert_allclose(v['info'],info,rtol=1e-6,atol=1e-10)
    assert np.all((np.asarray(v['step'])[:,2]<=1e-3)|(np.asarray(v['step'])[:,4]<=1e-9*float(v['scale'])))
    return maxgap

refs={};signatures={}
for ci in range(6):
    for arm in ['direct','affine']:
        ref=qr[arm]['fused'](u0[ci],float(cases[ci,4]),d,cold,sch);jax.block_until_ready(ref)
        with np.load(ROOT/f'expected/{arm}_case{ci}.npz') as z:
            np.testing.assert_allclose(ref['stats']['states'],z['states'],rtol=1e-9,atol=1e-11)
            np.testing.assert_array_equal(ref['stats']['step_it'],z['step_it'])
        refs[ci,arm]=ref
        _,v=split(ci,arm);gap=verify(v,ref)
        rep['parity'].append(dict(case=ci,arm=arm,max_state_gap=gap))
        # Save one complete reference field array; repetitions carry states/decision witnesses.
        np.savez_compressed(OUT/f'reference_{arm}_{ci}.npz',states=ref['stats']['states'],W=ref['W'],fields=ref['fields'],
            step_it=ref['stats']['step_it'],step_reason=ref['stats']['step_reason'],step_gn=ref['stats']['step_gn'])
        w=qt[arm]['fused'](u0[ci],float(cases[ci,4]),d,cold,sch);jax.block_until_ready(w)
        np.testing.assert_array_equal(w['W'],ref['W'])
        signatures[ci,arm]=(B.sha(w['W']),B.sha(w['fields']))
    print('PARITY',ci,flush=True);save()
rep['setup_compile_check_seconds']=time.perf_counter()-started
rng=np.random.default_rng(20261029)
for block in range(7):
    for ci0 in rng.permutation(6):
        ci=int(ci0)
        for arm in rng.permutation(['direct','affine']):
            for mode in rng.permutation(['split','fused']):
                QS.burn(.1)
                if mode=='split':
                    timing,v=split(ci,arm);gap=verify(v,refs[ci,arm])
                    witness=f'{arm}_{ci}_{block}.npz'
                    np.savez_compressed(OUT/witness,states=v['states'],W=v['W'],step=v['step'],info=v['info'],**v['ic'])
                    entry=dict(**timing,max_state_gap=gap,witness=witness,fields_sha256=B.sha(v['fields']))
                else:
                    secs,w=B.timed(lambda:qt[arm]['fused'](u0[ci],float(cases[ci,4]),d,cold,sch))
                    assert (B.sha(w['W']),B.sha(w['fields']))==signatures[ci,arm]
                    entry=dict(seconds=secs,W_sha256=B.sha(w['W']),fields_sha256=B.sha(w['fields']))
                rep['timings'].append(dict(block=block,case=ci,arm=str(arm),mode=str(mode),**entry))
    print('BLOCK',block,flush=True);save()
rep['complete']=True;rep['elapsed_seconds']=time.perf_counter()-started;save()
print('PROFILE_COMPLETE',rep['elapsed_seconds'],flush=True)
