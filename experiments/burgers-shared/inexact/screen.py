"""Explicit inexact G32/M64 proposal test after user authorization; prior gate failure preserved."""
import os,sys,time,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/burgers-shared-step'))
import pilot as B
from pilot import P,HC,H,C,Q,QS,T2,jax,jnp
OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
started=time.perf_counter();P.check_env({})
model=ROOT/'model';manifest=json.loads((model/'MODEL.json').read_text())
old=json.loads((ROOT/'expected/result.json').read_text())
assert HC.sha_file(model/'MODEL.json')==old['model_manifest_sha256']
for k in [16,32,64]:assert HC.sha_file(model/f'h{k}.npz')==manifest['heads'][str(k)]['sha256']
bank=P.Bank(1024,[0,128,256,384,512]);f=HC.Head(model/'h64.npz',bank.mdl);c=HC.Head(model/'h16.npz',bank.mdl)
fine=HC.HeadSetting(bank,'shared64',f,'gauss128',M=256,nrec=51,nstep=50)
fine.check('main',bank);df=fine.data['main'];cold=fine.cold
assert {k:B.sha(df[k]) for k in old['table_hashes']}==old['table_hashes']
hf,hc=H.head_fn(f.pf),H.head_fn(c.pf)
nl,nlJ=Q.tested_value('point',1024),Q.tested_jac('point',1024)
sch=T2.sched('BE',.005,1e-3,1e-9);cases=Q.cohort('dev6');U0=[jnp.asarray(Q.e.initial(1024,ph)) for ph in cases]
assert [B.sha(u) for u in U0]==old['initial_hashes']
assert np.array_equal(cases,old['cases'])
qref=fine.query(record=True);qdirect=fine.query(record=False)
_,res16,ev16=C.make_head_evolve(hc,nl,nlJ,16,c.trust)
_,res64,ev64=C.make_head_evolve(hf,nl,nlJ,64,f.trust)
configs=[('q32m64',64,32)]
coarse={};meta={}
for name,M,g in configs:
 if name=='full':coarse[name]=df;meta[name]=dict(fine.meta['main'],K=16,head_sha256=c.sha256,note='Physics table reused from fine target; coarse decoder is h16')
 else:
  cs=HC.HeadSetting(bank,name,c,f'gauss{g}',M=M,nrec=51,nstep=50);cs.check('main',bank)
  coarse[name]=cs.data['main'];meta[name]=cs.meta['main']
 for key in ['A','lam']:np.testing.assert_array_equal(coarse[name][key],df[key][:M])
assert np.array_equal(np.asarray(sch['co']),np.asarray(T2.BE_CO))
np.testing.assert_allclose(jax.vmap(hc)(c.Zsub[:10]),jax.vmap(hf)(jnp.pad(c.Zsub[:10],((0,0),(0,48)))),rtol=1e-11,atol=1e-12)
rep=dict(complete=False,job_id=os.environ.get('SLURM_JOB_ID'),commit=os.environ.get('SOURCE_COMMIT'),
 backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),gpu=P.gpu_info(),
 settings=dict(fine=old['settings'],coarse_configs=configs,budgets=[1,2],quadrature_gate=1e-3,paired_median_gate=.85,worst_case_gate=1.10,repetitions=7),
 model_sha256=HC.sha_file(model/'MODEL.json'),cases=np.asarray(cases).tolist(),initial_hashes=old['initial_hashes'],table_meta=meta,
 scope='User-authorized inexact q32m64 proposals despite prior conservative quadrature eligibility failure. True zero-tail D16, first16 restriction; fine target unchanged. Previous screen remains failed.',
 qualification={},components=[],records=[],timings=[],controls=[],gates={},tight_records=[],loss_agreement_tolerance=1e-3,field_agreement_tolerance=1e-3)
def save():
 (OUT/'result.json').write_text(json.dumps(B.clean(rep),indent=2,allow_nan=False)+'\n')

def compact(path,v,**extra):
 np.savez_compressed(path,W=v['W'],**{k:np.asarray(a) for k,a in v['stats'].items()},**extra)

refs={}
for ci in range(6):
 v=qref['fused'](U0[ci],float(cases[ci,4]),df,cold,sch);jax.block_until_ready(v);refs[ci]=v
 with np.load(ROOT/f'expected/direct_case{ci}.npz') as z:
  np.testing.assert_allclose(v['stats']['states'],z['states'],rtol=1e-9,atol=1e-11)
  np.testing.assert_array_equal(v['stats']['step_it'],z['step_it'])
 compact(OUT/f'direct_{ci}.npz',v)
 print('BASELINE',ci,flush=True)
save()

# Quadrature validation: same coarse modes and history, only quadrature changes.
# Whole residual/J and nonlinear term/tangent each checked to avoid diffusion masking.
@jax.jit
def panel(z,prev,nu,data):
 hist=(data['A']@hf(prev),data['A']@hf(prev),jnp.zeros(data['A'].shape[0]))
 args=(hist,nu,data,sch['co'],sch['dt'])
 r,J=ev16(z,args);cc=hc(z);Jh=jax.jacfwd(hc)(z);N,NJ=nlJ(cc,data,Jh)
 return r,J,N,NJ

def prefix(data,M):
 return dict(data,A=data['A'][:M],lam=data['lam'][:M],Psi=data['Psi'][:,:M],hmask=data['hmask'][:M])
for name,M,g in configs:
 values=[];references=[];checks=[]
 for ci in range(6):
  ss=refs[ci]['stats']['states']
  for k in [0,1,10,25,49]:
   z=ss[k,:16];prev=ss[k];nu=float(cases[ci,4])
   v=panel(z,prev,nu,coarse[name]);w=panel(z,prev,nu,prefix(df,M));jax.block_until_ready((v,w))
   v=[np.asarray(a) for a in v];w=[np.asarray(a) for a in w]
   ratios=[float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)) for a,b in zip(v,w)]
   checks.append(dict(case=ci,step=k,relative_errors=ratios));values.append(v);references.append(w)
 maxerr=np.max([q['relative_errors'] for q in checks],axis=0)
 passed=bool(np.all(np.isfinite(maxerr)) and np.max(maxerr)<=1e-3)
 rep['qualification'][name]=dict(passed=passed,metrics=['residual','Jacobian','advection','advection_tangent'],max_relative_errors=maxerr,rows=checks)
 np.savez_compressed(OUT/f'quadrature_{name}.npz',**{f'{label}_{kind}':np.stack([row[i] for row in data]) for label,data in [('candidate',values),('reference',references)] for i,kind in enumerate(['r','J','N','NJ'])})
 print('QUADRATURE',name,passed,maxerr,flush=True);save()

# Tight baseline uses the SAME production initial fit; only evolution tolerances change.
# It is a ROM comparator, not a full-order truth trajectory.
tightq=C.make_head_query(f.pf,64,1024,f.trust,budget=2000,nrec=51,nstep=50,ic_budget=400,ic_gtol=1e-6)
tightsch=T2.sched('BE',.005,1e-7,0.)
tightcert=H.make_head_certifier(hf,nl);tightrefs={}
for ci in range(6):
 v=tightq['fused'](U0[ci],float(cases[ci,4]),df,cold,tightsch);jax.block_until_ready(v)
 np.testing.assert_array_equal(v['stats']['states'][0],refs[ci]['stats']['states'][0])
 gamma,rn=tightcert(v['stats']['states'],float(cases[ci,4]),df,tightsch);jax.block_until_ready((gamma,rn))
 assert np.isfinite(gamma).all() and np.all(np.asarray(gamma)<=1e-7) and int(v['stats']['nfail'])==0
 compact(OUT/f'tight_{ci}.npz',v,gamma_fresh=gamma,residual_norm_fresh=rn)
 tightrefs[ci]=v
 rep['tight_records'].append(dict(case=ci,initial_state_matches_production=True,gtol=1e-7,tolf=0.,fine_updates=int(v['stats']['it_sum']),final_loss=float(rn[-1]**2),fields_sha256=B.sha(v['fields'])))
 print('TIGHT',ci,flush=True);save()

# Small proposals deliberately need not be converged; a finite budget exit is allowed.
# Fine guard still requires strict decrease of original fine objective.
def candidate(budget,M,diagnostic=False,force_failure=False):
 lm=T2.make_lm(ev16,budget,c.trust,1e-10,'lu',True)
 def seed(warm,args,tol):
  hist,nu,data,co,dt=args;dc=data['coarse'];ch=tuple(a[:M] for a in hist)
  a,rn,it,reason,gn,rej,damping=lm(warm[:16],(ch,nu,dc,co,dt),tol,1e-3,jnp.asarray(1e-6))
  padded=jnp.pad(a,(0,48));initial=jnp.pad(warm[:16],(0,48))
  rc=T2.lspg_residual(nl,co,dt,nu,data,hist)
  r0,r1=rc(hf(warm)),rc(hf(padded))
  f0,f1=jnp.vdot(r0,r0),jnp.vdot(r1,r1)
  if diagnostic:
   rt=rc(hf(initial));ft=jnp.vdot(rt,rt)
  else:ft=jnp.asarray(0.)
  valid=jnp.all(jnp.isfinite(padded))&jnp.isfinite(rn)&jnp.isfinite(gn)&(reason!=3)&(not force_failure)
  take=valid&jnp.isfinite(f0)&jnp.isfinite(f1)&(f1<f0)&(f1<=(1-1e-8)*f0)
  # Original16 slots reused; column15 now stores objective after truncating, before solving.
  info=jnp.array([take,it,reason,gn,rej,rn,damping,f0,f1,1+it,3 if diagnostic else 2,jnp.linalg.norm(a-warm[:16]),jnp.linalg.norm(warm[16:]),0,valid,ft])
  return jnp.where(take,padded,warm),info,jnp.concatenate((warm,padded))
 return seed

arms={'direct':dict(data=df,rec=qref,timed=qdirect)}
for name,M,g in configs:
 # Explicit new inexact-model test: failure is recorded, not an eligibility condition here.
 for budget in [1,2]:
  arm=f'{name}_b{budget}';seed=candidate(budget,M);seedrec=candidate(budget,M,diagnostic=True);data=dict(df,coarse=coarse[name])
  arms[arm]=dict(data=data,rec=C.make_head_query(f.pf,64,1024,f.trust,nrec=51,nstep=50,seed=seedrec),timed=C.make_head_query(f.pf,64,1024,f.trust,nrec=0,nstep=50,seed=seed))
# Disabled callback and forced coarse failure must reproduce the original baseline.
for control in ['disabled','forced_failure']:
 seed=None if control=='disabled' else candidate(1,64,force_failure=True)
 q=C.make_head_query(f.pf,64,1024,f.trust,nrec=51,nstep=50,seed=seed)
 dd=df if seed is None else dict(df,coarse=coarse['q32m64'])
 v=q['fused'](U0[0],float(cases[0,4]),dd,cold,sch);jax.block_until_ready(v)
 np.testing.assert_allclose(v['stats']['states'],refs[0]['stats']['states'],rtol=1e-12,atol=1e-13)
 np.testing.assert_array_equal(v['stats']['step_it'],refs[0]['stats']['step_it'])
 compact(OUT/f'control_{control}.npz',v)
 rep['controls'].append(dict(name=control,passed=True))
cert=H.make_head_certifier(hf,nl);witness={};certified={};scales={}
for arm,sp in arms.items():
 # Compile before any timed block.
 B.timed(lambda:sp['timed']['fused'](U0[0],float(cases[0,4]),sp['data'],cold,sch))
 for ci in range(6):
  v=refs[ci] if arm=='direct' else sp['rec']['fused'](U0[ci],float(cases[ci,4]),sp['data'],cold,sch)
  jax.block_until_ready(v);st=v['stats'];states=st['states']
  gamma,rn=cert(states,float(cases[ci,4]),df,sch);z0,scale,ic=qdirect['init'](U0[ci],cold);jax.block_until_ready((gamma,rn))
  valid=bool(np.isfinite(gamma).all() and np.all((np.asarray(gamma)<=1e-3)|(np.asarray(rn)<=float(scale)*1e-9)) and int(st['nfail'])==0)
  np.testing.assert_allclose(gamma,st['step_gn'],rtol=1e-7,atol=1e-12)
  np.testing.assert_array_equal(states[0],refs[ci]['stats']['states'][0])
  if arm!='direct':
   si=np.asarray(st['seed_info']);codes=np.asarray(st['seed_codes']);take=si[:,0].astype(bool)
   assert np.all(codes[:,80:]==0) and np.all(si[take,8]<(1-1e-8)*si[take,7])
  else:si=np.zeros((50,16))
  compact(OUT/f'{arm}_{ci}.npz',v,gamma_fresh=gamma,residual_norm_fresh=rn)
  certified[arm,ci]=(np.asarray(gamma),np.asarray(rn));scales[ci]=float(scale)
  # Compare returned fields to baseline in this invocation, without treating it as tight accuracy.
  gap=float(jnp.linalg.norm(v['fields']-refs[ci]['fields'])/(jnp.linalg.norm(refs[ci]['fields'])+1e-300))
  rep['records'].append(dict(arm=arm,case=ci,valid=valid,initial_scale=float(scale),fine_updates=int(st['it_sum']),coarse_updates=int(si[:,1].sum()),accepted=int(si[:,0].sum()),field_gap_to_production=gap))
  w=sp['timed']['fused'](U0[ci],float(cases[ci,4]),sp['data'],cold,sch);jax.block_until_ready(w)
  np.testing.assert_array_equal(w['W'],v['W']);np.testing.assert_array_equal(w['fields'],v['fields'])
  assert int(w['stats']['it_sum'])==int(st['it_sum'])
  witness[arm,ci]=(B.sha(w['W']),B.sha(w['fields']),int(st['it_sum']))
 print('RECORDED',arm,flush=True);save()
rep['setup_compile_certify_seconds']=time.perf_counter()-started
rng=np.random.default_rng(20261029)
for block in range(7):
 for ci0 in rng.permutation(6):
  ci=int(ci0)
  for arm in rng.permutation(list(arms)):
   sp=arms[arm];QS.burn(.1)
   elapsed,v=B.timed(lambda:sp['timed']['fused'](U0[ci],float(cases[ci,4]),sp['data'],cold,sch))
   sig=(B.sha(v['W']),B.sha(v['fields']),int(v['stats']['it_sum']))
   assert sig==witness[arm,ci] and int(v['stats']['nfail'])==0
   # Returned residual norms belong to THIS timed invocation. No extra residual calls in timed code.
   denominator=1e-9*scales[ci]+1e-300
   rn_timed=np.asarray(v['stats']['step_tolratio'])*denominator
   gn_timed=np.asarray(v['stats']['step_gn'])
   np.testing.assert_allclose(rn_timed,certified[arm,ci][1],rtol=1e-7,atol=1e-12)
   np.testing.assert_allclose(gn_timed,certified[arm,ci][0],rtol=1e-7,atol=1e-12)
   assert np.all((gn_timed<=1e-3)|(rn_timed<=1e-9*scales[ci]))
   field_gap_tight=float(jnp.linalg.norm(v['fields']-tightrefs[ci]['fields'])/(jnp.linalg.norm(tightrefs[ci]['fields'])+1e-300))
   field_gap_production=float(jnp.linalg.norm(v['fields']-refs[ci]['fields'])/(jnp.linalg.norm(refs[ci]['fields'])+1e-300))
   filename=f'timed_{arm}_{ci}_{block}.npz'
   compact(OUT/filename,v,residual_norm_timed=rn_timed,initial_scale=scales[ci],residual_denominator=denominator)
   rep['timings'].append(dict(arm=str(arm),case=ci,block=block,seconds=elapsed,W_sha256=sig[0],fields_sha256=sig[1],fine_updates=sig[2],witness=filename,final_loss=float(rn_timed[-1]**2),field_gap_tight=field_gap_tight,field_gap_production=field_gap_production))
 print('TIMING',block,flush=True);save()
lookup={(t['arm'],t['case'],t['block']):t['seconds'] for t in rep['timings']}
for arm in arms:
 if arm=='direct':continue
 ratios=np.array([[lookup[arm,ci,b]/lookup['direct',ci,b] for b in range(7)] for ci in range(6)])
 med=np.median(ratios,axis=1);valid=all(r['valid'] for r in rep['records'] if r['arm']==arm)
 rep['gates'][arm]=dict(per_case_ratios=ratios,median_of_case_medians=float(np.median(med)),worst_case_median=float(np.max(med)),passed=bool(valid and np.median(med)<=.85 and np.max(med)<=1.10))
rep['accuracy_qualification']='Compared to tight fine ROM evolution with same productionIC, not full-order physical truth. Final residual losses evaluated on each trajectory own history.'
rep['complete']=True;rep['elapsed_seconds']=time.perf_counter()-started;save();print('COMPLETE',rep['elapsed_seconds'],flush=True)
