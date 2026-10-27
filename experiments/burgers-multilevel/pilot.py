"""All dev6, same-invocation common-target costs. No truth/cohort files opened."""
import os,time,json,hashlib,sys,subprocess
from pathlib import Path
import core as C
from core import P,jax,jnp,np
import p1common as B
Q=B.Q
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
def sha(a):return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()
def clean(v):return B.QS.clean(v)
def save(name,v):(OUT/name).write_text(json.dumps(clean(v),indent=2))
assert jax.default_backend()=='gpu'
assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest' and jax.config.jax_enable_x64
print('jax_backend=gpu f64=True highest',flush=True)
t0=time.perf_counter()
bank=B.Bank(256,[0,128,256])
st=B.Setting(bank,'fine',256,'gauss128',M=1024,nrec=51,nstep=50)
data=st.data[('main','LSPG')];cold=st.cold
setup=time.perf_counter()-t0
qs={};original=None;compile_times={}
for arm in ('direct','nested','twolevel'):
    qs[arm],original=C.make_query(Q,256,128,256,st.trust,arm)
nl,nlJ=Q.tested_value('point',256),Q.tested_jac('point',256)
cert=P.make_certifier(nl,256)
ph=Q.cohort('dev6')
U=[jnp.asarray(Q.e.initial(256,p)) for p in ph]
N0=[float(jnp.linalg.norm(u)) for u in U]
# Full native Gram for every internal-step norm, not only six stored outputs.
t=time.perf_counter();G=Q.H.bank_apply(data['G'],jnp.eye(256));Gram=jax.block_until_ready(G.T@G);del G
setup+=time.perf_counter()-t
@jax.jit
def distances(X,Y,n0,gram):
    d=X-Y
    return jnp.sqrt(jnp.maximum(jnp.einsum('si,ij,sj->s',d,gram,d),0))/n0
# Constructed tables saved BEFORE any solve; exact runtime pins.
arr={k:np.asarray(data[k]) for k in ('A','lam','Gq','Gs','Psi')}
arr.update(Gram=np.asarray(Gram),kx=st.kx,ky=st.ky,physical=ph)
for i,v in enumerate(cold):
    if hasattr(v,'shape'):arr['cold_'+str(i)]=np.asarray(v)
np.savez_compressed(OUT/'tables.npz',**arr)
meta=dict(setup_seconds=setup,job_id=os.environ.get('SLURM_JOB_ID'),gpu=B.gpu_info(),trust=st.trust,
          fork='93f399faa',source_commit=os.environ.get('SOURCE_COMMIT'),L=256,N=257,r=256,rc=128,M=1024,m=16384,
          scheme='CNR',dt=.005,steps=50,cohort='dev6',seeds=[7090702,911702],physical=ph,
          table_hashes={k:sha(v) for k,v in arr.items()},scope='algebraic archived-objective diagnostic; no physical truth claim')
save('metadata.json',meta)
def call(arm,ci,tol):
    return qs[arm](U[ci],ph[ci,-1],data,cold,P.sched('CNR',.005,tol,0.))
for arm in qs:
    t=time.perf_counter();jax.block_until_ready(call(arm,0,1e-3));compile_times[arm]=time.perf_counter()-t
save('compile.json',compile_times)
# Small equation/transfer/negative controls precede accepted numerical results.
v=jax.block_until_ready(call('direct',0,1e-3))
o=jax.block_until_ready(original['fused'](U[0],ph[0,-1],data,cold,P.sched('CNR',.005,1e-3,0.)))
w=v['w0'];co=jnp.asarray(P.BE_CO);dt=.005;nu=ph[0,-1]
hist=(data['A']@w,data['A']@w,jnp.zeros(1024));args=(hist,nu,data,co,dt)
f=C.step_factory(nl,nlJ,256,128,st.trust,'twolevel')
r,j=f['ej'](w,args);ja=jax.jacfwd(lambda x:f['res'](x,args))(w)
z=jnp.zeros(128);rr,jc=f['cej'](z,(args,w))
padded=jnp.pad(w[:128],(0,128)); Gc=Q.BK.prefix(bank.Grot,bank.edges,128)
errtransfer=Q.H.bank_apply(data['G'],padded)-Q.H.bank_apply(Gc,w[:128])
base=float(jnp.linalg.norm(Q.H.bank_apply(Gc,w[:128])))
noop,_=C.guarded(w,w,f['res'],args)
nanback,ac=C.guarded(w,jnp.full_like(w,jnp.nan),f['res'],args)
upback,acup=C.guarded(w,w+1e4*jnp.ones_like(w),f['res'],args)
controls=dict(baseline_max=float(jnp.max(jnp.abs(v['stats']['states']-o['stats']['states']))),
 jac_rel=float(jnp.linalg.norm(j-ja)/jnp.linalg.norm(ja)),coarse_jac_rel=float(jnp.linalg.norm(jc-ja[:,:128])/jnp.linalg.norm(ja[:,:128])),
 coarse_gradient_rel=float(jnp.linalg.norm(jc.T@rr-(ja.T@r)[:128])/(jnp.linalg.norm((ja.T@r)[:128])+1e-30)),
 transfer_rel=float(jnp.linalg.norm(errtransfer)/(base+1e-30)),no_op=float(jnp.max(jnp.abs(noop-w))),
 samewidth_transfer=float(jnp.max(jnp.abs(w-jnp.eye(256)@w))),
 nan_fallback=bool(jnp.array_equal(nanback,w)&~ac),uphill_fallback=bool(jnp.array_equal(upback,w)&~acup),
 zero_budget_gamma=float(jnp.linalg.norm(j.T@r)/(jnp.linalg.norm(j)*jnp.linalg.norm(r))))
controls['pass']=bool(controls['baseline_max']<=1e-12 and controls['jac_rel']<=1e-9 and controls['coarse_jac_rel']<=1e-9 and controls['coarse_gradient_rel']<=1e-9 and controls['transfer_rel']<=1e-12 and controls['no_op']==0 and controls['samewidth_transfer']==0 and controls['nan_fallback'] and controls['uphill_fallback'] and controls['zero_budget_gamma']>1e-3)
save('controls.json',controls); print('CONTROLS',controls,flush=True)
assert controls['pass']
anchors={}; anchorrows=[]
for ci in range(6):
    vals={}
    for tol in (1e-7,1e-8):
        v=jax.block_until_ready(call('direct',ci,tol));ss=v['stats']['states']
        ga,rn=jax.block_until_ready(cert(ss,ph[ci,-1],data,P.sched('CNR',.005,tol,0.)))
        vals[tol]=(v,ga,rn)
        np.savez_compressed(OUT/f'anchor_{ci}_{tol}.npz',states=np.asarray(ss),fields=np.asarray(v['fields']),gamma=np.asarray(ga),rn=np.asarray(rn))
    d=np.asarray(distances(vals[1e-7][0]['stats']['states'],vals[1e-8][0]['stats']['states'],N0[ci],Gram))
    ok=bool(np.isfinite(d).all() and d.max()<=1e-6 and np.all(np.asarray(vals[1e-7][1])<=1e-7+1e-12) and np.all(np.asarray(vals[1e-8][1])<=1e-8+1e-12))
    anchorrows.append(dict(case=ci,max_distance=d.max(),tight_max_gamma=np.max(np.asarray(vals[1e-7][1])),tighter_max_gamma=np.max(np.asarray(vals[1e-8][1])),accepted=ok))
    anchors[ci]=vals[1e-8]
    save('anchors.json',anchorrows);print('ANCHOR',anchorrows[-1],flush=True)
# Independent fixed-tight-history coarse probes (first and last step of every dev case).
probes=[]
for ci in range(6):
    states=anchors[ci][0]['stats']['states'];nu=ph[ci,-1]
    for k in (0,49):
        co=jnp.asarray(P.BE_CO if k<2 else P.SCHEMES['CNR'][:5])
        wn,wm,wp=states[k],states[max(0,k-1)],states[max(0,k-2)]
        hist=(data['A']@wn,data['A']@wm,nl(wn,data)+nu*data['lam']*(data['A']@wn))
        args=(hist,nu,data,co,.005)
        for arm in ('nested','twolevel'):
            fn=C.step_factory(nl,nlJ,256,128,st.trust,arm)
            cand=jnp.stack([wn,2*wn-wm,3*wn-3*wm+wp if k>=2 else 2*wn-wm])
            wi=cand[jnp.argmin(jax.vmap(lambda c:jnp.linalg.norm(fn['res'](c,args)))(cand))]
            chosen,_,ss=jax.jit(fn['stage'])(wi,args,0.,1e-5,1e-6)
            before,pre,candidate,after=ss[5:9]
            ds=np.asarray(distances(jnp.stack([before,pre,candidate,after]),jnp.broadcast_to(states[k+1],(4,256)),N0[ci],Gram))
            norms=[float(jnp.linalg.norm(fn['res'](x,args))) for x in (before,pre,candidate,after)]
            _,J=fn['ej'](pre,args);sv=np.linalg.svd(np.asarray(J),compute_uv=False);sc=np.linalg.svd(np.asarray(J[:,:128]),compute_uv=False)
            probes.append(dict(case=ci,step=k,arm=arm,distances=ds,residual_norms=norms,accepted=bool(ss[4]),cond_fine=sv[0]/sv[-1],cond_coarse=sc[0]/sc[-1]))
save('probes.json',probes)
@jax.jit
def common_norms(states,an,nu,data,sch):
    def common(k):
        co=jnp.where(k<2,jnp.asarray(P.BE_CO),sch['co']);wn=an[k];wm=an[jnp.maximum(k-1,0)]
        hi=(data['A']@wn,data['A']@wm,nl(wn,data)+nu*data['lam']*(data['A']@wn))
        return jnp.linalg.norm(P.lspg_residual(nl,co,.005,nu,data,hi)(states[k+1]))
    return jax.lax.map(common,jnp.arange(50))
# Burn compiled first; all timed invocations keep returned numerical arrays.
B.QS.burn(.2)
order=[(rep,ci,arm,tol) for rep in range(3) for ci in range(6) for arm in qs for tol in (1e-3,1e-5)]
np.random.default_rng(20261027).shuffle(order);save('order.json',order)
rows=[]
for idx,(rep,ci,arm,tol) in enumerate(order):
    B.QS.burn(.1);t=time.perf_counter();v=jax.block_until_ready(call(arm,ci,tol));elapsed=time.perf_counter()-t
    states=v['stats']['states'];sch=P.sched('CNR',.005,tol,0.)
    gam,rn=jax.block_until_ready(cert(states,ph[ci,-1],data,sch))
    d=np.asarray(distances(states,anchors[ci][0]['stats']['states'],N0[ci],Gram))
    # Same states evaluated on common anchor histories, separate from own-history certificates.
    an=anchors[ci][0]['stats']['states']
    cn=common_norms(states,an,ph[ci,-1],data,sch)
    fn=f'inv_{idx:03d}.npz';stats={k:np.asarray(vv) for k,vv in v['stats'].items()}
    np.savez_compressed(OUT/fn,fields=np.asarray(v['fields']),states=np.asarray(states),gamma=np.asarray(gam),rn=np.asarray(rn),common_rn=np.asarray(cn),distances=d,**{'stat_'+k:vv for k,vv in stats.items() if k!='states'})
    bar=2.5e-4 if tol==1e-3 else 2.5e-6
    valid=bool(anchorrows[ci]['accepted'] and np.isfinite(d).all() and np.isfinite(np.asarray(v['fields'])).all() and np.all(np.asarray(gam)<=tol+1e-12) and d.max()<=bar)
    mw=stats['mlwork'];steps=stats['step_it']
    rows.append(dict(index=idx,rep=rep,case=ci,arm=arm,tol=tol,seconds=elapsed,max_distance=d.max(),max_gamma=np.max(np.asarray(gam)),accepted=valid,fine_iterations=int(steps.sum()+mw[:,0].sum()),coarse_iterations=int(mw[:,2].sum()),fine_jacobians=int((steps+1).sum()+mw[:,5].sum()),coarse_jacobians=int(mw[:,6].sum()),accepted_coarse=int(mw[:,4].sum()),file=fn,sha256=hashlib.sha256((OUT/fn).read_bytes()).hexdigest()))
    save('results.json',rows);print('INV',idx,arm,ci,tol,elapsed,valid,flush=True)
(OUT/'COMPLETE').write_text('complete\n')
