"""2D data/bank adapter. No reconstruction routine receives an IC descriptor or reference.

The original historical generator/feature source is vendored without edits. Imports are
ordered intentionally: smoothtrain pins its own historical sep_common before engines imports it.
"""
from __future__ import annotations
import gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SMOOTH=HERE/'vendor/experiments/jcp-smooth-bank'
REF=HERE/'vendor/experiments/jcp-references'
sys.path.insert(0,str(SMOOTH))
import smoothtrain as ST
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
sys.path.insert(0,str(REF))
import refcore as RC
import cohorts as CH

LF=256
COHORTS={'dev6':6,'val32':32}

def log(*a): print(*a,flush=True)
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def sha_array(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def write_json(p,d): Path(p).write_text(json.dumps(d,indent=2,default=lambda v: v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
def sync(x): return jax.block_until_ready(x)
def runtime():
    assert jax.default_backend()=='gpu'
    assert jax.config.jax_enable_x64
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest'
    assert Path(ST.sc.__file__).resolve()==(SMOOTH/'deps/sep_common.py').resolve()
    assert (ST.bf.SEED,ST.bf.N_TRAIN,ST.bf.N_VAL)==(0,512,64), 'historical generator environment changed'
    assert len(os.environ.get('SOURCE_COMMIT',''))==40, 'SOURCE_COMMIT must identify frozen source'
    return dict(backend=jax.default_backend(),f64=bool(jax.config.jax_enable_x64),precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],gpu=str(jax.devices()[0]),gpu_kind=jax.devices()[0].device_kind,jax=jax.__version__,job_id=os.environ.get('SLURM_JOB_ID'),commit=os.environ.get('SOURCE_COMMIT'))

def verify_inputs():
    manifest=json.loads((HERE/'INPUTS.json').read_text())
    for r in manifest['records']:
        p=HERE/r['path']
        if sha_file(p)!=r['sha256']: raise RuntimeError('input checksum failed '+str(p))
    return manifest

def nodes(L):
    a=np.arange(1,L,dtype=float)/L
    return np.stack(np.meshgrid(a,a,indexing='ij'),-1).reshape(-1,2)

def indices(Lc,Lf=LF):
    assert Lf%Lc==0
    i=np.arange(1,Lc)*(Lf//Lc)-1
    return (i[:,None]*(Lf-1)+i[None,:]).reshape(-1)

def load_reference(cohort,case):
    if cohort not in COHORTS or not 0<=case<COHORTS[cohort]: raise ValueError('cohort is not exposed screen cohort')
    stem=HERE/f'inputs/refstore/2d/{cohort}/ST/c{case:03d}'
    meta=json.loads(stem.with_suffix('.json').read_text())
    if meta['outcome']!='accepted': raise RuntimeError('ineligible reference '+str(stem))
    with np.load(stem.with_suffix('.npz')) as z:
        levels=np.stack([z[k][:,1:-1,1:-1].reshape(6,-1) for k in meta['final']])
    return levels,meta

@jax.jit
def _features(params,x): return ST.sc.features(params,x)

def bank_values(name,L,chunk=4096):
    p,_,cfg=ST.sc.load_pkl(HERE/'inputs'/name/'bank.pkl')
    with np.load(HERE/'inputs'/name/'rotation.npz') as z: rotation=jnp.asarray(z['T'])
    out=[]; X=nodes(L)
    for s in range(0,len(X),chunk):
        out.append(np.asarray(_features(p,jnp.asarray(X[s:s+chunk]))@rotation))
    B=np.concatenate(out)
    if B.shape!=(len(X),512) or not np.isfinite(B).all(): raise RuntimeError('bank eval invalid')
    return B,cfg

@jax.jit
def dst1(a):
    n=a.shape[-1]; z=jnp.zeros(a.shape[:-1]+(1,),a.dtype)
    ext=jnp.concatenate([z,a,z,-a[...,::-1]],-1)
    return -jnp.fft.fft(ext,axis=-1).imag[...,1:n+1]/jnp.sqrt(2.*(n+1))
def dst2(a): return dst1(dst1(a).swapaxes(-1,-2)).swapaxes(-1,-2)

def continuous_modes(phi,Ls,Lt,block=32):
    """Physical-amplitude sine extension of native uncentered POD modes (not norm-preserving resampling)."""
    ms=Ls-1
    S=jnp.sqrt(2./Ls)*jnp.sin(jnp.pi*jnp.asarray(np.arange(1,Lt)/Lt)[:,None]*jnp.arange(1,Ls)[None,:])
    @jax.jit
    def f(P,S):
        C=dst2(P.T.reshape(-1,ms,ms))
        V=jnp.einsum('ik,rkl->ril',S,C)
        return jnp.einsum('ril,jl->rij',V,S).reshape(P.shape[1],-1).T
    out=[]
    for s in range(0,phi.shape[1],block): out.append(np.asarray(f(jnp.asarray(phi[:,s:s+block]),S)))
    return np.concatenate(out,axis=1)

def pod_randomized(U,rmax=512,oversample=64,power=3,seed=17291,block=8192):
    """Uncentered equal-state POD via a fixed randomized subspace; retain projected eigen-residual certificate."""
    ns,np_=U.shape; q=min(ns,np_,rmax+oversample)
    def StX(X): return jnp.concatenate([jnp.asarray(U[:,a:a+block]).T@X for a in range(0,np_,block)],axis=0)
    def SX(X):
        Z=jnp.zeros((ns,X.shape[1]),dtype=jnp.float64)
        for a in range(0,np_,block): Z=Z+jnp.asarray(U[:,a:a+block])@X[a:a+block]
        return Z
    t=time.perf_counter()
    Om=jnp.asarray(np.random.default_rng(seed).standard_normal((ns,q)))
    Q,_=jnp.linalg.qr(StX(Om),mode='reduced'); del Om
    for _ in range(power):
        Z,_=jnp.linalg.qr(SX(Q),mode='reduced')
        Q,_=jnp.linalg.qr(StX(Z),mode='reduced')
    C=SX(Q); _,sig,Vt=jnp.linalg.svd(C,full_matrices=False)
    Phi=Q@Vt.T[:,:rmax]
    coeff=SX(Phi); ss=np.asarray(sig[:rmax])
    total=float(sum(np.sum(U[:,a:a+block]**2) for a in range(0,np_,block)))
    residual=StX(coeff)-Phi*(jnp.asarray(ss)**2)[None,:]
    relative_eigen_residual=np.asarray(jnp.linalg.norm(residual,axis=0)/jnp.maximum(ss**2,1e-300))
    ortho=float(jnp.max(jnp.abs(Phi.T@Phi-jnp.eye(rmax))))
    captured=np.sum(np.asarray(coeff)**2,axis=0)
    info=dict(method='fixed randomized subspace iteration; approximate POD',rmax=rmax,oversample=oversample,power=power,seed=seed,uncentered=True,weighting='equal snapshot rows, uniform native interior spatial weights',total_energy=total,singular_values=ss,relative_eigen_residual=relative_eigen_residual,orthogonality_max=ortho,captured_energy=captured,projection_tail_fraction=(total-np.cumsum(captured))/total,seconds=time.perf_counter()-t)
    if not np.isfinite(np.asarray(Phi)).all() or ortho>1e-9: raise RuntimeError('POD numerical failure')
    return np.asarray(Phi),info

def regenerate_training(name,out):
    n={'base':256,'coarse':129}[name]
    old=json.loads((HERE/'inputs'/name/'train.json').read_text())
    pick,_=ST.state_pick(576,ST.bf.NUM_STEPS+1,16384,5,0)
    if sha_array(pick)!=old['data']['pick_sha256']: raise RuntimeError('training pick mismatch')
    t=time.perf_counter(); U,worst,fp=ST.build_data(n,576,pick)
    for k in ['sum','sumsq']:
        if abs(fp[k]/old['data']['fingerprint'][k]-1)>1e-9: raise RuntimeError('training fingerprint mismatch '+k)
    if U.shape!=tuple([old['data']['n_states'],old['data']['n_points']]): raise RuntimeError('training shape mismatch')
    path=out/f'training_{name}.npy'; np.save(path,U)
    info=dict(nodes=n,seed=0,trajectories=576,state_pick=pick.tolist(),state_pick_sha256=sha_array(pick),native_data_sha256=sha_array(U),archived_data_sha256=old['data']['data_sha256'],bitwise_matches_archived=(sha_array(U)==old['data']['data_sha256']),fingerprint=fp,max_fom_residual=worst,seconds=time.perf_counter()-t,file_sha256=sha_file(path),recipe_source_sha256=sha_file(SMOOTH/'smoothtrain.py'))
    write_json(out/f'training_{name}.json',info)
    return U,info

def disjointness():
    cx,cy,w,a,nu,_=ST.bf.sample_params(seed=0)
    train=np.stack([cx,cy,w,a,nu],axis=1)[:576]
    dev={k:CH.descriptors_2d(k) for k in COHORTS}
    # Units normalized by family ranges; last coordinate log viscosity.
    def trans(x):
        z=x.copy(); z[:,:2]=(z[:,:2]-.15)/.70; z[:,2]=(z[:,2]-.05)/.15; z[:,3]=(z[:,3]-.5)/1.5; z[:,4]=(np.log(z[:,4])-np.log(.01))/np.log(10.)
        return z
    tr=trans(train); out={}
    for k,p in dev.items():
        d=np.linalg.norm(trans(p)[:,None]-tr[None],axis=-1)
        if np.any(d==0): raise RuntimeError('training/evaluation duplicate')
        out[k]=dict(descriptors=p,nearest_training_distance=d.min(axis=1),sha256=sha_array(p))
    return out

def build_bases(out,meshes=(32,64,128),training=True):
    out.mkdir(parents=True,exist_ok=True); info={}
    for name in ['base','coarse']:
        name_meshes=list(meshes) if name=='base' else [128]
        rec=dict(kind='frozen_coordinate_bank',training_nodes={'base':256,'coarse':129}[name],input_hashes={fn:sha_file(HERE/'inputs'/name/fn) for fn in ['bank.pkl','rotation.npz','train.json']},meshes={})
        t=time.perf_counter()
        for L in [LF]+name_meshes:
            B,cfg=bank_values(name,L); np.save(out/f'bank_{name}_L{L}.npy',B)
            rec['meshes'][str(L)]=dict(shape=list(B.shape),sha256=sha_array(B))
        rec['seconds']=time.perf_counter()-t; info['bank_'+name]=rec
        if training:
            U,tinfo=regenerate_training(name,out)
            phi,pinfo=pod_randomized(U)
            np.save(out/f'pod_{name}_native.npy',phi)
            for L in [LF]+name_meshes:
                P=continuous_modes(phi,{'base':255,'coarse':128}[name],L)
                np.save(out/f'pod_{name}_L{L}.npy',P)
            write_json(out/f'pod_{name}.json',pinfo)
            info['pod_'+name]=dict(training=tinfo,pod=pinfo,extension='full tensor sine series at native interval count; physical amplitude preserved')
            del U,phi; gc.collect()
        write_json(out/'bases.json',info)
    return info

def coarse_fields(out,dt_ladder=(.0025,.00125,.000625,.0003125),tol=1e-5,meshes=(32,64,128),cohorts=('dev6','val32')):
    """Generate each actual coarse input once; retain all refinement levels and failures. Parent fixes policy."""
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for L in meshes:
        solve=RC.make_solver(2,L,adv='central',max_newton=30,lin_maxiter=400)
        for coh in cohorts:
            if coh not in COHORTS: raise ValueError(coh)
            params=CH.descriptors_2d(coh)
            for j,p in enumerate(params):
                ref,meta=load_reference(coh,j)
                n0=meta['judge']['n0']; norm_c=n0*L/LF
                U0=CH.ic_2d(L,p)[1:-1,1:-1].reshape(-1)
                fields=[]; levels=[]; accepted=False
                for dt in dt_ladder:
                    t=time.perf_counter(); ans=solve(U0,p[-1],dt,'cnr',1e-11,1e-12); sync(ans)
                    u=np.asarray(ans['out']); n=int(ans['stages']); residual=np.asarray(ans['res'])[:n]
                    complete=bool(n==RC.n_stages(dt,'cnr') and int(ans['units'])==2*int(round(.25/dt)))
                    good=bool(complete and np.isfinite(u).all() and np.isfinite(residual).all() and np.all(residual<=2e-11))
                    gap=None if not fields else float(np.max(np.linalg.norm(u[1:]-fields[-1][1:],axis=1))/norm_c)
                    fields.append(u); levels.append(dict(dt=dt,stages=n,units=int(ans['units']),complete=complete,minimum_field=float(np.min(u)),maximum_field=float(np.max(u)),max_residual=float(np.max(residual)),solver_pass=good,relative_refinement_gap=gap,seconds=time.perf_counter()-t,newton=np.asarray(ans['nwt'])[:n].tolist(),residual=residual.tolist(),compile_seconds=solve.compile_seconds))
                    if len(fields)>1 and good and levels[-2]['solver_pass'] and gap<=tol:
                        accepted=True; break
                key=f'{coh}_c{j:03d}_L{L}'
                np.savez(out/f'{key}.npz',fields=np.stack(fields),reference_levels=ref,desc=p)
                rec=dict(key=key,cohort=coh,case=j,L=L,accepted=accepted,temporal_rule='successive coarse fields gap <= tolerance; empirical not a rigorous time error bound',tolerance=tol,n0=n0,ref_eps=meta['judge']['eps_ref'],levels=levels,selected=len(fields)-1,input_sha256=sha_array(fields[-1]),sampled_input_sha256=sha_array(ref[-1][:,indices(L)]),reference_meta=meta)
                write_json(out/f'{key}.json',rec); summary.append({k:v for k,v in rec.items() if k not in ['levels','reference_meta']})
                write_json(out/'summary.json',summary)
                log('coarse',key,'admitted',accepted,'dt',levels[-1]['dt'],'gap',levels[-1]['relative_refinement_gap'])
        del solve; gc.collect()
    return summary
