"""Fixed shared-prefix head training; all PDE targets are regenerated separately on GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/jcp-heads'))
sys.path.insert(0,str(ROOT/'experiments/separable-decoder'))
sys.path.insert(0,str(ROOT/'experiments/mr-burgers2d'))
import sep_hfit as HF
import engines as E
import jax
import jax.numpy as jnp
import optax

WIDTHS=(16,32,64)
STEPS=200000
SEED=20261028
BANK_SHA='18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def host(t):return jax.tree_util.tree_map(lambda x:np.asarray(x),t)
def write_json(p,x):
    q=Path(str(p)+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)

def prefix(p,k):
    return dict(h=[(p['h'][0][0][:k],p['h'][0][1])]+p['h'][1:],h_lin=p['h_lin'][:k])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--reference',required=True)
    ap.add_argument('--deadline-epoch',type=float,required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    assert jax.default_backend()=='gpu'
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    assert sha(a.checkpoint)==BANK_SHA
    started=time.perf_counter();d=np.load(a.data);ck=pickle.load(open(a.checkpoint,'rb'))
    ref=json.loads(Path(a.reference).read_text())
    assert np.array_equal(d['pick'],np.asarray(ck['cfg']['hfit_pick']))
    for key in ('pick','mu_tr','mu_te','traj_tr','t_tr','traj_te','t_te'):
        assert arrsha(d[key])==ref['array_hashes'][key],f'regenerated ordering/descriptor mismatch: {key}'
    for key in ('un2_tr','un2_te','fl2_tr','fl2_te','Gram'):
        x=np.asarray(d[key]);got=np.array([x.sum(),np.sum(x*x)])
        np.testing.assert_allclose(got,ref['moments'][key],rtol=1e-6,atol=1e-12)
    for key in ('C_tr','C_te','Gram','un2_tr','un2_te','fl2_tr','fl2_te'):
        assert np.isfinite(d[key]).all(),key
    Zold=jnp.asarray(ck['Z_tr'],dtype=jnp.float64)
    assert Zold.shape==(131072,16) and d['C_tr'].shape==(131072,512)
    assert 'hB' not in ck['params']
    mu=np.unique(d['mu_tr'],axis=0)
    dev6=np.concatenate((E.params_draw(7090702,4),E.params_draw(911702,2)))
    assert len(mu)==4608 and all(np.min(np.max(np.abs(mu-p),axis=1))>1e-9 for p in dev6)
    old=jax.tree_util.tree_map(jnp.asarray,dict(h=ck['params']['h'],h_lin=ck['params']['h_lin']))
    assert [x[0].shape for x in old['h']]==[(16,512),(512,512),(512,512)]
    Gram=jnp.asarray(d['Gram']);eps=1e-12*jnp.trace(Gram)/512
    L=jnp.linalg.cholesky(Gram+eps*jnp.eye(512,dtype=jnp.float64))
    Li=jax.scipy.linalg.solve_triangular(L,jnp.eye(512,dtype=jnp.float64),lower=True)
    A=jnp.asarray(d['C_tr'])@L;At=jnp.asarray(d['C_te'])@L
    un=jnp.asarray(d['un2_tr']);fl=jnp.asarray(d['fl2_tr'])
    inv=1/jnp.maximum(un,1e-300)
    qold=HF.to_q(old,L)
    key=jax.random.PRNGKey(SEED);key,k1,k2=jax.random.split(key,3)
    # Expansion in physical head coordinates preserves the existing map at zero tail.
    w0=old['h'][0][0];lin=old['h_lin']
    nw=.01*jnp.sqrt(jnp.mean(w0*w0))*jax.random.normal(k1,(48,512),dtype=jnp.float64)
    nl=.01*jnp.sqrt(jnp.mean(lin*lin))*jax.random.normal(k2,(48,512),dtype=jnp.float64)
    hp=dict(h=[(jnp.concatenate((w0,nw)),old['h'][0][1])]+old['h'][1:],h_lin=jnp.concatenate((lin,nl)))
    qp=HF.to_q(hp,L);Z=jnp.pad(Zold,((0,0),(0,48)))
    masks=jnp.asarray(np.arange(64)[None,:]<np.array(WIDTHS)[:,None],dtype=jnp.float64)
    rep=dict(complete=False,source_commit=os.environ.get('SOURCE_COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),
             source_ckpt_sha256=BANK_SHA,data_sha256=sha(a.data),reference_sha256=sha(a.reference),
             recipe=dict(seed=SEED,steps=STEPS,prefixes=list(WIDTHS),prefix_weights=[1/3]*3,batch=4096,lr=1e-3,
                         optimizer='dense optax.adam on shared parameters and all latent codes',weight_decay=0.,code_regularization=0.,
                         warmup=500,learning_rate_end=1e-5,initialization='incumbent16 with zero extra codes and .01-RMS new input/skip rows'),
             history=[],checks={},backend=jax.default_backend(),x64=True,precision='highest')
    report=out/'training.json'
    def save():write_json(report,rep)
    save()
    @jax.jit
    def loss_rows(q,z,aa,ii):
        def one(mask):
            diff=HF.head_apply(q,z*mask)-aa
            return jnp.mean(jnp.sum(diff*diff,axis=1)*ii)
        return jax.vmap(one)(masks)
    def metrics(q,z):
        losses=np.zeros(3);rels=[]
        for s in range(0,len(Zold),4096):
            e=min(s+4096,len(Zold));losses+=np.asarray(loss_rows(q,z[s:e],A[s:e],inv[s:e]))*(e-s)/len(Zold)
        for k in WIDTHS:
            r=np.asarray(HF.batched_rel(prefix(q,k),z[:,:k],A,fl,un))
            rels.append(dict(k=k,mean=float(r.mean()),max=float(r.max())))
        return losses,rels
    initial,initial_recon=metrics(qp,Z)
    oldloss=0.
    for s in range(0,len(Zold),4096):
        e=min(s+4096,len(Zold));rr=HF.head_apply(qold,Zold[s:e])-A[s:e]
        oldloss+=float(jnp.sum(jnp.sum(rr*rr,axis=1)*inv[s:e]))/len(Zold)
    np.testing.assert_allclose(initial,oldloss,rtol=1e-10,atol=1e-14)
    zcheck=Z[:16]
    np.testing.assert_allclose(HF.head_apply(qp,zcheck),HF.head_apply(qold,Zold[:16]),rtol=1e-11,atol=1e-12)
    J0=jax.jacfwd(lambda z:HF.head_apply(qp,z))(Z[0]);tailnorm=np.linalg.norm(np.asarray(J0[:,16:]),axis=0)
    assert np.all(np.isfinite(tailnorm)) and np.all(tailnorm>0)
    rep['checks']['initial']=dict(losses=initial.tolist(),old_loss=oldloss,reconstruction=initial_recon,
                                  tail_column_norm_min=float(tailnorm.min()),data_ordering_matches=True)
    schedule=optax.warmup_cosine_decay_schedule(0.,1e-3,500,STEPS,1e-5)
    opt=optax.adam(schedule);pz=(qp,Z);state=opt.init(pz)
    def loss(pz,Ab,ib,idx):
        q,z=pz
        return jnp.mean(loss_rows(q,z[idx],Ab,ib))
    @jax.jit
    def step(pz,st,kk,AA,ii):
        idx=jax.random.choice(kk,131072,shape=(4096,),replace=False)
        val,gr=jax.value_and_grad(loss)(pz,AA[idx],ii[idx],idx)
        updates,st=opt.update(gr,st,pz)
        return optax.apply_updates(pz,updates),st,val
    rep['setup_seconds']=time.perf_counter()-started;save()
    ttrain=time.perf_counter();probe_time=None;val=jnp.inf
    for i in range(STEPS):
        key,kk=jax.random.split(key);pz,state,val=step(pz,state,kk,A,inv)
        done=i+1
        if done==100:
            jax.block_until_ready(pz);probe_time=time.perf_counter()
        if done==2000:
            jax.block_until_ready(pz)
            rate=(time.perf_counter()-probe_time)/1900
            estimate=rate*(STEPS-done)+180
            remaining=a.deadline_epoch-time.time()
            rep['timing_probe']=dict(seconds_per_update=rate,estimated_remaining_seconds=estimate,reservation_remaining_seconds=remaining)
            save()
            if estimate>remaining:
                with open(out/'incomplete.pkl','wb') as f:pickle.dump(dict(qp=host(pz[0]),Z=host(pz[1]),steps_done=done),f)
                raise RuntimeError('Full prescribed training does not fit preparation reservation; stopped without admitting a partial model')
        if done%5000==0 or done==1:
            value=float(val);assert np.isfinite(value)
            rep['history'].append(dict(step=done,batch_loss=value,seconds=time.perf_counter()-ttrain));save()
            print('TRAIN',done,STEPS,value,round(time.perf_counter()-ttrain,2),flush=True)
            assert time.time()+90<a.deadline_epoch,'Preparation deadline reserve reached'
        if done%50000==0 and done<STEPS:
            with open(out/f'checkpoint_{done}.pkl','wb') as f:pickle.dump(dict(qp=host(pz[0]),Z=host(pz[1]),steps_done=done),f)
    qp,Z=pz;jax.block_until_ready(pz)
    final,recon=metrics(qp,Z)
    assert np.isfinite(final).all() and np.isfinite(np.asarray(Z)).all()
    rep['final']=dict(losses=final.tolist(),reconstruction=recon,train_seconds=time.perf_counter()-ttrain)
    # Fixed diagnostic only: nearest TRAINING field, then retained local fitting on exposed pipeline targets.
    @jax.jit
    def nearest(test,train):
        return jnp.argmin(jnp.sum(train*train,axis=1)[None,:]-2*(test@train.T),axis=1)
    idx=nearest(At,A);diag=[]
    for k in WIDTHS:
        err,zfit=HF.oracle(prefix(qp,k),At,jnp.asarray(d['fl2_te']),jnp.asarray(d['un2_te']),Z[idx,:k][:,None,:],150)
        diag.append(dict(k=k,mean=float(err.mean()),max=float(err.max()),note='best found from one field-selected training seed; diagnostic only'))
        np.savez_compressed(out/f'pipeline_fit_k{k}.npz',relative_error=err,codes=zfit)
    rep['pipeline_diagnostic']=diag
    hp=HF.to_h(qp,Li);full=host(hp);Zn=np.asarray(Z)
    roundtrip=float(jnp.linalg.norm(HF.head_apply(hp,Z[:16])@L-HF.head_apply(qp,Z[:16]))/(1+jnp.linalg.norm(HF.head_apply(qp,Z[:16]))))
    assert roundtrip<1e-10
    checks={};modeldir=out/'model';modeldir.mkdir()
    for k in WIDTHS:
        pp=prefix(hp,k);zz=Z[:16,:k];zz64=jnp.pad(zz,((0,0),(0,64-k)))
        vv=HF.head_apply(pp,zz);vf=HF.head_apply(hp,zz64)
        err=float(jnp.linalg.norm(vv-vf)/(1+jnp.linalg.norm(vf)));assert err<1e-12
        jacsmall=jax.jacfwd(lambda x:HF.head_apply(pp,x))(zz[0])
        jacfull=jax.jacfwd(lambda x:HF.head_apply(hp,x))(zz64[0])[:,:k]
        je=float(jnp.linalg.norm(jacsmall-jacfull)/(1+jnp.linalg.norm(jacfull)));assert je<1e-12
        sv=np.linalg.svd(np.asarray(jacsmall),compute_uv=False)
        assert np.isfinite(sv).all() and sv[-1]>1e-12*sv[0],f'dead/degenerate k{k} tangent'
        checks[str(k)]=dict(coefficient_identity=err,latent_jacobian_identity=je,jacobian_condition=float(sv[0]/sv[-1]))
    cfg=dict(ck['cfg']);cfg.update(k=64,nested_prefixes=list(WIDTHS),nested_seed=SEED,nested_steps=STEPS)
    newparams=dict(ck['params']);newparams.update(full)
    fullpath=out/'head64.pkl'
    with open(fullpath,'wb') as f:pickle.dump(dict(params=newparams,Z_tr=Zn,cfg=cfg),f)
    fullsha=sha(fullpath);heads={}
    for k in WIDTHS:
        pp=prefix(full,k);z=Zn[:,:k];dev=np.linalg.norm(z-z.mean(axis=0),axis=1);trust=.01*float(dev.max())
        args={f'h{i}_w':w for i,(w,b) in enumerate(pp['h'])};args.update({f'h{i}_b':b for i,(w,b) in enumerate(pp['h'])})
        path=modeldir/f'h{k}.npz'
        np.savez(path,h_lin=pp['h_lin'],n_layers=len(pp['h']),Zsub=z[::max(1,len(z)//8192)],trust=trust,
                 code_std=float(z.std()),code_max_dev=float(dev.max()),n_codes=len(z),k=k,seed0=SEED,steps=STEPS,
                 name=f'shared_k{k}',source_ckpt_sha256=BANK_SHA,npz_sha256=rep['data_sha256'],pkl_sha256=fullsha,**args)
        heads[str(k)]=dict(file=path.name,sha256=sha(path),k=k,trust=trust)
    model=dict(schema=1,complete=True,source_ckpt_sha256=BANK_SHA,training=rep['recipe'],heads=heads,prefix_checks=checks,
               data_sha256=rep['data_sha256'],training_source_commit=rep['source_commit'],full_checkpoint_sha256=fullsha,
               scope='shared N256-trained bank head; new L1024 solver admission required',reconstruction=recon,pipeline_diagnostic=diag)
    write_json(modeldir/'MODEL.json',model)
    rep['checks']['prefix']=checks;rep['checks']['whitening_roundtrip']=roundtrip
    rep['complete']=True;rep['elapsed_seconds']=time.perf_counter()-started;save()
    print('TRAINING_COMPLETE',rep['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
