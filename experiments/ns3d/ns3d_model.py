"""Learned periodic solenoidal vector bank + nonlinear coefficient head + q.

The spatial bank is learned by an MLP, not replaced with POD. POD is a separately
reported control and may initialize latent codes without defining bank columns.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pickle
import time
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import optax
import ns3d_fom as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/separable-decoder'))
sys.path.insert(0,str(ROOT/'experiments/ns2d'))
import sep_common as S
from ns2d_decoder import make_lm_fit, oracle_fit


def coords(n):
    x=np.arange(n,dtype=np.float64)/n
    return jnp.asarray(np.stack(np.meshgrid(x,x,x,indexing='ij'),axis=-1).reshape(-1,3))


def init(key,k_lat,r_feat,width=128,layers=2,n_ff=64):
    kb,kg,kh,kl=jax.random.split(key,4)
    # Deterministic low-to-high periodic modes, randomized order within equal radii.
    ks=np.array([(a,b,c) for a in range(-4,5) for b in range(-4,5)
                 for c in range(-4,5) if (a,b,c)!=(0,0,0)
                 and (a>0 or (a==0 and b>0) or (a==0 and b==0 and c>0))])
    ks=ks[np.argsort(np.sum(ks*ks,axis=1),kind='stable')][:n_ff]
    return dict(B=jnp.asarray(ks.T,dtype=jnp.float64),
                g=S.init_mlp(kg,[2*n_ff]+[width]*layers+[3*r_feat]),
                h=S.init_mlp(kh,[k_lat]+[width]*layers+[r_feat]),
                h_lin=.3*jax.random.normal(kl,(k_lat,r_feat),dtype=jnp.float64),
                out_scale=jnp.asarray(1.,dtype=jnp.float64))


def head(params,z):
    return S.head(params,z)


def bank(params,xy,geom,n):
    angles=2*jnp.pi*(xy@params['B'])
    ff=jnp.concatenate((jnp.sin(angles),jnp.cos(angles)),axis=-1)
    raw=params['out_scale']*S.apply_mlp(params['g'],ff)
    r=raw.shape[-1]//3
    columns=raw.reshape(n,n,n,3,r).transpose(4,3,0,1,2)
    cols=jax.vmap(F.project_field,in_axes=(0,None))(columns,geom)
    return cols.reshape(r,-1).T


def checkpoint(path,params,z,config,info,extra=None):
    obj=dict(params=jax.tree_util.tree_map(np.asarray,params),codes=np.asarray(z),
             config=config,info=info,extra=extra)
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(dest.suffix+'.tmp')
    with temp.open('wb') as stream:
        pickle.dump(obj,stream,protocol=5)
    temp.replace(dest)


def train_bank(U,n,k_lat,r_feat,seed,steps,seconds,checkpoint_path,
               batch=16,width=128,lr=.001,callback=None):
    U=jnp.asarray(U,dtype=jnp.float64)
    U=U.reshape(len(U),-1)
    key=jax.random.PRNGKey(seed)
    kp,kz=jax.random.split(key)
    p=init(kp,k_lat,r_feat,width=width)
    # A deterministic field-only initialization preserves neighboring snapshots.
    # It initializes free latent codes only; G remains a learned coordinate MLP.
    init_start=time.monotonic()
    gram=np.asarray(U@U.T)
    eig,evec=np.linalg.eigh(gram)
    ids=np.argsort(eig)[::-1][:k_lat]
    scores=evec[:,ids]*np.sqrt(np.maximum(eig[ids],0.))[None,:]
    if scores.shape[1] < k_lat:
        scores=np.pad(scores,((0,0),(0,k_lat-scores.shape[1])))
    z=jnp.asarray(.25*scores/max(float(np.sqrt(np.mean(scores*scores))),1e-12))
    code_init_seconds=time.monotonic()-init_start
    xy,geom=coords(n),F.geometry(n)
    scale=jnp.mean(U*U)
    schedule=optax.warmup_cosine_decay_schedule(0.,lr,min(200,steps//10+1),steps,lr*.03)
    optimizer=optax.adam(schedule)
    optstate=optimizer.init((p,z))

    def loss(pz,values,indices,xy,geom,scale):
        params,codes=pz
        G=bank(params,xy,geom,n)
        H=head(params,codes[indices])
        errors=H@G.T-values
        fitting=jnp.mean(errors*errors)/scale
        gram=G.T@G/G.shape[0]
        # A weak regularizer conditions the bank without forcing orthogonality.
        orth=jnp.mean((gram-jnp.eye(r_feat))**2)
        return fitting+1e-4*orth,fitting

    @jax.jit
    def update(pz,st,values,indices,xy,geom,scale):
        (total,fit),grads=jax.value_and_grad(loss,has_aux=True)(pz,values,indices,xy,geom,scale)
        grads[0]['B']=jnp.zeros_like(grads[0]['B'])
        grads[0]['out_scale']=jnp.zeros_like(grads[0]['out_scale'])
        delta,st=optimizer.update(grads,st,pz)
        return optax.apply_updates(pz,delta),st,total,fit

    rng=np.random.default_rng(seed+1)
    pz=(p,z)
    start=time.monotonic()
    curve=[]
    for step in range(steps):
        ids=np.sort(rng.choice(len(U),min(batch,len(U)),replace=False))
        pz,optstate,total,fit=update(pz,optstate,U[ids],jnp.asarray(ids),xy,geom,scale)
        if step==0 or (step+1)%100==0 or step+1==steps:
            val=float(fit)
            row=dict(step=step+1,relative_mse=val,seconds=time.monotonic()-start)
            curve.append(row)
            print('bank_train',row,flush=True)
            if not np.isfinite(val):
                raise RuntimeError('nonfinite bank training')
            checkpoint(checkpoint_path,*pz,dict(n=n,k=k_lat,r=r_feat,seed=seed,width=width),
                       dict(curve=curve,stage='joint_bank_head',complete=False))
            if callback is not None:
                callback(curve)
            if row['seconds']>=seconds:
                break
    p,z=pz
    info=dict(curve=curve,steps=step+1,requested_steps=steps,seconds=time.monotonic()-start,
              stopping='step_budget' if step+1==steps else 'wall_budget',stage='joint_bank_head',
              code_initialization='training_field_POD_scores',code_init_seconds=code_init_seconds)
    checkpoint(checkpoint_path,p,z,dict(n=n,k=k_lat,r=r_feat,seed=seed,width=width),info)
    return p,np.asarray(z),info


def whiten(G,U,rcond=1e-10):
    G=np.asarray(G,dtype=np.float64)
    Q,R=np.linalg.qr(G,mode='reduced')
    singular=np.linalg.svd(R,compute_uv=False)
    if not np.all(np.isfinite(Q)) or singular[-1]<=rcond*singular[0]:
        raise RuntimeError(f'bank whitening failed: singular range {singular[[0,-1]]}')
    X=np.asarray(U).reshape(len(U),-1)
    rotated=X@Q
    coefficients=np.linalg.solve(R,rotated.T).T
    # Compute directly; norm-square subtraction can erase a small floor.
    perp=np.sum((X-coefficients@G.T)**2,axis=1)
    return Q,R,coefficients,perp,dict(condition=float(singular[0]/singular[-1]),
                                    singular_values=singular.tolist())


def train_head(params,z,C,Rb,steps,seconds,seed,checkpoint_path,config,lr=.001):
    # Only the h and h_lin leaves are trained, preserving the spatial bank exactly.
    base=params
    hp=dict(h=params['h'],h_lin=params['h_lin'])
    targets=jnp.asarray(C)@jnp.asarray(Rb).T
    metric=jnp.asarray(Rb)
    scale=jnp.mean(targets*targets)
    schedule=optax.warmup_cosine_decay_schedule(0.,lr,min(100,steps//10+1),steps,lr*.01)
    opt=optax.adam(schedule)
    hz=(hp,jnp.asarray(z))
    state=opt.init(hz)

    def loss(hz,target,metric,scale):
        h,z=hz
        # h_theta is a nonlinear MLP plus the unchanged linear skip form.
        pred=S.apply_mlp(h['h'],z)+z@h['h_lin']
        return jnp.mean((pred@metric.T-target)**2)/scale

    @jax.jit
    def update(hz,state,target,metric,scale):
        val,grad=jax.value_and_grad(loss)(hz,target,metric,scale)
        updates,state=opt.update(grad,state,hz)
        return optax.apply_updates(hz,updates),state,val

    start=time.monotonic()
    curve=[]
    for step in range(steps):
        hz,state,val=update(hz,state,targets,metric,scale)
        if step==0 or (step+1)%500==0 or step+1==steps:
            row=dict(step=step+1,relative_mse=float(val),seconds=time.monotonic()-start)
            curve.append(row)
            print('head_train',row,flush=True)
            if not np.isfinite(row['relative_mse']):
                raise RuntimeError('nonfinite head training')
            p={**base,**hz[0]}
            checkpoint(checkpoint_path,p,hz[1],config,dict(curve=curve,complete=False,stage='head_refine'))
            if row['seconds']>=seconds:
                break
    params={**base,**hz[0]}
    info=dict(curve=curve,steps=step+1,requested_steps=steps,seconds=time.monotonic()-start,
              stopping='step_budget' if step+1==steps else 'wall_budget',stage='head_refine')
    checkpoint(checkpoint_path,params,hz[1],config,info)
    return params,np.asarray(hz[1]),info


def representation(params,G,Rb,coeff,perp,truth,Z,starts=4,budget=200):
    fn=lambda z:head(params,z)
    z,rn,it,reason=oracle_fit(fn,jnp.asarray(Rb),coeff,Z,n_starts=starts,budget=budget,
                             gtol=1e-7,chunk=8)
    initial_norms=np.linalg.norm(np.asarray(truth).reshape(len(truth),-1),axis=1)
    bank_error=np.sqrt(perp)/initial_norms
    total=np.sqrt(perp+rn*rn)/initial_norms
    if np.any(total+1e-12<bank_error):
        raise RuntimeError('head fit below bank projection floor')
    return dict(bank_error=bank_error,head_error=total,z=z,iterations=it,reasons=reason,
                stationary=np.asarray(reason)==4)


def correction_directions(params,Rb,C,Z):
    residual=(np.asarray(C)-np.asarray(head(params,jnp.asarray(Z))))@np.asarray(Rb).T
    _,singular,Vt=np.linalg.svd(residual,full_matrices=False)
    correction=np.linalg.solve(np.asarray(Rb),Vt.T)
    return correction,singular


def pod(U,maxrank):
    X=np.asarray(U).reshape(len(U),-1)
    # Snapshot Gram avoids a tall dense SVD on the vector spatial dimension.
    gram=X@X.T
    eig,V=np.linalg.eigh(gram)
    ids=np.argsort(eig)[::-1][:maxrank]
    positive=eig[ids]>max(eig[-1],1e-300)*1e-13
    ids=ids[positive]
    basis=X.T@V[:,ids]/np.sqrt(eig[ids])[None,:]
    basis,_=np.linalg.qr(basis,mode='reduced')
    return basis,np.sqrt(np.maximum(eig[ids],0.))
