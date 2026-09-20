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
    p['out_scale']=jnp.sqrt(jnp.mean(U*U))
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


def train_head(params,z,C,Rb,steps,seconds,seed,checkpoint_path,config,lr=.001,snapshot_variance=None):
    # Only the h and h_lin leaves are trained, preserving the spatial bank exactly.
    base=params
    hp=dict(h=params['h'],h_lin=params['h_lin'])
    targets=jnp.asarray(C)@jnp.asarray(Rb).T
    metric=jnp.asarray(Rb)
    scale=jnp.mean(targets*targets) if snapshot_variance is None else jnp.asarray(snapshot_variance)[:,None]*(3*config['n']**3)/Rb.shape[0]
    schedule=optax.warmup_cosine_decay_schedule(0.,lr,min(100,steps//10+1),steps,lr*.01)
    opt=optax.adam(schedule)
    hz=(hp,jnp.asarray(z))
    state=opt.init(hz)

    def loss(hz,target,metric,scale):
        h,z=hz
        # h_theta is a nonlinear MLP plus the unchanged linear skip form.
        pred=S.apply_mlp(h['h'],z)+z@h['h_lin']
        return jnp.mean((pred@metric.T-target)**2/scale)

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
    # Explicit parameter arguments avoid capturing large head weights in XLA.
    from ns2d_rom import make_lm
    theta={name:params[name] for name in ('h','h_lin')}
    def residual(z,c,metric,theta):
        return metric@(head(theta,z)-c)
    lm=make_lm(residual,budget,gtol=1e-7)
    Hrot=np.asarray(head(theta,jnp.asarray(Z)))@np.asarray(Rb).T
    hn=np.sum(Hrot*Hrot,axis=1)
    def one(c,metric,theta,codes,rotated,norms):
        scores=norms-2*rotated@(metric@c)
        initial=codes[jnp.argsort(scores)[:starts]]
        result=jax.vmap(lambda z0:lm(z0,(c,metric,theta),0.))(initial)
        index=jnp.argmin(result[1])
        return tuple(a[index] for a in result)
    fit=jax.jit(jax.vmap(one,in_axes=(0,None,None,None,None,None)))
    batches=[]
    for s in range(0,len(coeff),8):
        batches.append(tuple(np.asarray(a) for a in fit(jnp.asarray(coeff[s:s+8]),jnp.asarray(Rb),
                              theta,jnp.asarray(Z),jnp.asarray(Hrot),jnp.asarray(hn))))
    z,rn,it,reason,gradient=tuple(np.concatenate([b[i] for b in batches]) for i in range(5))
    initial_norms=np.linalg.norm(np.asarray(truth).reshape(len(truth),-1),axis=1)
    bank_error=np.sqrt(perp)/initial_norms
    total=np.sqrt(perp+rn*rn)/initial_norms
    if np.any(total+1e-12<bank_error):
        raise RuntimeError('head fit below bank projection floor')
    return dict(bank_error=bank_error,head_error=total,z=z,iterations=it,reasons=reason,
                stationary=np.asarray(reason)==4,normalized_gradient=gradient)


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


def pod_gpu(U,maxrank):
    """Exact snapshot-Gram POD with GPU eigensolve; return basis and scores."""
    X=jnp.asarray(U,dtype=jnp.float64).reshape(len(U),-1)
    values,vectors=jnp.linalg.eigh(X@X.T)
    ids=jnp.argsort(values)[::-1][:maxrank]
    eig=values[ids]
    if float(eig[-1]) <= float(eig[0])*1e-13:
        raise RuntimeError('requested POD rank exceeds measured snapshot rank')
    scores=vectors[:,ids]*jnp.sqrt(eig)[None,:]
    basis=(X.T@vectors[:,ids])/jnp.sqrt(eig)[None,:]
    return np.asarray(basis),np.asarray(scores),np.asarray(jnp.sqrt(eig))


def train_free_bank(U,n,k_lat,r_feat,seed,steps,seconds,checkpoint_path,
                    pod_scores,batch=16,width=512,n_ff=256,lr=.001,callback=None,
                    spatial_batch=1024,snapshot_variance=None):
    """Learn spatial bank with unrestricted coefficients before fitting a head.

    The POD scores initialize coefficients only. Spatial fields remain a learned
    periodic vector MLP followed by the exact solenoidal projection.
    """
    U=jnp.asarray(U,dtype=jnp.float64).reshape(len(U),-1)
    params=init(jax.random.PRNGKey(seed),k_lat,r_feat,width=width,n_ff=n_ff)
    scale=jnp.mean(U*U)
    fitting_scale=scale if snapshot_variance is None else jnp.asarray(snapshot_variance)[:,None]
    params['out_scale']=jnp.sqrt(scale)
    normalization=float(jnp.sqrt(scale*U.shape[1]))
    coefficients=jnp.asarray(pod_scores[:,:r_feat]/normalization)
    xy=coords(n)
    U3=U.reshape(len(U),3,n**3)
    opt=optax.adam(optax.warmup_cosine_decay_schedule(0.,lr,min(300,steps//10+1),steps,lr*.02))
    pc=(params,coefficients);state=opt.init(pc)
    def loss(pc,values,indices,points,scale):
        p,c=pc
        angle=2*jnp.pi*(points@p['B'])
        ff=jnp.concatenate((jnp.sin(angle),jnp.cos(angle)),axis=-1)
        raw=p['out_scale']*S.apply_mlp(p['g'],ff)
        G=raw.reshape(len(points),3,r_feat).transpose(1,0,2).reshape(3*len(points),r_feat)
        denominator=scale if scale.ndim==0 else scale[indices]
        return jnp.mean((c[indices]@G.T-values)**2/denominator)
    @jax.jit
    def update(pc,state,values,indices,points,scale):
        value,grad=jax.value_and_grad(loss)(pc,values,indices,points,scale)
        grad[0]['B']=jnp.zeros_like(grad[0]['B'])
        grad[0]['out_scale']=jnp.zeros_like(grad[0]['out_scale'])
        updates,state=opt.update(grad,state,pc)
        return optax.apply_updates(pc,updates),state,value
    rng=np.random.default_rng(seed+1)
    start=time.monotonic();curve=[]
    for step in range(steps):
        ids=np.sort(rng.choice(len(U),min(batch,len(U)),replace=False))
        positions=np.sort(rng.choice(n**3,min(spatial_batch,n**3),replace=False))
        values=U3[ids][:,:,positions].reshape(len(ids),-1)
        pc,state,value=update(pc,state,values,jnp.asarray(ids),xy[positions],fitting_scale)
        if step==0 or (step+1)%100==0 or step+1==steps:
            row=dict(step=step+1,relative_mse=float(value),seconds=time.monotonic()-start)
            if not np.isfinite(row['relative_mse']):
                raise RuntimeError('nonfinite free-bank training')
            curve.append(row);print('free_bank_train',row,flush=True)
            checkpoint(checkpoint_path,*pc,dict(n=n,k=k_lat,r=r_feat,seed=seed,width=width,n_ff=n_ff),
                       dict(stage='unrestricted_coefficient_bank',curve=curve,complete=False))
            if callback is not None:callback(curve)
            if row['seconds']>=seconds:break
    params,coefficients=pc
    scores=pod_scores[:,:k_lat]
    z=.25*scores/max(float(np.sqrt(np.mean(scores*scores))),1e-12)
    info=dict(stage='unrestricted_coefficient_bank',steps=step+1,requested_steps=steps,
              seconds=time.monotonic()-start,curve=curve,
              stopping='step_budget' if step+1==steps else 'wall_budget',
              coefficient_initialization='training_field_POD_scores',
              objective='raw field MSE with spatial point minibatches' if snapshot_variance is None else 'raw field MSE divided by trajectory initial mean-square velocity',spatial_batch=spatial_batch,
              spatial_bank='learned periodic vector coordinate MLP, not POD substitution')
    checkpoint(checkpoint_path,params,coefficients,dict(n=n,k=k_lat,r=r_feat,seed=seed,width=width,n_ff=n_ff),info)
    return params,np.asarray(z),info,np.asarray(coefficients)
