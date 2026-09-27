"""Fresh spatial-bank and coefficient-head training on the original B3D family.

The two-stage relative-loss recipe follows paper-h3d/train.py at e6460d73;
the model, boundary factors, family and FOM are the B3D primitives.
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax
import core as c
from run import dump, host, reference_audit
b3=c.b3


def checkpoint(path,value):
    path=Path(path);tmp=Path(str(path)+'.partial')
    value=jax.tree_util.tree_map(lambda x:np.asarray(x) if isinstance(x,jax.Array) else x,value)
    tmp.write_bytes(pickle.dumps(value));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tc=cfg['training'];assert jax.default_backend()=='gpu'
    print('jax_backend=gpu fresh bank/head training f64',flush=True)
    n=cfg['nodes'];coords=b3.grid_coords_3d(n);idx=b3.interior_indices_3d(n)
    raw=b3.draw_param_table(cfg['seed'],max(cfg['validation_rows'])+1)
    raw['s_star']=b3.peak_on_reference_grid(raw)
    rows=list(range(cfg['train_trajectories']));fom=b3.make_newton_tol_rollout(n,'fft')
    shape=(len(rows)*len(cfg['train_steps']),len(idx))
    u=np.lib.format.open_memmap(out/'training_fields.npy',mode='w+',dtype=np.float64,shape=shape)
    records=[];begin=time.perf_counter()
    for row in rows:
        u0=b3.blob_ic_3d(n,raw,row,coords)[idx];nu=float(raw['nu'][row])
        fields,its,rn=host(fom(jnp.asarray(u0),nu,1e-10,1e-11))
        defect=reference_audit(fields,nu,n,cfg['dt'])
        assert np.isfinite(fields).all() and defect<2e-9
        a=row*len(cfg['train_steps']);u[a:a+len(cfg['train_steps'])]=fields[cfg['train_steps']]
        records.append(dict(row=row,nu=nu,max_defect=defect,iterations=its.tolist()))
        if (row+1)%32==0:
            print('TRAIN DATA',row+1,len(rows),round(time.perf_counter()-begin,2),flush=True)
            dump(out/'data_progress.json',dict(rows=records,seconds=time.perf_counter()-begin))
    u.flush();dump(out/'data_progress.json',dict(rows=records,seconds=time.perf_counter()-begin,complete=True))
    k=tc['latent_dimension'];r=tc['rank'];key=jax.random.PRNGKey(tc['seed']);key,a,b=jax.random.split(key,3)
    scale=float(np.sqrt(np.mean(u*u)))
    params=b3.init_separable_3d(a,k,r,n_ff=tc['fourier_features'],ff_scale=tc['fourier_scale'],
        g_hidden=tc['bank_width'],g_layers=2,h_hidden=tc['head_width'],h_layers=2,out_scale=scale)
    gp={name:params[name] for name in ['B','g','out_scale']}
    eta=.1*jax.random.normal(b,(len(u),r),dtype=jnp.float64)
    data=jnp.asarray(u);x=jnp.asarray(coords[idx]);norms=jnp.maximum(jnp.mean(data*data,axis=1),1e-20)
    opt=optax.adam(optax.cosine_decay_schedule(tc['bank_learning_rate'],tc['bank_steps'],alpha=.03))
    pz=(gp,eta);state=opt.init(pz)
    bs=min(tc['batch_states'],len(u));bp=min(tc['batch_points'],len(idx))
    def loss(pz,values,points,norm,key):
        p,z=pz;a,b=jax.random.split(key)
        rows=jax.random.randint(a,(bs,),0,len(values));cols=jax.random.randint(b,(bp,),0,len(points))
        g=b3.features(p,points[cols]);pred=z[rows]@g.T
        e=jnp.mean((pred-values[rows[:,None],cols[None,:]])**2,axis=1)/norm[rows]
        gram=(g/scale).T@(g/scale)/bp
        return jnp.mean(e)+.1*jnp.mean(e*e)+1e-6*jnp.mean((gram-jnp.eye(r))**2)
    @jax.jit
    def step(pz,state,key,values,points,norm):
        value,grad=jax.value_and_grad(loss)(pz,values,points,norm,key)
        grad[0]['B']=jnp.zeros_like(grad[0]['B']);grad[0]['out_scale']=jnp.zeros_like(grad[0]['out_scale'])
        update,state=opt.update(grad,state,pz)
        return optax.apply_updates(pz,update),state,value
    curve=[];begin=time.perf_counter()
    for it in range(tc['bank_steps']):
        key,sub=jax.random.split(key);pz,state,value=step(pz,state,sub,data,x,norms)
        if it==0 or (it+1)%100==0 or it+1==tc['bank_steps']:curve.append(dict(step=it+1,objective=float(value),seconds=time.perf_counter()-begin))
        if (it+1)%tc['checkpoint_every']==0 or it+1==tc['bank_steps']:
            checkpoint(out/'bank_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg))
            dump(out/'bank_curve.json',curve);print('BANK',curve[-1],flush=True)
    gp,eta=pz;params.update(gp)
    G=np.concatenate([np.asarray(b3.features(gp,x[s:s+2048])) for s in range(0,len(idx),2048)])
    Q,Rb=np.linalg.qr(G,mode='reduced');sv=np.linalg.svd(Rb,compute_uv=False)
    assert sv[-1]>sv[0]*1e-12,(sv[0],sv[-1])
    target=np.asarray(data@jnp.asarray(Q));norm2=np.asarray(jnp.sum(data*data,axis=1))
    floor=np.maximum(norm2-np.sum(target*target,axis=1),0.)
    bank_info=dict(seconds=time.perf_counter()-begin,rank=r,condition=float(sv[0]/sv[-1]),
        mean_relative_projection=float(np.mean(np.sqrt(floor/norm2))),worst_relative_projection=float(np.max(np.sqrt(floor/norm2))))
    dump(out/'bank_info.json',bank_info);print('BANK FLOOR',bank_info,flush=True)
    checkpoint(out/'bank.pkl',dict(params=gp,Q=Q,Rb=Rb,cfg=cfg,info=bank_info))
    del G,eta,state,pz,data
    # PCA initializes only the latent codes and linear skip of the neural head;
    # the spatial functions remain the freshly trained coordinate network.
    _,_,vt=np.linalg.svd(target,full_matrices=False)
    coeff_scale=float(np.sqrt(np.mean(target*target)))
    normalized=target/coeff_scale
    z=jnp.asarray(normalized@vt[:k].T)
    hp={name:params[name] for name in ['h','h_lin']}
    hp['h_lin']=jnp.asarray(vt[:k])
    hp['h'][-1]=(hp['h'][-1][0]*.01,hp['h'][-1][1]*.01)
    tt=jnp.asarray(normalized);nn=jnp.asarray(norm2/coeff_scale**2);ff=jnp.asarray(floor/coeff_scale**2)
    opt=optax.adam(optax.cosine_decay_schedule(tc['head_learning_rate'],tc['head_steps'],alpha=.03))
    pz=(hp,z);state=opt.init(pz)
    def headloss(pz,targets,norm,perp,key):
        p,z=pz;rows=jax.random.randint(key,(min(tc['head_batch'],len(targets)),),0,len(targets))
        residual=b3.head(p,z[rows])-targets[rows]
        e=(jnp.sum(residual*residual,axis=1)+perp[rows])/norm[rows]
        return jnp.mean(e)+.1*jnp.mean(e*e)
    @jax.jit
    def headstep(pz,state,key,targets,norm,perp):
        value,grad=jax.value_and_grad(headloss)(pz,targets,norm,perp,key)
        update,state=opt.update(grad,state,pz)
        return optax.apply_updates(pz,update),state,value
    curve=[];begin=time.perf_counter()
    for it in range(tc['head_steps']):
        key,sub=jax.random.split(key);pz,state,value=headstep(pz,state,sub,tt,nn,ff)
        if it==0 or (it+1)%100==0 or it+1==tc['head_steps']:curve.append(dict(step=it+1,objective=float(value),seconds=time.perf_counter()-begin))
        if (it+1)%tc['checkpoint_every']==0 or it+1==tc['head_steps']:
            checkpoint(out/'head_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg,coeff_scale=coeff_scale))
            dump(out/'head_curve.json',curve);print('HEAD',curve[-1],flush=True)
    hp,z=pz
    pred=np.asarray(b3.head(hp,z))*coeff_scale
    err=np.sqrt((np.sum((pred-target)**2,axis=1)+floor)/norm2)
    dump(out/'head_info.json',dict(seconds=time.perf_counter()-begin,mean=float(np.mean(err)),median=float(np.median(err)),
        worst=float(np.max(err)),steps=tc['head_steps'],converged_claim=False))
    # Convert the head output from Q coordinates back to the raw learned bank.
    transform=jnp.asarray(np.linalg.inv(Rb).T*coeff_scale)
    w,b=hp['h'][-1];hp['h'][-1]=(w@transform,b@transform);hp['h_lin']=hp['h_lin']@transform
    params.update(hp)
    reconstruction=np.asarray(b3.head(params,z))@Rb.T
    assert np.linalg.norm(reconstruction-pred)/np.linalg.norm(pred)<1e-10
    ck=dict(params=params,Z_tr=z,cfg=dict(k=k,r=r,training=cfg),training_rows=rows,
            training_steps=cfg['train_steps'],bank_info=bank_info,source='fresh relative-loss B3D learned spatial bank and head')
    checkpoint(out/'checkpoint.pkl',ck)
    np.savez_compressed(out/'coordinates.npz',target=target,norm2=norm2,floor=floor,codes=np.asarray(z))
    print('TRAIN COMPLETE',flush=True)


if __name__=='__main__':main()
