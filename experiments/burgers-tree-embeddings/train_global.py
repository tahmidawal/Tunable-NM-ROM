"""Matched-data ordered global bank and nonlinear heads; no final-test access."""
import argparse
import optax
from common import *
import decoder as dec, fit
from training_io import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--train',required=True);ap.add_argument('--dev',required=True)
    ap.add_argument('--updates',type=int,default=20000);ap.add_argument('--seed',type=int,default=2026111851)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    T=load_reference(a.train,'train');D=load_reference(a.dev,'dev');x,w=T['xy'],T['w'];Y=T['Y'];V=D['Y']
    result=dict(meta=preflight(),seed=a.seed,train_sources=T['sources'],dev_sources=D['sources'],
                status='training_global',checkpoints=[],heads={},recipes=['global_bank','global_head16','global_head32','global_head64'])
    save(out/'result.json',result)
    key=jax.random.PRNGKey(a.seed)
    # Equal trajectory/time, CURRENT-time normalization. Training-only POD teacher.
    normalized=Y/T['norm'][:,None]
    U,s,Vh=jax.jit(lambda A:jnp.linalg.svd(A,full_matrices=False))(normalized*jnp.sqrt(w)[None])
    basis=Vh[:128].T/jnp.sqrt(w)[:,None]
    scales=s[:128]/jnp.sqrt(len(Y));teacher=basis*scales[None]
    checkpoint(out/'pod_teacher.pkl',dict(basis=basis,scales=scales,singular_values=s,weights=w))
    result['pod_control']={str(r):summary(errors((V*w)@basis[:,:r]@basis[:,:r].T,V,w)) for r in (16,32,64,128)}
    p=dec.init_global(key);opt=optax.adam(optax.cosine_decay_schedule(1e-3,a.updates,alpha=.01));state=opt.init(p)
    @jax.jit
    def step(p,state,indices,x,target,w):
        def loss(p):
            pred=dec.global_features(p,x[indices]);delta=pred-target[indices]
            # Ordered-prefix energy objective with independently trained columns.
            weights=jnp.array([sum(1/r for r in (16,32,64,128) if i<r) for i in range(128)])
            return jnp.sum(w[indices,None]*delta**2*weights[None])*len(x)/len(indices)
        loss,grad=jax.value_and_grad(loss)(p);updates,state=opt.update(grad,state,p)
        return optax.apply_updates(p,updates),state,loss
    rng=np.random.default_rng(a.seed);best=np.inf;bestp=None;stale=0
    for it in range(1,a.updates+1):
        inds=jnp.asarray(rng.integers(len(x),size=1024));p,state,loss=step(p,state,inds,x,teacher,w)
        if it%1000==0 or it==a.updates:
            G=dec.global_features(p,x);C,pred,sv=fit.linear_fit(G,V,w);err=errors(pred,V,w);metric=float(jnp.max(err))
            row=dict(update=it,loss=float(loss),projection=summary(err),condition=float(sv[0]/jnp.maximum(sv[-1],1e-300)))
            path=out/f'global_bank_{it:05d}.pkl';row['sha256']=checkpoint(path,dict(params=p,update=it,metrics=row))
            result['checkpoints'].append(row);save(out/'result.json',result);print('global',it,metric,flush=True)
            if metric<best:
                improve=metric<best*.99;best=metric;bestp=jax.device_get(p);result['selected_bank_update']=it
                stale=0 if improve else stale+1
            else:stale+=1
            if stale>=5:break
    p=jax.tree_util.tree_map(jnp.asarray,bestp);Graw=dec.global_features(p,x)
    # Global whitening only; compact local supports are never mixed globally.
    Q,sv,Vh=jnp.linalg.svd(Graw*jnp.sqrt(w[:,None]),full_matrices=False)
    transform=Vh.T/jnp.maximum(sv,1e-12)[None];G=Graw@transform
    C,_,_=fit.linear_fit(G,Y,w);CD,_,_=fit.linear_fit(G,V,w)
    model=dict(global_params=p,global_transform=transform,heads={},seed=a.seed,selected_bank_update=result['selected_bank_update'])
    result['global_direct']={}
    for r in (16,32,64,128):
        cc,pr,svr=fit.linear_fit(Graw[:,:r],V,w)
        result['global_direct'][str(r)]=summary(errors(pr,V,w))
    for k in (16,32,64):
        key,sub=jax.random.split(key)
        _,_,vh=jnp.linalg.svd(C/T['norm'][:,None],full_matrices=False)
        L=vh[:k].T;head=dec.init_head(sub,k,128,64,L);codes=C@L
        params=dict(head=head,codes=codes);opt=optax.adam(optax.cosine_decay_schedule(1e-3,a.updates,alpha=.01));os_=opt.init(params)
        @jax.jit
        def hstep(params,os_,idx,C,norm):
            def loss(q):
                r=dec.head(q['head'],q['codes'][idx])-C[idx]
                return jnp.mean(jnp.sum(r*r,axis=1)/norm[idx]**2)
            loss,g=jax.value_and_grad(loss)(params);up,os_=opt.update(g,os_,params)
            return optax.apply_updates(params,up),os_,loss
        # Projection in orthonormal physical coefficient metric; missing-span floor added by field score.
        def evaluate(h):
            projector=fit.make_project(lambda z:dec.head(h,z),25)
            starts=[CD@jnp.linalg.pinv(h['linear']).T,jnp.zeros((len(V),k))]
            starts.append(starts[0]+jax.random.normal(sub,starts[0].shape,dtype=jnp.float64)*.05)
            zz,err,stat,restarts=fit.multi_fit(projector,jnp.eye(128),CD,jnp.ones(128),starts)
            warm=zz.reshape(-1,11,k)
            warm=jnp.concatenate([warm[:,:1],warm[:,:-1]],axis=1).reshape(-1,k)
            zz,err,stat,extra=fit.multi_fit(projector,jnp.eye(128),CD,jnp.ones(128),[zz,warm])
            physical=errors(dec.head(h,zz)@G.T,V,w)
            return zz,physical,stat,jnp.concatenate([restarts,extra],axis=0)
        best=np.inf;besth=None;history=[];stale=0
        for it in range(1,a.updates+1):
            idx=jnp.asarray(rng.integers(len(Y),size=64));params,os_,loss=hstep(params,os_,idx,C,T['norm'])
            if it%1000==0 or it==a.updates:
                z,err,stat,restarts=evaluate(params['head']);metric=float(jnp.max(err))
                row=dict(update=it,train_loss=float(loss),projection=summary(err),max_stationarity=float(jnp.max(stat)),
                         restart_spread=float(jnp.max(jnp.max(restarts,axis=0)-jnp.min(restarts,axis=0))))
                checkpoint(out/f'global_head{k}_{it:05d}.pkl',dict(params=params,dev_codes=z,metrics=row))
                history.append(row);result['heads'][str(k)]=dict(history=history)
                save(out/'result.json',result);print('head',k,it,metric,float(jnp.max(stat)),flush=True)
                if metric<best:
                    improve=metric<best*.99;best=metric;besth=jax.device_get(params['head']);bestz=jax.device_get(z)
                    bestit=it;stale=0 if improve else stale+1
                else:stale+=1
                if stale>=5:break
        h=jax.tree_util.tree_map(jnp.asarray,besth);model['heads'][str(k)]=h
        projector=fit.make_project(lambda z:dec.head(h,z),25)
        zt,et,st,rt=fit.multi_fit(projector,jnp.eye(128),C,jnp.ones(128),[C@jnp.linalg.pinv(h['linear']).T,jnp.zeros((len(Y),k))])
        et=errors(dec.head(h,zt)@G.T,Y,w)
        checkpoint(out/f'global_head{k}_selected.pkl',dict(head=h,train_codes=zt,dev_codes=bestz,
                   train_stationarity=st,selected_update=bestit,train_error=et))
        result['heads'][str(k)].update(selected_update=bestit,worst_projection=best,train_projection=summary(et))
        checkpoint(out/'global_model.pkl',model);save(out/'result.json',result)
    result['status']='complete_global_representation_only';result['model_sha256']=checkpoint(out/'global_model.pkl',model)
    save(out/'result.json',result)

if __name__=='__main__':main()
