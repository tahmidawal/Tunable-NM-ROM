"""B1 scaled nonlinear global head; spectral feature control, not native reproduction."""
import argparse,optax
from common import *
from training_io import checkpoint,summary
import normalized_heads as NH,fit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--span',required=True)
    ap.add_argument('--latent',type=int,default=16);ap.add_argument('--updates',type=int,default=20000)
    ap.add_argument('--seed',type=int,default=2026111851);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta=preflight();span=Path(a.span);record=json.loads((span/'result.json').read_text())
    assert digest(span/'reduced_data.npz')==record['reduced_data_sha256'];assert digest(span/'pod_basis.npz')==record['basis_sha256']
    d=np.load(span/'reduced_data.npz');C=jnp.asarray(d['train_coefficients']);V=jnp.asarray(d['dev_coefficients'])
    tn=jnp.asarray(d['train_norm']);vn=jnp.asarray(d['dev_norm']);tm=jnp.asarray(d['train_missing']);vm=jnp.asarray(d['dev_missing'])
    p,mean,scale=NH.initialize(C,tn,a.latent,jax.random.PRNGKey(a.seed))
    opt=optax.adam(optax.cosine_decay_schedule(1e-3,a.updates,alpha=.01));state=opt.init(p)
    def decode(h,z):return NH.head(h,z,mean,scale)
    projector=fit.make_parametric_project(lambda z,h:decode(h,z),75)
    trainprojector=fit.make_parametric_project(lambda z,h:decode(h,z),25)
    batched=jax.jit(jax.vmap(projector,in_axes=(None,0,None,0,None)))
    trainbatch=jax.jit(jax.vmap(trainprojector,in_axes=(None,0,None,0,None)))
    eye=jnp.eye(C.shape[1]);ones=jnp.ones(C.shape[1]);rng=np.random.default_rng(a.seed)
    result=dict(meta=meta,status='training_normalized_parent',latent=a.latent,seed=a.seed,feature_span=f'training-only spectral POD{C.shape[1]}',
                native_reproduction=False,global_zero_anchored=False,basis_sha256=record['basis_sha256'],
                data_sha256=record['reduced_data_sha256'],train_trajectories=len(C)//11,dev_trajectories=len(V)//11,
                train_labels=record.get('train_labels'),dev_labels=record.get('dev_labels'),
                checkpoints=[],final_test_opened=False)
    @jax.jit
    def step(p,state,idx,C,norm,joint):
        def loss(p):
            ze=NH.encode(p['encoder'],C[idx],mean,scale)
            ae=jnp.sum((decode(p['head'],ze)-C[idx])**2,axis=1)/norm[idx]**2
            code=jnp.sum((decode(p['head'],p['codes'][idx])-C[idx])**2,axis=1)/norm[idx]**2
            return jnp.mean(jnp.where(joint,.5*ae+.5*code,ae))
        value,grad=jax.value_and_grad(loss)(p);up,state=opt.update(grad,state,p)
        return optax.apply_updates(p,up),state,value
    def evaluate(p):
        ze=NH.encode(p['encoder'],V,mean,scale)
        candidates=[ze,jnp.zeros_like(ze)]
        results=[batched(eye,V,ones,z,p['head']) for z in candidates]
        losses=jnp.stack([v[1] for v in results]);which=jnp.argmin(losses,axis=0)
        z=jnp.stack([v[0] for v in results])[which,jnp.arange(len(V))]
        warm=z.reshape(-1,11,a.latent);warm=jnp.concatenate([warm[:,:1],warm[:,:-1]],axis=1).reshape(z.shape)
        extra=batched(eye,V,ones,warm,p['head']);old=batched(eye,V,ones,z,p['head']);use=extra[1]<old[1]
        z=jnp.where(use[:,None],extra[0],old[0]);station=jnp.where(use,extra[2],old[2])
        err=jnp.sqrt(jnp.sum((decode(p['head'],z)-V)**2,axis=1)+vm)/vn
        encoder_error=jnp.sqrt(jnp.sum((decode(p['head'],ze)-V)**2,axis=1)+vm)/vn
        return z,err,station,encoder_error
    best=np.inf;stale=0;bestp=None
    for it in range(1,a.updates+1):
        idx=jnp.asarray(rng.integers(len(C),size=128));p,state,loss=step(p,state,idx,C,tn,it>=3000)
        if it%1000==0 or it==a.updates:
            start=NH.encode(p['encoder'],C,mean,scale)
            tz,_,ts,_=trainbatch(eye,C,ones,start,p['head'])
            p=dict(p);p['codes']=jax.lax.stop_gradient(tz)
            z,err,station,encerr=evaluate(p);metric=float(jnp.max(err))
            trainerr=jnp.sqrt(jnp.sum((decode(p['head'],tz)-C)**2,axis=1)+tm)/tn
            row=dict(update=it,loss=float(loss),projection=summary(err.reshape(-1,11)),train_projection=summary(trainerr.reshape(-1,11)),
                     encoder_projection=summary(encerr.reshape(-1,11)),max_stationarity=float(jnp.max(station)),train_stationarity=float(jnp.max(ts)))
            row['sha256']=checkpoint(out/f'normalized_parent_{it:05d}.pkl',dict(params=p,mean=mean,scale=scale,metrics=row,dev_codes=z))
            result['checkpoints'].append(row);save(out/'result.json',result);print('normalized_parent',a.latent,it,metric,float(jnp.max(trainerr)),flush=True)
            if metric<best:
                significant=metric<best*.99;best=metric;bestp=jax.device_get(p);bestz=np.asarray(z);bestit=it
                stale=0 if significant else stale+1
            else:stale+=1
            if it>=8000 and stale>=5:break
    selected=dict(params=bestp,mean=np.asarray(mean),scale=np.asarray(scale),dev_codes=bestz,latent=a.latent,
                  selected_update=bestit,span_path=str(span),basis_sha256=record['basis_sha256'])
    result.update(status='complete_normalized_parent_representation',selected_update=bestit,worst_projection=best)
    result['model_sha256']=checkpoint(out/'normalized_parent.pkl',selected);save(out/'result.json',result)

if __name__=='__main__':main()
