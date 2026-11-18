"""Frozen shared features; train a small common nonlinear child head and independent codes."""
import argparse,optax
from common import *
import decoder as dec,geometry as geo,fit,local_study as LS
from training_io import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--train',required=True)
    ap.add_argument('--dev',required=True);ap.add_argument('--local-model',required=True);ap.add_argument('--global-model',required=True)
    ap.add_argument('--updates',type=int,default=20000);ap.add_argument('--latent',type=int,default=4)
    ap.add_argument('--seed',type=int,default=2026111851)
    ap.add_argument('--refit-each-update',action='store_true');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    T=load_reference(a.train,'train');D=load_reference(a.dev,'dev')
    localdir=Path(a.local_model);globaldir=Path(a.global_model)
    for p in [localdir,globaldir]:assert str(p).startswith(os.environ['STUDY_NAMESPACE']+'/')
    localmodel=read_checkpoint(localdir/'local_model.pkl');model=localmodel['global_model'];pl=localmodel['local']
    rootfit=read_checkpoint(globaldir/'global_head16_selected.pkl');zt=jnp.asarray(rootfit['train_codes']);zd=jnp.asarray(rootfit['dev_codes'])
    parent=model['heads']['16'];x,w,Y=T['xy'],T['w'],T['Y'];G=LS.global_matrix(model,x)
    rootpred=dec.head(parent,zt)@G.T;cells,masks,trees=LS.padded_trees(T,rootpred,a.latent,a.seed)
    P=cells.shape[2];K=16+P*a.latent;codes=np.zeros((8,len(Y),K));codes[:,:,:16]=np.asarray(zt)[None]
    key=jax.random.PRNGKey(a.seed);child=dec.init_head(key,a.latent,16,32)
    result=dict(meta=preflight(),seed=a.seed,latent=a.latent,refit_each_update=a.refit_each_update,status='residual_initialization',
                local_model_sha256=digest(localdir/'local_model.pkl'),checkpoints=[],local_features_frozen=True,
                zero_child_formula='L*z+f(z)-f(0)',parent_weights_frozen=True)
    # Fit all corrections from the missing physical field; no encoder shortcut.
    cache={}
    for i in range(len(T['parameters'])):
        residual=Y[i*11:(i+1)*11]-rootpred[i*11:(i+1)*11]
        for q,cs in enumerate(trees[i]):
            ident=tuple(cs)
            if ident not in cache:cache[ident]=dec.local_matrix(pl,x,cs,a.latent)
            c,_,_=fit.linear_fit(cache[ident],residual,w)
            codes[q,i*11:(i+1)*11,16:16+len(cs)*a.latent]=np.asarray(c)
    params=dict(head=child,codes=jnp.asarray(codes));rng=np.random.default_rng(a.seed)
    opt=optax.adam(optax.cosine_decay_schedule(1e-3,a.updates,alpha=.01));os_=opt.init(params)
    projector=fit.make_parametric_project(lambda z,h:dec.coefficients(parent,h,z,P,a.latent),25)
    batchproject=jax.vmap(projector,in_axes=(0,0,None,0,None))
    @jax.jit
    def step(params,os_,idx,pts,q,Y,norm,x,w,G,cells,masks):
        pc=cells[idx//11,q];mask=masks[idx//11,q]
        B=jax.vmap(lambda cs:jax.vmap(lambda c:dec.local_features(pl,x[pts],c))(cs))(pc)
        station=jnp.float64(0)
        if a.refit_each_update:
            BB=(B*mask[:,:,None,None]).transpose(0,2,1,3).reshape(len(idx),len(pts),P*16)
            GG=jnp.concatenate([jnp.broadcast_to(G[pts],(len(idx),len(pts),128)),BB],axis=-1)
            zz,_,ss,_=batchproject(GG,Y[idx[:,None],pts[None]],w[pts],params['codes'][q,idx],params['head'])
            params=dict(params);params['codes']=params['codes'].at[q,idx].set(jax.lax.stop_gradient(zz));station=jnp.max(ss)
        def loss(p):
            code=p['codes'][q,idx];root=dec.head(parent,code[:,:16])@G[pts].T
            coeff=dec.head(p['head'],code[:,16:].reshape(len(idx),P,a.latent))
            local=jnp.einsum('bpnr,bpr,bp->bn',B,coeff,mask)
            delta=root+local-Y[idx[:,None],pts[None]]
            return jnp.mean(jnp.sum(delta**2*w[pts][None],axis=1)/norm[idx]**2)*len(x)/len(pts)
        loss,g=jax.value_and_grad(loss)(params);up,os_=opt.update(g,os_,params)
        return optax.apply_updates(params,up),os_,loss,station
    best=np.inf;besthead=None;stale=0
    for it in range(1,a.updates+1):
        q=int(rng.choice([0,5,6,7] if it%2==0 else [1,2,3,4]))
        idx=jnp.asarray(rng.integers(len(Y),size=8));pts=jnp.asarray(rng.integers(len(x),size=512))
        params,os_,loss,station=step(params,os_,idx,pts,q,Y,T['norm'],x,w,G,cells,masks)
        if it%1000==0 or it==a.updates:
            rows,saved=LS.evaluate_pair(model,pl,params['head'],D,zd,a.latent,25,3)
            metric=max(r['projection']['worst'] for r in rows)
            record=dict(update=it,loss=float(loss),comparisons=rows,worst=metric,last_minibatch_fit_stationarity=float(station))
            record['sha256']=checkpoint(out/f'child_head_{it:05d}.pkl',dict(head=params['head'],metrics=record))
            result['checkpoints'].append(record);save(out/'result.json',result);print('children',it,metric,flush=True)
            if metric<best:
                improve=metric<best*.99;best=metric;besthead=jax.device_get(params['head']);bestit=it
                stale=0 if improve else stale+1
                checkpoint(out/'child_training_codes_selected.pkl',dict(codes=params['codes']))
            else:stale+=1
            if stale>=5:break
    child=jax.tree_util.tree_map(jnp.asarray,besthead)
    result.update(status='complete_nonlinear_child_representation_only',selected_update=bestit)
    result['model_sha256']=checkpoint(out/'tree_model.pkl',dict(global_model=model,local=pl,child=child,
                     latent=a.latent,seed=a.seed,selected_update=bestit,local_model_sha256=digest(localdir/'local_model.pkl')))
    save(out/'result.json',result)
if __name__=='__main__':main()
