"""Parent-first residual initialization and assembled shared-feature training."""
import argparse, optax
from common import *
import decoder as dec, geometry as geo, fit, local_study as LS
from training_io import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--train',required=True)
    ap.add_argument('--dev',required=True);ap.add_argument('--global-model',required=True)
    ap.add_argument('--updates',type=int,default=20000);ap.add_argument('--seed',type=int,default=2026111851)
    ap.add_argument('--width',type=int,default=64);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    T=load_reference(a.train,'train');D=load_reference(a.dev,'dev');globaldir=Path(a.global_model)
    assert str(globaldir).startswith(os.environ['STUDY_NAMESPACE']+'/')
    model=read_checkpoint(globaldir/'global_model.pkl');rootfit=read_checkpoint(globaldir/'global_head16_selected.pkl')
    zt=jnp.asarray(rootfit['train_codes']);zd=jnp.asarray(rootfit['dev_codes']);parent=model['heads']['16']
    x,w,Y=T['xy'],T['w'],T['Y'];G=LS.global_matrix(model,x);rootpred=dec.head(parent,zt)@G.T
    key=jax.random.PRNGKey(a.seed);pl=dec.init_local(key,a.width);rng=np.random.default_rng(a.seed)
    result=dict(meta=preflight(),seed=a.seed,width=a.width,global_model_sha256=digest(globaldir/'global_model.pkl'),
                status='local_residual_initialization',checkpoints=[],recipes=['shared_local_direct'],
                training_mixture='equal prefixes4/8/16; equal 8 configurations, half shallow; equal snapshots/trajectories')
    # A short smooth polynomial teacher warm start, followed by actual missing-field loss.
    opt=optax.adam(1e-3);os_=opt.init(pl)
    @jax.jit
    def poly_step(p,os_,v):
        loss=lambda p:jnp.mean((dec.mlp(p,v)-LS.polynomial_inputs(v))**2)
        loss,g=jax.value_and_grad(loss)(p);up,os_=opt.update(g,os_,p)
        return optax.apply_updates(p,up),os_,loss
    for it in range(500):
        inp=np.concatenate([rng.uniform(-1.5,1.5,(1024,2)),-rng.integers(1,4,(1024,1))],axis=1)
        pl,os_,loss=poly_step(pl,os_,jnp.asarray(inp))
    result['polynomial_warmstart_updates']=500;result['polynomial_warmstart_loss']=float(loss)
    trees={};cells={};masks={};params={'local':pl}
    for r in (4,8,16):
        cells[r],masks[r],trees[r]=LS.padded_trees(T,rootpred,r,a.seed)
        P=cells[r].shape[2];codes=np.zeros((8,len(Y),16+P*r));codes[:,:,:16]=np.asarray(zt)[None]
        cache={}
        for i in range(len(T['parameters'])):
            residual=Y[i*11:(i+1)*11]-rootpred[i*11:(i+1)*11]
            for q,cs in enumerate(trees[r][i]):
                ident=tuple(cs)
                if ident not in cache:cache[ident]=dec.local_matrix(pl,x,cs,r)
                c,_,_=fit.linear_fit(cache[ident],residual,w)
                codes[q,i*11:(i+1)*11,16:16+len(cs)*r]=np.asarray(c)
        params[f'codes{r}']=jnp.asarray(codes)
        print('residual_initialization',r,flush=True)
    result['status']='assembled_training';save(out/'result.json',result)
    opt=optax.adam(optax.cosine_decay_schedule(1e-3,a.updates-500,alpha=.01));os_=opt.init(params)
    def make_step(r):
        @jax.jit
        def step(params,os_,idx,pts,q,Y,norm,x,w,G,treecells,treemasks):
            pc=treecells[idx//11,q];mask=treemasks[idx//11,q]
            def loss(p):
                code=p[f'codes{r}'][q,idx];root=dec.head(parent,code[:,:16])@G[pts].T
                def single(cells):
                    return jax.vmap(lambda c:dec.local_features(p['local'],x[pts],c)[:,:r])(cells)
                B=jax.vmap(single)(pc)
                local=jnp.einsum('bpnr,bpr,bp->bn',B,code[:,16:].reshape(len(idx),-1,r),mask)
                delta=root+local-Y[idx[:,None],pts[None]]
                return jnp.mean(jnp.sum(delta**2*w[pts][None],axis=1)/norm[idx]**2)*len(x)/len(pts)
            loss,g=jax.value_and_grad(loss)(params);up,os_=opt.update(g,os_,params)
            return optax.apply_updates(params,up),os_,loss
        return step
    steps={r:make_step(r) for r in (4,8,16)};best=np.inf;stale=0;bestpl=None
    for it in range(501,a.updates+1):
        r=(4,8,16)[(it-501)%3];q=int(rng.choice([0,5,6,7] if it%2==0 else [1,2,3,4]))
        idx=jnp.asarray(rng.integers(len(Y),size=8));pts=jnp.asarray(rng.integers(len(x),size=512))
        params,os_,loss=steps[r](params,os_,idx,pts,q,Y,T['norm'],x,w,G,cells[r],masks[r])
        if it%1000==0 or it==a.updates:
            rows,saved=LS.evaluate_pair(model,params['local'],None,D,zd,4,25,3)
            # Select one shared checkpoint for both allocations, by the worse panel.
            metric=max(row['projection']['worst'] for row in rows)
            record=dict(update=it,loss=float(loss),comparisons=rows,worst=metric)
            record['checkpoint_sha256']=checkpoint(out/f'local_features_{it:05d}.pkl',dict(local=params['local'],metrics=record))
            result['checkpoints'].append(record);save(out/'result.json',result)
            print('local',it,metric,flush=True)
            if metric<best:
                improve=metric<best*.99;best=metric;bestpl=jax.device_get(params['local']);bestit=it
                stale=0 if improve else stale+1
                checkpoint(out/'direct_training_codes_selected.pkl',{k:v for k,v in params.items() if k!='local'})
            else:stale+=1
            if stale>=5:break
    pl=jax.tree_util.tree_map(jnp.asarray,bestpl)
    result['selected_update']=bestit;result['status']='complete_direct_local_representation_only'
    result['model_sha256']=checkpoint(out/'local_model.pkl',dict(local=pl,global_model=model,seed=a.seed,width=a.width,selected_update=bestit))
    save(out/'result.json',result)

if __name__=='__main__':main()
