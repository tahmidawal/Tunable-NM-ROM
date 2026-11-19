"""Residual-initialized shared nonlinear children and joint multidepth training."""
import argparse,optax
from common import *
from training_io import checkpoint,read_checkpoint
from neural import fields as F,allocations as A
from neural.initialize import missing_field
from neural.build_data import resolve


def apply(p,z,xy):return F.tree(p,z,xy,p['cells'])
def jac(p,z,xy):return F.tree_jacobian(p,z,xy,p['cells'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--updates',type=int,default=20000);ap.add_argument('--seed',type=int,default=2026111952)
    ap.add_argument('--parent-newton',action='store_true')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight()
    ns=Path(os.environ['STUDY_NAMESPACE']);parent=resolve(ns/'b6parent/output/neural_parent.pkl',ns)
    folder=parent.parent;report=json.loads((folder/'result.json').read_text());assert report['status']=='complete_coordinate_parent_diagnostic'
    assert digest(parent)==report['selected_sha256']
    saved=read_checkpoint(parent);cache=folder/'coordinate_data.npz';assert digest(cache)==saved['data_sha256']
    arrays=dict(np.load(cache));d={k:jnp.asarray(v) for k,v in arrays.items()}
    result=dict(meta=metadata,status='parent_stability_audit',parent_sha256=digest(parent),seed=a.seed,
        layouts=['uniform','prescribed','random','oracle'],nodes=20,max_depth=3,root_k=8,child_k=8,
        final_test_opened=False,history=[],representation_admitted=False,
        parent_stability_protocol='Resume unfinished parent fits at fixed checkpoint for120 additional diagnostic iterations; no PDE solver setting changed',
        parent_newton=a.parent_newton)
    rootparams=saved['params']['decoder'];parentfits=read_checkpoint(folder/'all_development_fits.pkl')
    parent_codes=np.array(parentfits['codes']);parent_status=np.array(parentfits['status'])
    fitparent=F.make_fit(F.root,120,newton=a.parent_newton)
    # This is a fixed-checkpoint qualification diagnostic, not another training recipe.
    for i in np.where(parent_status[:,-1]==0)[0]:
        code,status=fitparent(rootparams,jnp.asarray(parent_codes[i]),d['xy'],d['dev_y'][i],d['w'],d['dev_norm'][i])
        parent_codes[i]=np.asarray(code);parent_status[i]=np.asarray(status)
    result['parent_unfinished_after_audit']=int(np.sum(parent_status[:,-1]==0))
    result['parent_audit_sha256']=checkpoint(out/'parent_stability.pkl',dict(codes=parent_codes,status=parent_status))
    save(out/'result.json',result)
    if result['parent_unfinished_after_audit']:
        result['status']='stopped_parent_fit_stability';save(out/'result.json',result);return
    params=dict(root=rootparams,child=F.init_mlp(jax.random.PRNGKey(a.seed),[13,128,128,128,1],True))
    rng=np.random.default_rng(a.seed);xy=arrays['xy'];kinds=result['layouts'];layouts=[]
    for ci in range(128):
        kind=kinds[ci%4];truth=arrays['train_y'][11*ci:11*(ci+1)]
        cells=A.allocate(kind,xy,truth[0],a.seed+ci,target=truth if kind=='oracle' else None)
        assert A.validate(cells);layouts.append(cells)
    layouts=np.repeat(np.stack(layouts),11,axis=0);cells=jnp.asarray(layouts)
    root_fit=F.make_fit(F.root,60)
    initial_codes=[];initial_diagnostics=[]
    sample=rng.choice(len(xy),1024,replace=False);iw=d['w'][sample]*len(xy)/len(sample)
    for i in range(len(d['train_y'])):
        z0=F.intrinsic(saved['params']['intrinsic'],d['train_features'][i])
        rz,rs=root_fit(rootparams,z0,d['xy'],d['train_y'][i],d['w'],d['train_norm'][i])
        z,loss=missing_field(params,rz,cells[i],d['xy'][sample],d['train_y'][i,sample],iw,d['train_norm'][i])
        initial_codes.append(np.asarray(z));initial_diagnostics.append(dict(root_status=np.asarray(rs),child_loss=np.asarray(loss)))
        if i%128==0:print('tree_initialization',i,float(loss[-1]),flush=True)
    result['initialization_sha256']=checkpoint(out/'initialization.pkl',dict(codes=np.stack(initial_codes),
        cells=layouts,diagnostics=initial_diagnostics,point_indices=sample))
    p=dict(network=params,codes=jnp.asarray(np.stack(initial_codes)))
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(optax.cosine_decay_schedule(2e-4,a.updates,alpha=.05)))
    state=opt.init(p)
    @jax.jit
    def step(p,state,ids,point_ids,active,iteration,xy,w,y,norm,cells):
        def loss(p):
            codes=p['codes'][ids]
            mask=jnp.concatenate([jnp.ones(8),jnp.repeat(jnp.arange(20)<active,8)])
            codes=codes*mask
            pred=jax.vmap(lambda z,c,x:F.tree(p['network'],z,x,c))(codes,cells[ids],xy[point_ids])
            error=(pred-y[ids[:,None],point_ids])/norm[ids,None]
            reconstruction=jnp.mean(len(w)*w[point_ids]*error**2)
            regular=sum(jnp.mean(layer['w']**2) for layer in p['network']['child'])
            return reconstruction+1e-6*regular+1e-7*jnp.mean(codes**2)
        value,grad=jax.value_and_grad(loss)(p)
        updates,state=opt.update(grad,state,p)
        # First1000 updates warm the child network/codes around the trained parent.
        updates['network']['root']=jax.tree.map(lambda u:jnp.where(iteration<1000,0.,u),updates['network']['root'])
        updates['codes']=updates['codes']*10
        updates['codes']=updates['codes'].at[:,:8].set(jnp.where(iteration<1000,0.,updates['codes'][:,:8]))
        return optax.apply_updates(p,updates),state,value
    fit=F.make_fit(apply,60,jacobian=jac)
    ids=np.array([11*c+j for c in [0,3,13,43] for j in [0,5,10]])
    devlayouts={kind:np.stack([A.allocate(kind,xy,arrays['dev_y'][11*c],a.seed+c) for c in range(44)])
                for kind in ['uniform','prescribed']}
    def evaluate(network,indices,kinds):
        rows=[]
        for kind in kinds:
            for i in indices:
                cell=jnp.asarray(devlayouts[kind][i//11]);pp=dict(network,cells=cell)
                rz=parent_codes[i]
                # Reference-assisted initialization is OFFLINE representation fitting only.
                z,_=missing_field(network,jnp.asarray(rz),cell,d['xy'][sample],d['dev_y'][i,sample],iw,d['dev_norm'][i])
                z,status=fit(pp,z,d['xy'],d['dev_y'][i],d['w'],d['dev_norm'][i])
                rows.append(dict(layout=kind,index=int(i),codes=np.asarray(z),cells=np.asarray(cell),status=np.asarray(status)))
        return rows
    best=(np.inf,np.inf);stale=0;selected=None
    for it in range(1,a.updates+1):
        si=jnp.asarray(rng.integers(len(d['train_y']),size=2));ci=jnp.asarray(rng.integers(len(d['w']),size=(2,512)))
        active=int(rng.choice([0,4,8,12,16,20],p=[.1,.1,.1,.1,.1,.5]))
        p,state,value=step(p,state,si,ci,jnp.int32(active),jnp.int32(it),d['xy'],d['w'],d['train_y'],d['train_norm'],cells)
        if it%2000:continue
        ev=evaluate(p['network'],ids,['prescribed']);errors=np.array([r['status'][0] for r in ev])
        metric=(float(errors.max()),float(np.median(errors)))
        row=dict(update=it,loss=float(value),subset_worst=metric[0],subset_median=metric[1],
                 unfinished_fits=sum(int(r['status'][-1]==0) for r in ev))
        payload=dict(network=jax.device_get(p['network']),training_codes=np.asarray(p['codes']),cells=layouts,
                     evaluation=ev,parent_sha256=digest(parent),seed=a.seed,metrics=row)
        row['sha256']=checkpoint(out/f'tree_{it:05d}.pkl',payload);result['history'].append(row)
        print('coordinate_tree',row,flush=True)
        if metric<best:
            significant=metric[0]<best[0]*.99;stale=0 if significant else stale+1;best=metric;selected=payload
            result['selected_sha256']=checkpoint(out/'neural_tree.pkl',payload)
        else:stale+=1
        result['status']='training_neural_tree';save(out/'result.json',result)
        if it>=10000 and stale>=5:break
    # Full admission/independent-grid scoring is a separate bounded job only if
    # this diagnostic can meet the threshold. Preserve unsuccessful checkpoints.
    ev=evaluate(selected['network'],ids,['uniform','prescribed'])
    result['comparison_sha256']=checkpoint(out/'allocation_comparison.pkl',ev)
    result.update(status='complete_neural_tree_subset_diagnostic',selection_subset=ids,
        worst_subset_error=max(float(r['status'][0]) for r in ev),
        unfinished_subset_fits=sum(int(r['status'][-1]==0) for r in ev),
        admission='Diagnostic subset only. Full44x11 independent scoring, Jacobian audits and actual4096 outputs remain mandatory before any PDE admission.')
    save(out/'result.json',result)


if __name__=='__main__':main()
