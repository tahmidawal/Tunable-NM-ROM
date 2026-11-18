"""Shared zero-child head: residual initialization, then joint multi-depth physical fits."""
import argparse,optax
from common import *
from training_io import checkpoint,summary
import shared_child_head as H


def load_cache(path,group,out):
    record=json.loads((path/'result.json').read_text());assert record['status']=='complete_local_physical_cache_and_POD'
    assert record['root']==128 and record['nodes']==52
    fields={k:[] for k in ['S','b','constant','norm2','geometry','h','mask','teacher','AinvB','Ainvb']}
    rows=record['cases'][group];labels=[];source=[]
    for row in rows:
        f=path/row['file'];assert digest(f)==row['sha256'];v=np.load(f)
        S,b,constant,AinvB,Ainvb=H.schur(jnp.asarray(v['gram']),jnp.asarray(v['rhs']),jnp.asarray(v['norm2']))
        g,h=H.geometry(v['cells']);masks=np.array([v['cells'][:,0]<=d for d in (2,3,4)])
        teacher=[]
        for mask in masks:
            active=np.flatnonzero(np.repeat(mask,36));s=S[jnp.ix_(jnp.asarray(active),jnp.asarray(active))]
            c=jnp.linalg.solve(s+1e-8*jnp.eye(len(active)),b[:,active].T).T
            full=jnp.zeros((11,52*36)).at[:,active].set(c);teacher.append(np.asarray(full.reshape(11,52,36)))
        vals=dict(S=S,b=b,constant=constant,norm2=v['norm2'],geometry=g,h=h,mask=masks,
                  teacher=np.stack(teacher),AinvB=AinvB,Ainvb=Ainvb)
        for k,value in vals.items():fields[k].append(np.asarray(value))
        labels.append(row['case']);source.append(dict(file=str(f),sha256=row['sha256']))
    # Large matrices are explicit runtime arguments to compiled training kernels.
    arrays={k:jnp.asarray(np.stack(v)) for k,v in fields.items()}
    save(out/f'{group}_cache_sources.json',source)
    return arrays,labels


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--cache',required=True)
    ap.add_argument('--latent',type=int,required=True);ap.add_argument('--pretrain',type=int,default=4000)
    ap.add_argument('--updates',type=int,default=12000);ap.add_argument('--seed',type=int,default=2026111851)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();path=Path(a.cache)
    assert str(path).startswith(os.environ['STUDY_NAMESPACE']+'/')
    tr,tl=load_cache(path,'train',out);dv,vl=load_cache(path,'dev',out)
    local=tr['teacher'][:,2];norm=jnp.sqrt(tr['norm2'])
    p,scale=H.initialize(local,tr['h'][:,None,:],norm,a.latent,jax.random.PRNGKey(a.seed))
    rng=np.random.default_rng(a.seed);opt=optax.adam(optax.cosine_decay_schedule(1e-3,a.pretrain,alpha=.01));state=opt.init(p)
    geom=jnp.broadcast_to(tr['geometry'][:,None],local.shape[:-1]+(3,)).reshape(-1,3)
    hs=jnp.broadcast_to(tr['h'][:,None],local.shape[:-1]).reshape(-1)
    ns=jnp.broadcast_to(norm[:,:,None],local.shape[:-1]).reshape(-1)
    patches=local.reshape(-1,36)
    @jax.jit
    def pre_step(p,state,idx,patches,g,h,norm,scale):
        def loss(p):
            z=H.encode(p['encoder'],patches[idx],g[idx],h[idx],scale)
            c=H.head(p['head'],z,g[idx],h[idx],scale)
            return jnp.mean(jnp.sum((c-patches[idx])**2,axis=-1)/norm[idx]**2)
        v,grad=jax.value_and_grad(loss)(p);up,state=opt.update(grad,state,p)
        return optax.apply_updates(p,up),state,v
    result=dict(meta=meta,status='shared_child_pretraining',latent=a.latent,root=128,nodes=52,local_features=36,
        active_variables=128+52*a.latent,seed=a.seed,cache_sha256=digest(path/'result.json'),train_labels=tl,dev_labels=vl,
        training_allocations='prescribed initial-field swept tree; equal full-depth and shallower-truncation minibatches',
        limitations=['Analytical coordinate features; not a coordinate-MLP reproduction',
                     'Focused prescribed-tree training amendment; balanced uniform/random/oracle mixture not yet reproduced'],
        normalization='current-time physical L2',pretraining=[],checkpoints=[],final_test_opened=False)
    for it in range(1,a.pretrain+1):
        idx=jnp.asarray(rng.integers(len(patches),size=1024));p,state,v=pre_step(p,state,idx,patches,geom,hs,ns,scale)
        if it%1000==0:
            result['pretraining'].append(dict(update=it,loss=float(v)));print('child_pretrain',a.latent,it,float(v),flush=True)
            save(out/'result.json',result)
    del patches,geom,hs,ns
    project25=jax.jit(jax.vmap(H.make_project(25),in_axes=(None,0,0,0,0,None,None,None,None,None)))
    project75=jax.jit(jax.vmap(H.make_project(75),in_axes=(None,0,0,0,0,None,None,None,None,None)))
    def encoded(params,d,i,cfg):
        return H.encode(params['encoder'],d['teacher'][i,cfg],d['geometry'][i],d['h'][i],scale)
    def refit(params,d,i,cfg,start,which=project25):
        return which(d['S'][i],d['b'][i],d['constant'][i],d['norm2'][i],start,params['head'],
                     d['geometry'][i],d['h'][i],scale,d['mask'][i,cfg])
    codes=[]
    for i in range(len(tl)):
        configs=[]
        for cfg in range(3):configs.append(refit(p,tr,i,cfg,encoded(p,tr,i,cfg))[0])
        codes.append(jnp.stack(configs))
    p=dict(p,codes=jnp.stack(codes));opt=optax.adam(optax.cosine_decay_schedule(1e-3,a.updates,alpha=.01));state=opt.init(p)
    @jax.jit
    def assembled_step(p,state,i,cfg,S,b,constant,norm2,g,h,mask,teacher,scale):
        def loss(p):
            codes=p['codes'][i,cfg]
            ze=H.encode(p['encoder'],teacher[i,cfg],g[i],h[i],scale)
            def value(z):
                c=(H.head(p['head'],z,g[i],h[i],scale)*mask[i,cfg][None,:,None]).reshape(11,-1)
                return jnp.mean((jnp.einsum('ti,ij,tj->t',c,S[i],c)-2*jnp.sum(c*b[i],axis=1)+constant[i])/norm2[i])
            return .5*value(codes)+.5*value(ze)
        v,grad=jax.value_and_grad(loss)(p);up,state=opt.update(grad,state,p)
        return optax.apply_updates(p,up),state,v
    def evaluate(p):
        codes=[];errs=[];stats=[];accept=[];full=[]
        for i in range(len(vl)):
            ze=encoded(p,dv,i,2)
            results=[refit(p,dv,i,2,ze,project75),refit(p,dv,i,2,jnp.zeros_like(ze),project75)]
            which=jnp.argmin(jnp.stack([r[1] for r in results]),axis=0)
            z=jnp.stack([r[0] for r in results])[which,jnp.arange(11)]
            warm=jnp.concatenate([z[:1],z[:-1]],axis=0)
            results.append(refit(p,dv,i,2,warm,project75))
            which=jnp.argmin(jnp.stack([r[1] for r in results]),axis=0)
            chosen=[jnp.stack([r[k] for r in results])[which,jnp.arange(11)] for k in range(4)]
            z,e,s,n=chosen;codes.append(np.asarray(z));errs.append(np.asarray(e));stats.append(np.asarray(s));accept.append(np.asarray(n))
            c=H.head(p['head'],z,dv['geometry'][i],dv['h'][i],scale).reshape(11,-1)
            root=dv['Ainvb'][i]-c@dv['AinvB'][i].T
            full.append(np.asarray(jnp.concatenate([root,c],axis=1)))
        return dict(codes=np.stack(codes),errors=np.stack(errs),stationarity=np.stack(stats),accepted=np.stack(accept),coefficients=np.stack(full))
    best=(float('inf'),float('inf'));stale=0;result['status']='shared_child_assembled_training'
    for it in range(1,a.updates+1):
        i=int(rng.integers(len(tl)));cfg=int(rng.choice(3,p=[.25,.25,.5]))
        p,state,v=assembled_step(p,state,i,cfg,tr['S'],tr['b'],tr['constant'],tr['norm2'],tr['geometry'],tr['h'],tr['mask'],tr['teacher'],scale)
        if it%1000==0:
            tc=[];te=[];ts=[]
            for i in range(len(tl)):
                cs=[];es=[];ss=[]
                for cfg in range(3):
                    z,e,s,n=refit(p,tr,i,cfg,p['codes'][i,cfg])
                    cs.append(z);es.append(e);ss.append(s)
                tc.append(jnp.stack(cs));te.append(jnp.stack(es));ts.append(jnp.stack(ss))
            p=dict(p,codes=jnp.stack(tc));dev=evaluate(p);metric=(float(dev['errors'].max()),float(np.median(dev['errors'])))
            row=dict(update=it,loss=float(v),projection=summary(dev['errors']),train_full_projection=summary(jnp.stack(te)[:,2]),
                     train_shallow_projection=summary(jnp.stack(te)[:,:2]),max_stationarity=float(dev['stationarity'].max()),
                     train_max_stationarity=float(jnp.max(jnp.stack(ts))))
            row['sha256']=checkpoint(out/f'child_{a.latent}_{it:05d}.pkl',dict(params=p,scale=scale,dev=dev,metrics=row))
            result['checkpoints'].append(row);save(out/'result.json',result)
            print('child_assembled',a.latent,it,metric,'station',row['max_stationarity'],flush=True)
            if metric<best:
                significant=metric[0]<best[0]*.99;best=metric;stale=0 if significant else stale+1
                selected=dict(params=jax.device_get(p),scale=np.asarray(scale),dev=dev,latent=a.latent,selected_update=it,
                              cache_path=str(path),cache_sha256=result['cache_sha256'],train_labels=tl,dev_labels=vl)
                result['model_sha256']=checkpoint(out/'shared_child.pkl',selected);result['selected_update']=it
            else:stale+=1
            if it>=6000 and stale>=5:break
    result['worst_projection']=best[0];result['status']='complete_shared_child_representation'
    result['zero_jacobian']=H.zero_audit(selected['params']['head'],jnp.asarray(selected['scale']),dv['geometry'][0],dv['h'][0],dv['S'][0])
    assert result['zero_jacobian']['zero_child_max']<1e-13
    result['representation_gate']=best[0]<=.005
    save(out/'result.json',result)

if __name__=='__main__':main()
