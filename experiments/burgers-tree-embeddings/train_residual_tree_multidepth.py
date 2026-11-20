"""B8 continuation: independent jointly fitted codes for each tree layout/depth."""
import argparse,optax
from functools import partial
from common import *
from neural import stable as S,fields as F,allocations as A,code_optimizer as C,residual_tree as R
from neural.build_data import resolve
from training_io import checkpoint,read_checkpoint
import data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--updates',type=int,default=30000)
    ap.add_argument('--train-jobs',required=True);ap.add_argument('--dev-jobs',required=True);ap.add_argument('--warm-job',required=True)
    ap.add_argument('--all-final-refit',action='store_true',help='Refit all times/all variants at final checkpoint; otherwise rotating time only')
    ap.add_argument('--child-k',type=int,default=16);ap.add_argument('--features',type=int,default=64);ap.add_argument('--nodes',type=int,default=20)
    ap.add_argument('--seed',type=int,default=2026111989);a=ap.parse_args()
    assert a.train_jobs and a.dev_jobs and a.warm_job and a.nodes==20
    assert a.updates>0
    variant_labels=['coarse4','uniform20','prescribed20'];variant_probabilities=[.2,.4,.4]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    pp=resolve(ns/'b7parentrepair/output/stable_parent.pkl',ns);pr=json.loads((pp.parent/'result.json').read_text());assert digest(pp)==pr['selected_sha256']
    parent=read_checkpoint(pp);dp=resolve(ns/'b6parent/output/coordinate_data.npz',ns);assert digest(dp)==parent['data_sha256']
    arrays=dict(np.load(dp));dr=parent['data_record'];root=jax.tree.map(jnp.asarray,parent['network'])
    xy,w=jnp.asarray(arrays['xy']),jnp.asarray(arrays['w']);rng=np.random.default_rng(a.seed)
    groups={};provenance=[]
    for group,jobs in [('train',a.train_jobs),('dev',a.dev_jobs)]:
        groups[group]=dict(y=[arrays[group+'_y']],encoder=[arrays[group+'_encoder']],norm=[arrays[group+'_norm']],
                           labels=list(dr['groups'][group]['labels']),sources=list(dr['groups'][group]['sources']),family=['primary']*len(dr['groups'][group]['labels']))
        joblist=list(filter(None,jobs.split(',')));assert len(joblist)==len(set(joblist))
        for job in joblist:
            rp=resolve(ns/job/'output/result.json',ns);record=json.loads(rp.read_text());assert record['cohort']==group
            assert record['status']=='complete_expanded_ladders_pending_independent_crosscheck' and record['all_ladders_pass']
            assert not record['final_test_opened']
            cache=rp.parent/'coordinate_data.npz';assert digest(cache)==record['coordinate_data']['sha256'];v=np.load(cache)
            assert np.array_equal(v['xy'],arrays['xy']) and np.array_equal(v['w'],arrays['w'])
            assert np.array_equal(v['encoder_xy'],arrays['encoder_xy'])
            assert np.array_equal(v['case_ids'],np.repeat([r['case'] for r in record['cases']],11))
            assert np.array_equal(v['times'],np.tile(data.TIMES,len(record['cases'])))
            provenance.append(dict(job=job,record_sha256=digest(rp),cache_sha256=digest(cache),cohort=group))
            for key in ['y','encoder','norm']:groups[group][key].append(v[key])
            for row in record['cases']:
                assert row['provisional_ladder_pass']
                level=row['levels'][-1];groups[group]['labels'].append(row['case']);groups[group]['family'].append(row['family'])
                groups[group]['sources'].append(dict(case=row['case'],file=str(rp.parent/level['file']),sha256=level['sha256']))
        for key in ['y','encoder','norm']:groups[group][key]=jnp.asarray(np.concatenate(groups[group][key]))
    tr,dv=groups['train'],groups['dev'];train_count=len(tr['labels']);dev_count=len(dv['labels'])
    assert len(set(tr['labels'])&set(dv['labels']))==0
    assert len(set(tr['labels']))==train_count and len(set(dv['labels']))==dev_count
    params=R.initialize(jax.random.PRNGKey(a.seed),root,child_k=a.child_k,features=a.features,width=64)
    if a.warm_job:
        wp=resolve(ns/a.warm_job/'output/nonlinear_tree.pkl',ns);wr=json.loads((wp.parent/'result.json').read_text());assert digest(wp)==wr['selected_sha256']
        assert wr['status'].startswith('complete_')
        warm=read_checkpoint(wp);params=jax.tree.map(jnp.asarray,warm['params']);assert params['scale'].shape==(a.features,) and R.dimensions(params)==(32,a.child_k)
        assert warm['parent_sha256']==digest(pp)
    rootfit=F.make_fit(S.apply,60);rootmore=F.make_fit(S.apply,120,newton=True)
    rootcodes=[jnp.asarray(parent['training_codes'])];devfits=read_checkpoint(resolve(ns/'b7parentrepair/output/development_fits.pkl',ns))
    devroot=[jnp.asarray(devfits['codes'])]
    for group,codes in [(tr,rootcodes),(dv,devroot)]:
        old=len(codes[0]);new=[]
        for i in range(old,len(group['y'])):
            near,interp=S.nearest_starts(group['encoder'][i],jnp.asarray(parent['training_encoder']),jnp.asarray(parent['training_codes']))
            candidates=[rootfit(root,z,xy,group['y'][i],w,group['norm'][i]) for z in [near,interp,jnp.zeros(32)]]
            z,st=min(candidates,key=lambda v:float(v[1][0]))
            if not bool(st[-1]):z,st=rootmore(root,z,xy,group['y'][i],w,group['norm'][i])
            new.append(z)
        if new:codes.append(jnp.stack(new))
    rootcodes=jnp.concatenate(rootcodes);devroot=jnp.concatenate(devroot)
    # Every trajectory gets all layouts; no family/parity confounding.
    cells=[]
    for variant in variant_labels:
        kind='prescribed' if variant=='prescribed20' else 'uniform'
        # coarse4 stores the same retained prefix, padded only in the code bank.
        layout=np.stack([A.allocate(kind,np.asarray(xy),np.asarray(tr['y'][11*ci]),
            a.seed+ci,nodes=a.nodes,max_depth=3) for ci in range(train_count)])
        for cell in layout:A.validate(cell)
        cells.append(layout)
    cells=np.stack(cells)  # variant, trajectory, node, geometry
    snapshots=len(tr['y']);dimension=32+a.nodes*a.child_k
    cellarray=jnp.asarray(np.repeat(cells,11,axis=1).reshape(3*snapshots,a.nodes,3))
    @jax.jit
    def initialize_codes(p,rz,cell,y,norm,xy,w):
        B=R.feature_matrix(p['feature'],cell,xy);zero=jnp.zeros(a.child_k)
        D=jax.jacfwd(R.coefficients,argnums=1)(p['head'],zero,p['scale'])
        J=jnp.einsum('pnr,rk->pnk',B,D).reshape(len(xy),-1);Jw=J*jnp.sqrt(w)[:,None]
        H=Jw.T@Jw;scale=jax.lax.rsqrt(jnp.maximum(jnp.diag(H),1e-14))
        pred=S.apply(p['root'],rz[:,None,:],xy[None]);rhs=(y-pred)@(w[:,None]*J)
        z=(jnp.linalg.solve(H*scale[:,None]*scale[None,:]+1e-3*jnp.eye(H.shape[0]),(rhs*scale).T).T)*scale
        zz=z.reshape(len(y),len(cell),a.child_k);zz=zz*jnp.minimum(1.,2/jnp.maximum(jnp.linalg.norm(zz,axis=-1,keepdims=True),1e-30))
        def candidate(alpha):
            cs=R.coefficients(p['head'],alpha*zz,p['scale']);yp=pred+jnp.einsum('pnr,tnr->tp',B,cs)
            return jnp.sum(w*(yp-y)**2,axis=1)/norm**2
        alphas=jnp.array([0.,1.,.5,.25,.125]);loss=jax.vmap(candidate)(alphas);ix=jnp.argmin(loss,axis=0)
        z=zz*alphas[ix,None,None]
        return jnp.concatenate([rz,z.reshape(len(y),-1)],axis=1),jnp.sqrt(jnp.min(loss,axis=0))
    initial=[];initerr=[]
    for variant in range(3):
        nodes=4 if variant==0 else a.nodes
        errors=[]
        for ci in range(train_count):
            sl=slice(11*ci,11*(ci+1))
            z,e=initialize_codes(params,rootcodes[sl],jnp.asarray(cells[variant,ci,:nodes]),
                tr['y'][sl],tr['norm'][sl],xy,w)
            initial.append(jnp.pad(z,((0,0),(0,dimension-z.shape[1]))))
            errors.extend(np.asarray(e).tolist())
            if ci%16==0:print('residual_initialization',variant_labels[variant],ci,float(jnp.max(e)),flush=True)
        initerr.append(errors)
    codes=jnp.concatenate(initial);assert codes.shape==(3*snapshots,dimension)
    co=C.initialize_masked(codes)
    trainable=dict(feature=params['feature'],head=params['head']);opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(optax.cosine_decay_schedule(3e-4,a.updates,alpha=.1)));no=opt.init(trainable)
    rate=optax.cosine_decay_schedule(3e-3,a.updates,alpha=.1)
    @partial(jax.jit,static_argnames=('active',))
    def step(net,codes,no,co,si,bank_ids,pi,active,it,root,scale,xy,w,y,norm,cs):
        def loss(net,z):
            def one(zz,cell,points):
                p=dict(net,root=root,scale=scale)
                return R.tree(p,zz[:32+active*a.child_k],points,cell[:active])
            pred=jax.vmap(one)(z,cs[bank_ids],xy[pi]);target=y[si[:,None],pi]
            return jnp.mean(len(w)*w[pi]*((pred-target)/norm[si,None])**2)
        value,(gn,gz)=jax.value_and_grad(loss,argnums=(0,1))(net,codes[bank_ids]);up,no=opt.update(gn,no,net)
        mask=jnp.concatenate([jnp.ones(32,dtype=bool),jnp.repeat(jnp.arange(a.nodes)<active,a.child_k)])
        codes,co=C.update_masked(codes,co,bank_ids,gz,rate(it),mask[None,:]);return optax.apply_updates(net,up),codes,no,co,value
    fit=F.make_fit(R.cached,60,jacobian=R.cached_jacobian);more=F.make_fit(R.cached,120,jacobian=R.cached_jacobian)
    xf,wf=data.grid(256,True);xf,wf=jnp.asarray(xf),jnp.asarray(wf)
    selection=[11*c+t for c in [0,3,13,20] for t in [0,5,10]]
    selection += [11*c+t for c in range(44,dev_count) for t in [0,5,10]]
    fine={}
    for ci in sorted(set(i//11 for i in selection)):
        row=dv['sources'][ci];p=resolve(row['file'],ns);assert digest(p)==row['sha256']
        fine[ci]=data.synth_grid(jnp.asarray(np.load(p)['coefficients']),jnp.unique(xf[:,0])).reshape(11,-1)
    def evaluate(p,kind,indices):
        rows=[];cache={}
        for i in indices:
            ci,ti=divmod(i,11)
            if ci not in cache:
                cache.clear()
                cell=jnp.asarray(A.allocate('uniform' if kind=='coarse4' else kind,np.asarray(xy),np.asarray(dv['y'][11*ci]),a.seed+ci,nodes=4 if kind=='coarse4' else a.nodes,max_depth=3))
                cache[ci]=(cell,R.feature_matrix(p['feature'],cell,xy),R.feature_matrix(p['feature'],cell,xf))
            cell,B,Bf=cache[ci];z,ie=initialize_codes(p,devroot[i:i+1],cell,dv['y'][i:i+1],dv['norm'][i:i+1],xy,w)
            cp=dict(p,B=B);z,st=fit(cp,z[0],xy,dv['y'][i],w,dv['norm'][i])
            if not bool(st[-1]):z,st=more(cp,z,xy,dv['y'][i],w,dv['norm'][i])
            pred=R.cached(dict(p,B=Bf),z,xf);err=float(jnp.sqrt(jnp.sum(wf*(pred-fine[ci][ti])**2))/dv['norm'][i])
            rows.append(dict(index=i,case=dv['labels'][ci],time=float(data.TIMES[ti]),layout=kind,codes=np.asarray(z),cells=np.asarray(cell),status=np.asarray(st),independent_error=err,scoring_change=abs(err-float(st[0]))))
        return rows
    result=dict(meta=metadata,status='training',parent_sha256=digest(pp),primary_data_sha256=digest(dp),expanded_sources=provenance,
        seed=a.seed,child_k=a.child_k,features=a.features,nodes=a.nodes,active_variables=32+a.nodes*a.child_k,
        parent_weights_frozen=True,final_test_opened=False,primary_only=False,train_labels=tr['labels'],dev_labels=dv['labels'],
        normalization='current-time reference physical L2',initial_errors=initerr,
        variant_labels=variant_labels,variant_probabilities=variant_probabilities,independent_codes_per_variant=True,
        bank_shape=[3,snapshots,dimension],bank_index='variant * snapshots + snapshot',
        objective='Family-balanced physical relative reconstruction; independent jointly fitted parent/child codes for each depth/layout; shared frozen root and shared learned feature/head weights',
        all_final_refit=a.all_final_refit,warm_job=a.warm_job,warm_checkpoint_sha256=digest(wp),
        selection_metric='Worst independent error across both full20 layouts; coarse4 is diagnostic only' ,history=[],representation_admitted=False,pde_admitted=False,
        admission='Selection subset diagnostic only; all-case independent scoring and actual4096 required before PDE',
        reference_status='Primary qualified; expanded spectral ladders with independent FD check pending')
    checkpoint(out/'initialization.pkl',dict(codes=codes.reshape(3,snapshots,dimension),cells=cells,root_codes=rootcodes,dev_root_codes=devroot,variant_labels=variant_labels))
    save(out/'result.json',result)
    best=np.inf;selected=None
    family=np.repeat(np.array(tr['family']),11);families=sorted(set(family));family_indices=[np.flatnonzero(family==f) for f in families]
    for it in range(1,a.updates+1):
        full=it>a.updates-2000;batch=2 if full else 4;pool=family_indices[int(rng.integers(len(families)))];si=rng.choice(pool,batch,replace=False)
        pi=np.tile(np.arange(len(w)),(batch,1)) if full else rng.integers(len(w),size=(batch,512))
        variant=int(rng.choice(3,p=variant_probabilities));active=4 if variant==0 else a.nodes
        bank_ids=variant*snapshots+si
        trainable,codes,no,co,value=step(trainable,codes,no,co,jnp.asarray(si),jnp.asarray(bank_ids),jnp.asarray(pi),active,it-1,root,params['scale'],xy,w,tr['y'],tr['norm'],cellarray)
        if it%1000==0:
            print('child_training',it,float(value),flush=True)
            checkpoint(out/'latest_training.pkl',dict(params=dict(trainable,root=root,scale=params['scale']),training_codes=codes.reshape(3,snapshots,dimension),training_cells=cells,variant_labels=variant_labels,update=it,parent_sha256=digest(pp),expanded_sources=provenance))
        if it%5000 and it!=a.updates:continue
        params=dict(trainable,root=root,scale=params['scale'])
        refresh_status=[];refreshed=[];refresh_variants=[]
        for variant in range(3):
            nodes=4 if variant==0 else a.nodes;k=32+nodes*a.child_k
            for ci in range(train_count):
                B=R.feature_matrix(params['feature'],jnp.asarray(cells[variant,ci,:nodes]),xy);cp=dict(params,B=B)
                times=range(11) if it==a.updates and a.all_final_refit else [((it+4999)//5000-1)%11]
                for ti in times:
                    i=11*ci+ti;bi=variant*snapshots+i
                    z,st=fit(cp,codes[bi,:k],xy,tr['y'][i],w,tr['norm'][i])
                    if not bool(st[-1]):z,st=more(cp,z,xy,tr['y'][i],w,tr['norm'][i])
                    codes=codes.at[bi,:k].set(z);refreshed.append(bi);refresh_status.append(np.asarray(st));refresh_variants.append(variant)
                if ci%32==0:print('child_code_refresh',it,variant_labels[variant],ci,flush=True)
        ri=jnp.asarray(refreshed)
        co={k:v.at[ri].set(jnp.zeros_like(v[ri])) for k,v in co.items()}
        refresh_status=np.stack(refresh_status);refresh_variants=np.array(refresh_variants)
        refresh_record=dict(count=len(refreshed),all_times=bool(it==a.updates and a.all_final_refit),
            snapshot_times=list(data.TIMES) if it==a.updates and a.all_final_refit else [float(data.TIMES[((it+4999)//5000-1)%11])],
            variants={variant_labels[v]:dict(worst=float(refresh_status[refresh_variants==v,0].max()),
                unfinished=int(np.sum(refresh_status[refresh_variants==v,-1]==0))) for v in range(3)})
        checkpoint(out/f'code_refresh_{it:06d}.pkl',dict(bank_indices=np.array(refreshed),codes=codes[ri],
            status=refresh_status,variant_labels=variant_labels,variant_indices=refresh_variants))
        ev=evaluate(params,'uniform',selection)+evaluate(params,'prescribed',selection)
        worst=max(r['independent_error'] for r in ev)
        row=dict(update=it,loss=float(value),training_refit=refresh_record,subset_worst=worst,
            subset_median=float(np.median([r['independent_error'] for r in ev])),
            unfinished=sum(not bool(r['status'][-1]) for r in ev))
        payload=dict(params=params,training_codes=codes.reshape(3,snapshots,dimension),training_cells=cells,
            variant_labels=variant_labels,evaluation=ev,metrics=row,parent_sha256=digest(pp),expanded_sources=provenance)
        row['sha256']=checkpoint(out/f'tree_{it:06d}.pkl',payload);result['history'].append(row)
        if worst<best:best=worst;result['selected_sha256']=checkpoint(out/'nonlinear_tree.pkl',payload);result['selected_update']=it;selected=payload
        save(out/'result.json',result);print('child_selection',row,flush=True)
    ev=selected['evaluation'];coarse=evaluate(selected['params'],'coarse4',selection)
    result['comparison_sha256']=checkpoint(out/'allocation_comparison.pkl',ev+coarse)
    result['coarse4_diagnostic']=dict(worst=max(r['independent_error'] for r in coarse),unfinished=sum(not bool(r['status'][-1]) for r in coarse))
    result.update(status='complete_nonlinear_child_multidepth_subset',worst_subset_error=max(r['independent_error'] for r in ev),unfinished_subset_fits=sum(not bool(r['status'][-1]) for r in ev))
    save(out/'result.json',result)

if __name__=='__main__':main()
