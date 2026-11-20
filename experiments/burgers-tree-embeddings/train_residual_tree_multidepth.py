"""B8 continuation: independent jointly fitted codes for each tree layout/depth."""
import argparse,optax
from functools import partial
from common import *
from neural import stable as S,fields as F,allocations as A,code_optimizer as C,residual_tree as R
from neural.build_data import resolve
from training_io import checkpoint,read_checkpoint
import data
from neural import spatial_sampling as SS
from evaluate_residual_tree import load_group, owned_job, zero_child_audit
from extend_residual_fits import verified_manifest


def fresh_block_jacobian_audit(params, xy, child_k, out, training_root_code):
    """Small in-allocation derivative check for the actual fresh architecture."""
    started=time.perf_counter()
    first=A.children((0,0,0));cells=jnp.asarray(first+[c for parent in first for c in A.children(parent)])
    assert len(xy)==128**2
    axis_indices=np.array([16,48,80,112]);ix,iy=np.meshgrid(axis_indices,axis_indices,indexing='ij')
    indices=(128*ix+iy).reshape(-1);points=xy[jnp.asarray(indices)]
    dimension=32+20*child_k
    code=.2*jnp.sin(jnp.arange(dimension,dtype=jnp.float64)+1.)
    assert training_root_code.shape==(32,)
    code=code.at[:32].set(training_root_code+.01*jnp.sin(jnp.arange(32,dtype=jnp.float64)+1.))
    p=dict(params,B=R.feature_matrix(params['feature'],cells,points))
    full=jax.jit(jax.jacfwd(R.cached,argnums=1))(p,code,points)
    forward=jax.jit(R.cached_jacobian)(p,code,points)
    reverse=jax.jit(R.cached_jacobian_reverse)(p,code,points)
    jax.block_until_ready((full,forward,reverse))
    checks={}
    for name,left,right in [('forward_vs_full',forward,full),('reverse_vs_full',reverse,full),('reverse_vs_forward',reverse,forward)]:
        finite=bool(jnp.isfinite(left).all() & jnp.isfinite(right).all())
        absolute=float(jnp.max(jnp.abs(left-right))) if finite else None
        relative=float(jnp.linalg.norm(left-right)/jnp.maximum(jnp.linalg.norm(right),1e-30)) if finite else None
        checks[name]=dict(finite=finite,max_absolute=absolute,relative=relative,
            passed=finite and absolute<=1e-9 and relative<=1e-10)
    filename='fresh_block_jacobian_audit.pkl'
    sha=checkpoint(out/filename,dict(code=code,cells=cells,points=points,full_autodiff=full,forward_block=forward,reverse_block=reverse,checks=checks))
    record=dict(file=filename,sha256=sha,checks=checks,passed=all(v['passed'] for v in checks.values()),
        child_k=child_k,features=int(params['scale'].shape[0]),nodes=20,dimension=dimension,points=16,
        code_rule='Root: first retained training code +0.01*sin(arange(32)+1); children:0.2*sin(arange(D)+1)[32:]',
        point_rule='4x4 tensor indices[16,48,80,112] on128x128 quadrature; one point in each depth2 cell',
        absolute_tolerance=1e-9,relative_tolerance=1e-10,relative_norm='Frobenius relative to comparison target',
        elapsed_seconds=time.perf_counter()-started,scope='Fresh-model derivative parity only; not field accuracy or PDE trace equivalence')
    save(out/'fresh_block_jacobian_audit.json',record)
    assert record['passed'], 'Fresh architecture derivative parity failed; evidence retained'
    return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--updates',type=int,default=30000)
    ap.add_argument('--train-jobs',required=True);ap.add_argument('--dev-jobs',required=True);ap.add_argument('--warm-job',default='')
    ap.add_argument('--parent-job',default='b7parentrepair')
    ap.add_argument('--fresh-children',action='store_true')
    ap.add_argument('--assembled-jacobian',choices=['forward','reverse'],default='forward')
    ap.add_argument('--qualification-job',default='b8implprofile')
    ap.add_argument('--reuse-codes',action='store_true',help='Reuse verified root and selected multidepth codes from warm-job; reset all Adam clocks/moments and learning-rate schedules')
    ap.add_argument('--parent-initializer',choices=['newton','gn-only'],default='newton',help='Three60-step parent GN starts; optionally finish selected unfinished start with120 Newton steps. Does not alter assembled fitting.')
    ap.add_argument('--spatial-sampling',choices=['uniform-index','residual-mixture'],default='uniform-index')
    ap.add_argument('--family-sampling',choices=['legacy','split-primary'],default='legacy')
    ap.add_argument('--sampling-alpha',type=float,default=.5)
    ap.add_argument('--all-final-refit',action='store_true',help='Refit all times/all variants at final checkpoint; otherwise rotating time only')
    ap.add_argument('--child-k',type=int,default=16);ap.add_argument('--features',type=int,default=64);ap.add_argument('--nodes',type=int,default=20)
    ap.add_argument('--seed',type=int,default=2026111989);a=ap.parse_args()
    assert a.train_jobs and a.dev_jobs and a.nodes==20
    owned_job(a.parent_job);expanded_parent=a.parent_job!='b7parentrepair'
    assert bool(a.warm_job)!=a.fresh_children, 'Choose exactly one warm checkpoint or fresh children'
    assert not a.reuse_codes or a.warm_job
    if expanded_parent:
        assert a.fresh_children and not a.reuse_codes and a.assembled_jacobian=='reverse' and a.parent_initializer=='gn-only'
    assert a.updates>0
    variant_labels=['coarse4','uniform20','prescribed20'];variant_probabilities=[.2,.4,.4]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    pp=resolve(ns/a.parent_job/'output/stable_parent.pkl',ns);pr=json.loads((pp.parent/'result.json').read_text());assert digest(pp)==pr['selected_sha256']
    parent_retention=None;derivative_proof=None;wp=None
    if expanded_parent:
        assert pr['status']=='complete_expanded_global_selection' and pr['latent']==32 and not pr['final_test_opened']
        parent_retention=verified_manifest(a.parent_job,ns,{'output/result.json':pp.parent/'result.json','output/stable_parent.pkl':pp})
    if a.assembled_jacobian=='reverse':
        owned_job(a.qualification_job)
        top=resolve(ns/a.qualification_job/'output/result.json',ns);qp=top.parent/'jacobian/result.json';q=json.loads(qp.read_text())
        qret=verified_manifest(a.qualification_job,ns,{'output/result.json':top,'output/jacobian/result.json':qp})
        assert q['status']=='complete_profile' and len(q['states'])==3
        checks=[v['parity'][k] for v in q['states'] for k in ['root_jacobian','residual_jacobian']]
        assert all(v['passed'] and v['max_absolute']<=1e-9 and v['relative']<=1e-10 for v in checks)
        qm=json.loads((HERE/'manifests'/f'retention-{a.qualification_job}.json').read_text())
        hashes={v['file']:v['sha256'] for v in qm['files']};sm=top.parent.parent/'SOURCE.sha256';assert digest(sm)==hashes['SOURCE.sha256']
        entries=[line.split(maxsplit=1) for line in sm.read_text().splitlines() if line.strip()]
        for module in ['stable.py','residual_tree.py']:
            matches=[sha for sha,name in entries if name=='experiments/burgers-tree-embeddings/neural/'+module]
            assert len(matches)==1 and digest(HERE/'neural'/module)==matches[0]
        derivative_proof=dict(job=a.qualification_job,sha256=digest(qp),retention_sha256=qret,source_manifest_sha256=digest(sm),
            checks=checks,overall_pde_trace_passed=q['parity_passed'],scope='Derivative values only; no PDE trace claim')
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
    if expanded_parent:
        assert (train_count,dev_count)==(160,52) and root['modulation']['w'].shape[0]==32
        assert parent['train_labels']==pr['train_labels']==tr['labels'] and parent['dev_labels']==pr['dev_labels']==dv['labels']
        assert parent['expanded_sources']==pr['expanded_sources']==provenance
        assert np.asarray(parent['training_codes']).shape==(1760,32) and np.isfinite(parent['training_codes']).all()
        assert np.array_equal(parent['training_encoder'],np.asarray(tr['encoder']))
        for group,jobs in [('train',a.train_jobs),('dev',a.dev_jobs)]:
            checked=load_group(ns,group,jobs.split(','),arrays,parent)
            assert checked['labels']==groups[group]['labels'] and checked['provenance']==[v for v in provenance if v['cohort']==group]
            assert np.array_equal(checked['y'],groups[group]['y']) and np.array_equal(checked['norm'],groups[group]['norm'])
    params=R.initialize(jax.random.PRNGKey(a.seed),root,child_k=a.child_k,features=a.features,width=64)
    if a.warm_job:
        wp=resolve(ns/a.warm_job/'output/nonlinear_tree.pkl',ns);wr=json.loads((wp.parent/'result.json').read_text());assert digest(wp)==wr['selected_sha256']
        assert wr['status'].startswith('complete_')
        warm=read_checkpoint(wp);params=jax.tree.map(jnp.asarray,warm['params']);assert params['scale'].shape==(a.features,) and R.dimensions(params)==(32,a.child_k)
        assert warm['parent_sha256']==digest(pp)
    zero_preservation=None;fresh_derivative_audit=None
    if a.fresh_children:
        fresh_derivative_audit=fresh_block_jacobian_audit(params,xy,a.child_k,out,jnp.asarray(parent['training_codes'][0]))
        probe=xy[::1024];audit_cells=jnp.asarray([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
        audit_root=jnp.asarray(parent['training_codes'][0]);audit_code=jnp.concatenate([audit_root,jnp.zeros(4*a.child_k)])
        zero_preservation=float(jnp.max(jnp.abs(R.tree(params,audit_code,probe,audit_cells)-S.apply(root,audit_root,probe))))
        assert zero_preservation==0.
    reused=None;reuse_provenance=None;root_fit_records=[]
    if a.reuse_codes:
        assert wr['status']=='complete_nonlinear_child_multidepth_subset'
        assert wr['train_labels']==tr['labels'] and wr['dev_labels']==dv['labels']
        assert wr['expanded_sources']==provenance==warm['expanded_sources']
        assert wr['primary_data_sha256']==digest(dp)
        assert wr['nodes']==a.nodes and wr['child_k']==a.child_k and wr['features']==a.features
        assert wr['variant_labels']==variant_labels and warm['variant_labels']==variant_labels
        assert jax.tree.structure(params['root'])==jax.tree.structure(root)
        assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in
            zip(jax.tree.leaves(params['root']),jax.tree.leaves(root)))
        manifest_path=HERE/'manifests'/f'retention-{a.warm_job}.json'
        manifest=json.loads(manifest_path.read_text())
        assert manifest['name']==a.warm_job and manifest['local_and_remote_verified']
        assert manifest['terminal_state']=='COMPLETED'
        assert str(manifest['job'])==str(wr['meta']['job']) and manifest['source_commit']==wr['meta']['commit']
        retained={}
        for filename in ['initialization.pkl','nonlinear_tree.pkl','result.json']:
            entries=[r for r in manifest['files'] if r['file']=='output/'+filename]
            assert len(entries)==1
            path=wp.parent/filename
            assert path.stat().st_size==entries[0]['bytes'] and digest(path)==entries[0]['sha256']
            retained[filename]=entries[0]['sha256']
        reused=read_checkpoint(wp.parent/'initialization.pkl')
        assert reused['variant_labels']==variant_labels
        rootcodes=jnp.asarray(reused['root_codes']);devroot=jnp.asarray(reused['dev_root_codes'])
        assert rootcodes.shape==(len(tr['y']),32) and devroot.shape==(len(dv['y']),32)
        assert np.isfinite(np.asarray(rootcodes)).all() and np.isfinite(np.asarray(devroot)).all()
        reuse_provenance=dict(job=a.warm_job,retention_manifest=str(manifest_path.relative_to(HERE)),
            retention_manifest_sha256=digest(manifest_path),verified_files=retained,
            selected_update=wr['selected_update'],optimizer_state='Reset all moments and clocks; restart declared learning-rate schedule; not exact optimizer resume')
        print('verified_multidepth_code_reuse',a.warm_job,rootcodes.shape,devroot.shape,flush=True)
    elif expanded_parent:
        rootcodes=jnp.asarray(parent['training_codes']);dev_codes=[]
        rootfit=F.make_fit(S.apply,60,jacobian=S.code_jacobian)
        rootmore=F.make_fit(S.apply,120,jacobian=S.code_jacobian)
        for i in range(len(dv['y'])):
            near,interp=S.nearest_starts(dv['encoder'][i],tr['encoder'],rootcodes)
            candidates=[rootfit(root,z,xy,dv['y'][i],w,dv['norm'][i]) for z in [near,interp,jnp.zeros(32)]]
            assert all(np.isfinite(np.asarray(v)).all() for pair in candidates for v in pair)
            selected_start=min(range(3),key=lambda k:float(candidates[k][1][0]));z,st=candidates[selected_start]
            attempts=[dict(start=name,status=np.asarray(v[1])) for name,v in zip(['nearest_training_field','interpolated_training_fields','zero'],candidates)]
            if not bool(st[-1]):
                z,st=rootmore(root,z,xy,dv['y'][i],w,dv['norm'][i]);attempts.append(dict(start='selected_parent_gn_continuation',status=np.asarray(st)))
            assert np.isfinite(np.asarray(z)).all() and np.isfinite(np.asarray(st)).all()
            dev_codes.append(z);root_fit_records.append(dict(cohort='dev',index=i,case=dv['labels'][i//11],time=float(data.TIMES[i%11]),
                selected_gn_start=selected_start,attempts=attempts,status=np.asarray(st),converged=bool(st[-1])))
            if (i+1)%16==0 or i+1==len(dv['y']):
                checkpoint(out/'root_initialization_partial_dev.pkl',dict(root_codes=rootcodes,dev_root_codes=jnp.stack(dev_codes),
                    parent_sha256=digest(pp),expanded_sources=provenance,train_labels=tr['labels'],dev_labels=dv['labels'],
                    completed=i+1,expected=572,complete=False,new_fit_records=root_fit_records))
                print('expanded_parent_dev_fit',i+1,572,float(st[0]),bool(st[-1]),flush=True)
        devroot=jnp.stack(dev_codes)
    else:
        rootfit=F.make_fit(S.apply,60)
        rootmore=F.make_fit(S.apply,120,newton=True) if a.parent_initializer=='newton' else None
        rootcodes=[jnp.asarray(parent['training_codes'])]
        devfits=read_checkpoint(resolve(ns/'b7parentrepair/output/development_fits.pkl',ns))
        devroot=[jnp.asarray(devfits['codes'])]
        partial_codes=dict(train=np.asarray(rootcodes[0]),dev=np.asarray(devroot[0]))
        for group_name,group,code_list in [('train',tr,rootcodes),('dev',dv,devroot)]:
            old=len(code_list[0]);new=[]
            for i in range(old,len(group['y'])):
                near,interp=S.nearest_starts(group['encoder'][i],jnp.asarray(parent['training_encoder']),jnp.asarray(parent['training_codes']))
                candidates=[rootfit(root,z,xy,group['y'][i],w,group['norm'][i]) for z in [near,interp,jnp.zeros(32)]]
                selected_start=min(range(len(candidates)),key=lambda k:float(candidates[k][1][0]))
                z,st=candidates[selected_start]
                attempts=[dict(start=name,status=np.asarray(v[1])) for name,v in
                    zip(['nearest_training_field','interpolated_training_fields','zero'],candidates)]
                if not bool(st[-1]) and rootmore is not None:
                    z,st=rootmore(root,z,xy,group['y'][i],w,group['norm'][i])
                    attempts.append(dict(start='selected_parent_newton_continuation',status=np.asarray(st)))
                new.append(z)
                root_fit_records.append(dict(cohort=group_name,index=i,case=group['labels'][i//11],time=float(data.TIMES[i%11]),
                    selected_gn_start=selected_start,attempts=attempts,status=np.asarray(st),
                    converged=bool(st[-1]) and bool(np.isfinite(np.asarray(st)).all())))
                if (i-old+1)%16==0 or i==len(group['y'])-1:
                    print('extension_parent_fit',group_name,i-old+1,len(group['y'])-old,float(st[0]),bool(st[-1]),flush=True)
                    partial_codes[group_name]=np.concatenate([np.asarray(code_list[0]),np.asarray(new)],axis=0)
                    checkpoint(out/f'root_initialization_partial_{group_name}.pkl',dict(
                        root_codes=partial_codes['train'],dev_root_codes=partial_codes['dev'],
                        parent_sha256=digest(pp),expanded_sources=provenance,train_labels=tr['labels'],dev_labels=dv['labels'],
                        parent_initializer=a.parent_initializer,complete=False,
                        completed_counts={name:len(value) for name,value in partial_codes.items()},
                        expected_counts=dict(train=len(tr['y']),dev=len(dv['y'])),new_fit_records=root_fit_records,
                        role='Partial initialization evidence only; no reconstruction or PDE admission'))
            if new:code_list.append(jnp.stack(new))
        rootcodes=jnp.concatenate(rootcodes);devroot=jnp.concatenate(devroot)
    root_fit_summary={name:dict(new_fields=sum(r['cohort']==name for r in root_fit_records),
        unfinished=sum(r['cohort']==name and not r['converged'] for r in root_fit_records)) for name in ['train','dev']}
    root_initialization_sha256=checkpoint(out/'root_initialization.pkl',dict(root_codes=rootcodes,
        dev_root_codes=devroot,parent_sha256=digest(pp),expanded_sources=provenance,
        train_labels=tr['labels'],dev_labels=dv['labels'],new_fit_records=root_fit_records,
        parent_initializer=a.parent_initializer,new_field_fit_summary=root_fit_summary,
        initialization_only=True,reused=reuse_provenance))
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
    def representative_zero_audits(p,bank,stage,model_sha256=None):
        started=time.perf_counter();primary_count=len(dr['groups']['train']['labels'])
        primary=data.parameters('train',primary_count)
        narrow=np.flatnonzero((primary[:,0]==0)&(primary[:,3]<=.05))
        assert len(narrow)>0
        representatives=[('narrow_primary',int(narrow[0])),('ridge',tr['family'].index('ridge')),('two_pulse',tr['family'].index('two_pulse'))]
        rows=[]
        for family,ci in representatives:
            for variant,layout in enumerate(variant_labels):
                active=4 if variant==0 else a.nodes;k=32+active*a.child_k
                cell=jnp.asarray(cells[variant,ci,:active]);B=R.feature_matrix(p['feature'],cell,xy)
                audit=zero_child_audit(p,jnp.asarray(bank[variant,11*ci,:k]),B,xy,w)
                rows.append(dict(family=family,case=tr['labels'][ci],cohort='train',time=0.,layout=layout,cells=np.asarray(cell),audit=audit))
        passed=all(row['audit']['zero_child_preserved'] and row['audit']['useful_independent_directions'] for row in rows)
        filename=f'zero_child_audit_{stage}.pkl'
        sha=checkpoint(out/filename,dict(stage=stage,rows=rows,model_sha256=model_sha256,parent_sha256=digest(pp),
            passed=passed,scope='Three training trajectories at t0 across three layouts; not a global direction or accuracy gate'))
        return dict(file=filename,sha256=sha,passed=passed,fields=len(rows),elapsed_seconds=time.perf_counter()-started,
            cases=[tr['labels'][ci] for _,ci in representatives],scope='Representative training directions only; all-case qualification remains required')
    initial=[];initerr=[]
    if a.reuse_codes:
        assert np.array_equal(np.asarray(reused['cells']),cells), 'Initialization tree layouts changed'
        assert np.array_equal(np.asarray(warm['training_cells']),cells), 'Selected tree layouts changed'
        bank=jnp.asarray(warm['training_codes'])
        assert bank.shape==(3,snapshots,dimension) and np.isfinite(np.asarray(bank)).all()
        assert np.count_nonzero(np.asarray(bank[0,:,32+4*a.child_k:]))==0, 'Coarse bank padding must remain zero'
        codes=bank.reshape(3*snapshots,dimension)
    else:
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
        codes=jnp.concatenate(initial)
    assert codes.shape==(3*snapshots,dimension)
    initial_zero_audit=representative_zero_audits(params,codes.reshape(3,snapshots,dimension),'initial') if a.fresh_children else None
    co=C.initialize_masked(codes)
    trainable=dict(feature=params['feature'],head=params['head']);opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(optax.cosine_decay_schedule(3e-4,a.updates,alpha=.1)));no=opt.init(trainable)
    rate=optax.cosine_decay_schedule(3e-3,a.updates,alpha=.1)
    @partial(jax.jit,static_argnames=('active',))
    def step(net,codes,no,co,si,bank_ids,pi,importance,active,it,root,scale,xy,w,y,norm,cs):
        def loss(net,z):
            def one(zz,cell,points):
                p=dict(net,root=root,scale=scale)
                return R.tree(p,zz[:32+active*a.child_k],points,cell[:active])
            pred=jax.vmap(one)(z,cs[bank_ids],xy[pi]);target=y[si[:,None],pi]
            return jnp.mean(importance*((pred-target)/norm[si,None])**2)
        value,(gn,gz)=jax.value_and_grad(loss,argnums=(0,1))(net,codes[bank_ids]);up,no=opt.update(gn,no,net)
        mask=jnp.concatenate([jnp.ones(32,dtype=bool),jnp.repeat(jnp.arange(a.nodes)<active,a.child_k)])
        codes,co=C.update_masked(codes,co,bank_ids,gz,rate(it),mask[None,:]);return optax.apply_updates(net,up),codes,no,co,value
    assembled_jac=R.cached_jacobian_reverse if a.assembled_jacobian=='reverse' else R.cached_jacobian
    fit=F.make_fit(R.cached,60,jacobian=assembled_jac);more=F.make_fit(R.cached,120,jacobian=assembled_jac)
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
    sampler=None;strata=None;sampling_start=time.perf_counter()
    hostw=np.asarray(w)
    if a.family_sampling=='split-primary':
        primary_count=len(dr['groups']['train']['labels'])
        assert tr['labels'][:primary_count]==dr['groups']['train']['labels']
        strata=SS.stratify(data.parameters('train',primary_count),tr['family'][primary_count:])
    if a.spatial_sampling=='residual-mixture':
        # Fixed parent proposals use training truth only; assemble CDF in small chunks.
        probabilities=np.empty((snapshots,len(w)),dtype=np.float64)
        cdf=np.empty_like(probabilities)
        parent_prediction=jax.jit(lambda z:S.apply(root,z[:,None,:],xy[None]))
        for start in range(0,snapshots,4):
            stop=min(start+4,snapshots)
            residual=np.asarray(tr['y'][start:stop]-parent_prediction(rootcodes[start:stop]))
            chunk=SS.prepare(hostw,residual,a.sampling_alpha)
            probabilities[start:stop]=chunk['probabilities'];cdf[start:stop]=chunk['cdf']
        sampler=dict(weights=hostw,probabilities=probabilities,cdf=cdf,fields=snapshots,
                     points=len(w),cohort='train',alpha=a.sampling_alpha)
    sampling_record=dict(spatial=a.spatial_sampling,family=a.family_sampling,alpha=a.sampling_alpha,
        preparation_seconds=time.perf_counter()-sampling_start,
        proposal_bytes=0 if sampler is None else probabilities.nbytes+cdf.nbytes+hostw.nbytes,
        parent_sha256=digest(pp),root_initialization_sha256=root_initialization_sha256,
        training_data_sha256=digest(dp),expanded_sources=[r for r in provenance if r['cohort']=='train'],
        proposal_sha256=None if sampler is None else hashlib.sha256(probabilities.tobytes()).hexdigest(),
        strata=None if strata is None else {k:v for k,v in strata.items() if k!='groups'},
        importance='Original quadrature weight / sampling probability; full quadrature N*w',
        proposal_role='Fixed offline training-only frozen-parent residual; no development or online sampling')
    print('sampling_preparation',sampling_record,flush=True)
    result=dict(meta=metadata,status='training',parent_sha256=digest(pp),primary_data_sha256=digest(dp),expanded_sources=provenance,
        sampling=sampling_record,seed=a.seed,child_k=a.child_k,features=a.features,nodes=a.nodes,active_variables=32+a.nodes*a.child_k,
        parent_weights_frozen=True,final_test_opened=False,primary_only=False,train_labels=tr['labels'],dev_labels=dv['labels'],
        normalization='current-time reference physical L2',initial_errors=initerr,
        variant_labels=variant_labels,variant_probabilities=variant_probabilities,independent_codes_per_variant=True,
        bank_shape=[3,snapshots,dimension],bank_index='variant * snapshots + snapshot',
        objective='Family-balanced physical relative reconstruction; independent jointly fitted parent/child codes for each depth/layout; shared frozen root and shared learned feature/head weights',
        all_final_refit=a.all_final_refit,warm_job=a.warm_job,warm_checkpoint_sha256=None if wp is None else digest(wp),
        parent_job=a.parent_job,parent_retention_sha256=parent_retention,expanded_parent=expanded_parent,
        fresh_children=a.fresh_children,fresh_block_jacobian_audit=fresh_derivative_audit,
        representative_zero_audit_required=a.fresh_children,initial_zero_child_audit=initial_zero_audit,
        zero_child_preservation_max_abs=zero_preservation,assembled_jacobian=a.assembled_jacobian,derivative_qualification=derivative_proof,
        parent_training_admission='Completed retained matched parent; no standalone parent2percent gate. Representation/PDE gates remain separate',
        reuse_codes=a.reuse_codes,reuse_provenance=reuse_provenance,
        parent_initializer='reused' if a.reuse_codes else a.parent_initializer,
        new_parent_field_fit_summary=root_fit_summary,
        initialization_mode='Verified selected codes; no new residual-tangent initialization' if a.reuse_codes else 'New residual-tangent initialization',
        root_initialization_file='root_initialization.pkl',root_initialization_sha256=root_initialization_sha256,
        optimizer_continuation='All moments/clocks/schedules reset, not exact resume',
        selection_metric='Worst independent error across both full20 layouts; coarse4 is diagnostic only' ,history=[],representation_admitted=False,pde_admitted=False,
        admission='Selection subset diagnostic only; all-case independent scoring and actual4096 required before PDE',
        reference_status='Primary qualified; expanded spectral ladders with independent FD check pending')
    result['initialization_file']='initialization.pkl'
    result['initialization_sha256']=checkpoint(out/'initialization.pkl',dict(codes=codes.reshape(3,snapshots,dimension),cells=cells,root_codes=rootcodes,dev_root_codes=devroot,variant_labels=variant_labels,
        parent_sha256=digest(pp),expanded_sources=provenance,train_labels=tr['labels'],dev_labels=dv['labels'],reuse_provenance=reuse_provenance))
    save(out/'result.json',result)
    best=np.inf;selected=None
    family=np.repeat(np.array(tr['family']),11);families=sorted(set(family));family_indices=[np.flatnonzero(family==f) for f in families]
    for it in range(1,a.updates+1):
        full=it>a.updates-2000;batch=2 if full else 4
        if strata is None:
            pool=family_indices[int(rng.integers(len(families)))];si=rng.choice(pool,batch,replace=False)
        else:
            _,trajectories=SS.draw_stratum(rng,strata,batch);si=11*trajectories+rng.integers(11,size=batch)
        if full or sampler is None:
            pi=np.tile(np.arange(len(w)),(batch,1)) if full else rng.integers(len(w),size=(batch,512))
            importance=len(w)*w[pi]
        else:
            sampled=SS.draw(rng,sampler,si,512);pi=sampled['indices'];importance=jnp.asarray(sampled['importance_weights'])
        variant=int(rng.choice(3,p=variant_probabilities));active=4 if variant==0 else a.nodes
        bank_ids=variant*snapshots+si
        trainable,codes,no,co,value=step(trainable,codes,no,co,jnp.asarray(si),jnp.asarray(bank_ids),jnp.asarray(pi),importance,active,it-1,root,params['scale'],xy,w,tr['y'],tr['norm'],cellarray)
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
    if a.fresh_children:
        result['selected_zero_child_audit']=representative_zero_audits(selected['params'],selected['training_codes'],'selected',result['selected_sha256'])
        result['selected_zero_child_directions_pass']=result['selected_zero_child_audit']['passed']
        save(out/'result.json',result)
    ev=selected['evaluation'];coarse=evaluate(selected['params'],'coarse4',selection)
    result['comparison_sha256']=checkpoint(out/'allocation_comparison.pkl',ev+coarse)
    result['coarse4_diagnostic']=dict(worst=max(r['independent_error'] for r in coarse),unfinished=sum(not bool(r['status'][-1]) for r in coarse))
    result.update(status='complete_nonlinear_child_multidepth_subset',worst_subset_error=max(r['independent_error'] for r in ev),unfinished_subset_fits=sum(not bool(r['status'][-1]) for r in ev))
    save(out/'result.json',result)

if __name__=='__main__':main()
