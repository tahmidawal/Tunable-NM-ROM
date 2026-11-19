"""Bounded same-GPU query diagnostic, requiring explicit development admission.

The default is one equivalence invocation per case/method. Paired repeated
timing is an explicit mode and must have its own finite registration. This
driver does not activate an allocation or claim native NM-ROM confirmation.
"""
import argparse
from common import *
from training_io import read_checkpoint,checkpoint
from build_local_cache import cohort,expand_matrix
from direct_query import FixedQuery
import data,cell_catalogue as CC,tensor_metric as TM


def load_frozen(a):
    cp,sp,gp,ap=map(Path,[a.cache,a.span,a.stable_global,a.admission])
    assert all(str(p).startswith(os.environ['STUDY_NAMESPACE']+'/') for p in [cp,sp,gp,ap])
    cr,sr,gr,ar=[json.loads((p/'result.json').read_text()) for p in [cp,sp,gp,ap]]
    assert ar['representation_gate'] and digest(cp/'result.json')==ar['cache_sha256']
    assert digest(cp/'local_POD.pkl')==cr['local_POD_sha256']==ar['checkpoint_sha256']
    assert digest(sp/'pod_basis.npz')==sr['basis_sha256']==gr['base_sha256']
    assert digest(gp/'stable_global_basis.npz')==gr['basis_sha256']
    full=np.load(sp/'pod_basis.npz')['coefficients'];root=full[:128]
    assert np.array_equal(root,np.load(cp/'global_basis.npz')['coefficients'])
    refs={v['case']:v['reference_sha256'] for v in cr['cases']['train']}
    assert refs=={v['case']:v['sha256'] for v in sr['training_sources']}=={v['case']:v['sha256'] for v in gr['training_sources']}
    U=read_checkpoint(cp/'local_POD.pkl')['rotation'][:,:24]
    glob=np.load(gp/'stable_global_basis.npz')['coefficients'][:960]
    return root,U,glob,dict(admission_sha256=digest(ap/'result.json'),global_basis_sha256=sr['basis_sha256'],
        local_rotation_sha256=cr['local_POD_sha256'],global_control_sha256=gr['basis_sha256'])


def score(run,basis,rotation,reference,refrow,old,oldpath,out,label,method,tag,output_count):
    if not run['complete']:
        return {k:v for k,v in run.items() if k not in ['fields','coefficients']},False
    C=jnp.asarray(run['coefficients']);bb=jnp.asarray(reference);norm2=jnp.sum(bb*bb,axis=(1,2))*.25
    assert C.shape[0]==6
    if method=='tree_direct24':
        metric=TM.make_geometry(6);G,tr,mx,my=metric(jnp.asarray(basis),jnp.asarray(run['cells']),*TM.rule())
        T=expand_matrix(jnp.asarray(rotation),len(run['cells']),len(basis));Gram=T.T@G@T
        rhs=TM.rhs(jnp.asarray(basis),bb,mx,my)@T
        assert np.array_equal(old['cells'],run['cells'])
    else:
        tr=[];Gram=jnp.eye(len(basis));N=basis.shape[-1]
        rhs=.25*jnp.einsum('tij,kij->tk',bb[:,:N,:N],jnp.asarray(basis))
    err2=jnp.einsum('ti,ij,tj->t',C,Gram,C)-2*jnp.sum(C*rhs,axis=1)+norm2
    errors=np.asarray(jnp.sqrt(jnp.maximum(err2,0)/norm2))
    delta=C-jnp.asarray(old['coefficients'])
    changes=np.asarray(jnp.sqrt(jnp.maximum(jnp.einsum('ti,ij,tj->t',delta,Gram,delta),0)/norm2))
    indices=[5] if output_count==1 else list(range(1,6));axis=np.linspace(0,1,4096)
    actual=[];fields=[]
    # Actual grid scoring and host transfer are outside the device-output timer.
    t0=time.perf_counter();host=[np.asarray(v) for v in run['fields']];transfer=time.perf_counter()-t0
    for index,pred,arr in zip(indices,run['fields'],host):
        assert arr.shape==(4096,4096) and arr.dtype==np.float64
        truth=data.synth_grid(bb[index],axis)
        actual.append(float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth)))
        rawhash=hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()
        dest=out/'fields'/f'{rawhash}.npz';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():np.savez_compressed(dest,prediction=arr)
        fields.append(dict(time=index*.05,shape=list(arr.shape),dtype=str(arr.dtype),raw_float64_sha256=rawhash,
                           file=str(dest.relative_to(out)),stored_file_sha256=digest(dest)))
    allowance=float(max(refrow.get('allowance',refrow.get('reference_allowance'))))
    passed=max(max(errors[1:]),max(actual))+allowance<=.01 and max(changes)<=1e-7
    record={k:v for k,v in run.items() if k not in ['fields','coefficients']}
    record.update(physical_error=errors,actual4096_errors=actual,reference_allowance_max=allowance,
        previous_state_relative_change=changes,previous_state_file_sha256=digest(oldpath),
        actual_fields=fields,host_field_transfer_seconds=transfer,physical_and_equivalence_gate=bool(passed),
        reference_coefficients_sha256=refrow['levels'][-1]['sha256'])
    state=dict(coefficients=np.asarray(C),cells=run['cells'],transforms=np.asarray(tr),method=record)
    statefile=out/f'{label}_{method}_{tag}_states.pkl';record['state_file_sha256']=checkpoint(statefile,state)
    record['state_file']=statefile.name
    canonical=out/f'{label}_{method}_states.pkl'
    if not canonical.exists():checkpoint(canonical,state)
    return record,bool(passed)


def main():
    ap=argparse.ArgumentParser()
    for key in ['out','cache','span','stable-global','admission','dev','development-admission','compare-previous']:
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--mode',choices=['equivalence','paired'],default='equivalence')
    ap.add_argument('--qualified-implementation',help='retained successful equivalence result for this exact implementation')
    ap.add_argument('--normal-reuse',action='store_true')
    ap.add_argument('--block-index',type=int,choices=[0,1],default=0,help='paired blocks use distinct owned allocations')
    ap.add_argument('--labels',default='dev000,dev003,align002')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    gatepath=Path(a.development_admission)
    gate=json.loads(gatepath.read_text());assert not gate['remaining'] and not gate['failed'] and len(gate['records'])==3
    assert all(v['passed'] and len(v['rows'])==88 for v in gate['records'])
    assert {v['name'] for v in gate['records']}=={'b4base','b4time','b4quad'}
    tload=time.perf_counter();root,U,glob,identities=load_frozen(a);loading=time.perf_counter()-tload
    previous=Path(a.compare_previous);assert str(previous).startswith(os.environ['STUDY_NAMESPACE']+'/')
    pr=json.loads((previous/'result.json').read_text())
    for key,value in identities.items():assert pr[key]==value
    labels=a.labels.split(',');assert labels==['dev000','dev003','align002']
    implementation={name:digest(HERE/name) for name in ['common.py','data.py','geometry.py','local_polynomial.py',
        'tensor_metric.py','discrete_weak.py','packed_fields.py','cell_catalogue.py','direct_query.py',
        'fixed_tree_pilot.py','complete_query_diagnostic.py']}
    if a.mode=='paired':
        assert a.qualified_implementation
        qr=json.loads((Path(a.qualified_implementation)/'result.json').read_text())
        assert qr['status']=='complete_query_equivalence_diagnostic' and qr['all_passed']
        assert qr['normal_reuse']==a.normal_reuse
        assert qr['implementation_sha256']==implementation
        for key,value in identities.items():assert qr[key]==value
    refs={r['case']:(p,r) for p,r in cohort(a.dev)}
    for label in labels:
        if 'case_parameters' in pr:
            old_parameters=pr['case_parameters'][label]
            assert pr['reference_hashes'][label]==refs[label][1]['levels'][-1]['sha256']
        else:
            oldrow=next(v for v in pr['cases'] if v['case']==label)
            old_parameters=oldrow['parameters']
            assert oldrow['reference_coefficients_sha256']==refs[label][1]['levels'][-1]['sha256']
        assert np.array_equal(old_parameters,refs[label][1]['parameters'])
    config=dict(L=16384,Q=512,tests=22016,dt=.00125)
    if 'config' in pr:assert pr['config']==config
    else:
        assert (pr['L'],pr['quadrature_intervals'],pr['dt'],pr['time_integrator'])==(config['L'],config['Q'],config['dt'],'cn')
        assert all(c['common_tests']==config['tests'] for c in pr['cases'])
    t0=time.perf_counter();catalogue=CC.build(root,U,**config_except_dt(config))
    tree=FixedQuery(root,U,config['L'],config['Q'],config['tests'],config['dt'],catalogue=catalogue,reuse_normal=a.normal_reuse)
    tree_build=time.perf_counter()-t0
    t0=time.perf_counter();global_query=FixedQuery(glob,U,config['L'],config['Q'],config['tests'],config['dt'],reuse_normal=a.normal_reuse)
    global_build=time.perf_counter()-t0
    queries={'tree_direct24':tree,'global_stable960':global_query};bases={'tree_direct24':root,'global_stable960':glob}
    # One registered training-only smooth input prewarms all runtime shapes.
    warm_parameter=data.parameters('train',4)[3];warm={}
    for method,query in queries.items():
        burn_gpu();t0=time.perf_counter();r=query(lambda xy:data.initial(xy,warm_parameter),outputs=5)
        assert r['complete'],'Training-only prewarm did not complete; preserve failed job'
        warm[method]=dict(seconds=time.perf_counter()-t0,case='train003',parameters=warm_parameter,
                          cached_solutions_reused=False,query_specific_geometry_in_compile_key=False)
        del r
    repeats,contracts=(5,[1,5]) if a.mode=='paired' else (1,[1])
    result=dict(meta=meta,status='complete_query_diagnostic_running',**identities,mode=a.mode,normal_reuse=a.normal_reuse,
        config=config,labels=labels,development_admission_sha256=digest(gatepath),previous_result_sha256=digest(previous/'result.json'),
        reusable_setup=dict(shared_source_verification_and_loading_seconds=loading,
            tree_catalogue_and_constructor_seconds=tree_build,global_constructor_seconds=global_build,
            training_only_prewarm=warm,catalogue_cells=len(catalogue['cells']),
            catalogue_array_bytes=sum(int(v.nbytes) for v in jax.tree_util.tree_leaves(catalogue['arrays']))),
        output_contracts=contracts,repetitions=repeats,blocks=1,block_index=a.block_index,order_seed=2026111901+a.block_index,
        implementation_sha256=implementation,
        case_parameters={label:refs[label][1]['parameters'] for label in labels},
        reference_hashes={label:refs[label][1]['levels'][-1]['sha256'] for label in labels},
        timing_scope='same-GPU complete device-output queries; independent training-only reusable setup; no native NM-ROM confirmation',
        final_test_opened=False,cases=[],invocations=[])
    save(out/'result.json',result);rng=np.random.default_rng(result['order_seed']);passed=[]
    for label in labels:
        folder,row=refs[label];parameter=row['parameters'];initial=lambda xy:data.initial(xy,parameter)
        # Both methods finish this invocation pair before reference scoring.
        for block in [a.block_index]:
            for contract in contracts:
                # Across the two separately allocated blocks, each method goes
                # first exactly five times per case/output contract.
                orientations=[a.block_index]*3+[1-a.block_index]*2 if repeats==5 else [int(rng.integers(2))]
                rng.shuffle(orientations)
                for repeat in range(repeats):
                    order=list(queries)
                    if orientations[repeat]:order.reverse()
                    pair={}
                    for method in order:
                        burn_gpu();tquery=time.perf_counter()
                        invocation=queries[method](initial,outputs=contract,components=a.mode!='paired')
                        jax.block_until_ready((invocation.get('fields',()),invocation.get('coefficients',())))
                        external=time.perf_counter()-tquery
                        invocation['internal_instrumented_timer_seconds']=invocation['total_query_seconds']
                        invocation['total_query_seconds']=external
                        invocation['external_total_timer']=True
                        pair[method]=invocation
                    refpath=folder/row['levels'][-1]['file'];assert digest(refpath)==row['levels'][-1]['sha256']
                    reference=np.load(refpath)['coefficients'][::2]
                    for method in order:
                        oldpath=previous/f'{label}_{method}_states.pkl';old=read_checkpoint(oldpath)
                        tag=f'b{block}_r{repeat}_o{contract}'
                        record,good=score(pair[method],bases[method],U,reference,row,old,oldpath,out,label,method,tag,contract)
                        passed.append(good)
                        result['invocations'].append(dict(case=label,method=method,block=block,repetition=repeat,
                            output_contract=contract,method_order=order,record=record))
                        print('complete_query',label,method,tag,record['total_query_seconds'],good,flush=True)
                    save(out/'result.json',result)
        result['cases'].append(label)
    result['all_passed']=all(passed)
    result['status']='complete_query_equivalence_diagnostic' if a.mode=='equivalence' else 'complete_paired_query_diagnostic'
    result['jax_allocator_memory_stats']=jax.devices()[0].memory_stats()
    save(out/'result.json',result)


def config_except_dt(config):
    return {k:v for k,v in config.items() if k!='dt'}


if __name__=='__main__':main()
