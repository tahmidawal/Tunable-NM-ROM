"""Fixed-tree embedding PDE solve with charged cold start and actual4096 output.

Reference fields enter only the post-output scoring function. No development
projection codes, future fields, allocation oracle or adaptation enters a solve.
This driver alone does not establish time/integration convergence or a paired
competitor comparison. An explicit diagnostic bypass never admits a method.
"""
import argparse
from common import *
from neural import stable as S, fields as F, allocations as A, residual_tree as R
from neural import residual_weak as W, packed_residual as P, weak as BaseWeak
from neural.build_data import resolve
from neural.parent_provenance import load_parent
from training_io import read_checkpoint
from evaluate_residual_tree import owned_job, job_list, equal_tree, make_initializer
import data
from neural.cached_output import OutputCache


def finite_json(value):
    """Failure evidence keeps nonfinite raw arrays in NPZ, nulls in strict JSON."""
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,np.ndarray):return finite_json(value.tolist())
    if isinstance(value,np.generic):return finite_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def reference_metadata(ns, parent, label):
    rows=parent['data_record']['groups']['dev']['sources']
    matches=[r for r in rows if r['case']==label];assert len(matches)==1
    source=matches[0];path=resolve(source['file'],ns)
    record_path=path.parent/'result.json';record=json.loads(record_path.read_text())
    rows=[r for r in record['cases'] if r['case']==label];assert len(rows)==1
    row=rows[0];assert row['provisional_ladder_pass'] and max(row['allowance'])<=.001
    assert any(r['sha256']==source['sha256'] for r in row['levels'])
    return source,path,dict(record_file=str(record_path),record_sha256=digest(record_path),
        reference_file=str(path),reference_sha256=source['sha256'],allowance=row['allowance'],
        qualification_manifest='manifests/reference-qualification.json',
        qualification_manifest_sha256=digest(HERE/'manifests/reference-qualification.json'),
        qualification_scope='Inherited primary reference qualification plus this case spectral ladder')


def verify_fd(ns, model, jobs):
    declared=model.get('expanded_sources',[])
    if not declared:return dict(required=False,passed=True,sources=[])
    proofs=[];checked={}
    for job in jobs:
        path=resolve(ns/job/'output/result.json',ns);record=json.loads(path.read_text())
        assert record['status']=='complete' and record['passed'] and not record['final_test_opened']
        proofs.append(dict(job=job,sha256=digest(path)))
        for row in record['cases']:
            assert row['passed'] and row['admitted_reference']
            checked[(row['source_job'],row['case'])]=row['source_sha256']
    for entry in declared:
        path=resolve(ns/entry['job']/'output/result.json',ns)
        assert digest(path)==entry['record_sha256']
        record=json.loads(path.read_text())
        for row in record['cases']:
            assert checked.get((entry['job'],row['case']))==entry['record_sha256'], 'Missing expanded FD qualification'
    return dict(required=True,passed=True,sources=proofs)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True);ap.add_argument('--model-job',required=True)
    ap.add_argument('--qualification-job',default='');ap.add_argument('--expanded-fd-jobs',default='')
    ap.add_argument('--diagnostic',action='store_true')
    ap.add_argument('--case-label',default='dev003');ap.add_argument('--layout',choices=['uniform','prescribed'],default='prescribed')
    ap.add_argument('--nodes',type=int,choices=[4,20],default=20)
    ap.add_argument('--L',type=int,default=16384);ap.add_argument('--Q',type=int,default=128)
    ap.add_argument('--M',type=int,default=2048);ap.add_argument('--dt',type=float,default=.0005)
    ap.add_argument('--iterations',type=int,default=60);ap.add_argument('--theta',type=float,choices=[.5,1.],default=.5)
    ap.add_argument('--station-tolerance',type=float,default=1e-8);ap.add_argument('--update-tolerance',type=float,default=1e-10)
    ap.add_argument('--root-jacobian',choices=['forward','reverse'],default='forward')
    ap.add_argument('--output-mode',choices=['baseline16','cached64'],default='baseline16')
    ap.add_argument('--no-save-fields',action='store_true');a=ap.parse_args()
    assert a.L>=16384 and a.Q<=a.L and a.L%a.Q==0
    assert a.dt>0 and abs(round(.05/a.dt)*a.dt-.05)<1e-13
    assert a.station_tolerance>0 and a.update_tolerance>0
    owned_job(a.model_job)
    if a.qualification_job:owned_job(a.qualification_job)
    else:assert a.diagnostic,'Qualification job may be omitted only for an explicit diagnostic'
    if a.case_label.startswith('dev'):
        index=int(a.case_label[3:]);assert 0<=index<32 and a.case_label==f'dev{index:03d}'
        initial_parameters=data.parameters('dev',32)[index]
    elif a.case_label.startswith('align'):
        index=int(a.case_label[5:]);assert 0<=index<12 and a.case_label==f'align{index:03d}'
        initial_parameters=data.alignment()[index]
    else:raise ValueError('First fixed-tree driver supports original development/alignment cases only')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight()
    ns=Path(os.environ['STUDY_NAMESPACE']);load_start=time.perf_counter()
    query_start=load_start
    excluded_seconds=dict(gpu_conditioning=0.,reference_scoring=0.,optional_output_host_transfer=0.,archival_io=0.)
    mp=resolve(ns/a.model_job/'output/nonlinear_tree.pkl',ns)
    mr=json.loads((mp.parent/'result.json').read_text());assert mr['status'].startswith('complete_')
    assert digest(mp)==mr['selected_sha256'] and not mr['final_test_opened']
    model=read_checkpoint(mp);params=jax.tree.map(jnp.asarray,model['params'])
    pp,pr,parent,parent_proof=load_parent(mr,ns)
    assert digest(pp)==model['parent_sha256'];assert equal_tree(params['root'],parent['network'])
    root_k,child_k=R.dimensions(params);assert root_k==32 and a.nodes<=mr['nodes']
    assert a.M>2*(root_k+a.nodes*child_k), 'Weak tests must comfortably exceed state dimension'
    qp=None;qr={}
    if a.qualification_job:
        qp=resolve(ns/a.qualification_job/'output/result.json',ns);qr=json.loads(qp.read_text())
        assert qr['model_sha256']==digest(mp) and not qr['final_test_opened']
    gate=bool(qr.get('numerical_representation_gate',False) and qr.get('nodes')==a.nodes
              and not qr.get('diagnostic_subset',True)
              and qr.get('layouts',{}).get(a.layout,{}).get('numerical_representation_gate',False))
    if not a.diagnostic:assert gate,'Representation qualification required; diagnostic bypass must be explicit'
    fd=dict(required=bool(model.get('expanded_sources')),passed=False,sources=[])
    if a.expanded_fd_jobs or not model.get('expanded_sources'):
        fd=verify_fd(ns,model,job_list(a.expanded_fd_jobs))
    elif not a.diagnostic:raise AssertionError('Expanded independent FD proof required')
    source,reference_path,reference=reference_metadata(ns,parent,a.case_label)
    qualification=json.loads((HERE/'manifests/reference-qualification.json').read_text())
    assert qualification['all_required_crosschecks_passed']
    source_load_seconds=time.perf_counter()-load_start
    result=dict(meta=metadata,status='preparing_fixed_tree_solve',parent_job=parent_proof['parent_job'],parent_provenance=parent_proof,root_jacobian=a.root_jacobian,output_mode=a.output_mode,model_job=a.model_job,model_sha256=digest(mp),
        qualification_job=a.qualification_job or None,qualification_sha256=digest(qp) if qp else None,representation_gate_passed=gate,
        expanded_reference_proof=fd,diagnostic=a.diagnostic,method_admitted=False,final_test_opened=False,
        case=a.case_label,layout=a.layout,nodes=a.nodes,root_k=root_k,child_k=child_k,
        L=a.L,Q=a.Q,M=a.M,dt=a.dt,theta=a.theta,iterations=a.iterations,
        station_tolerance=a.station_tolerance,update_tolerance=a.update_tolerance,
        normalization='current-time reference physical L2',reference=reference,outputs=[],steps_completed=0,steps_attempted=0,
        initial_input='Shared initial-field callable; parameter descriptors used only inside that callable',
        no_future_truth_in_online_path=True,adaptation=False,source_load_seconds=source_load_seconds,
        timing_contract='Single invocation, unpaired. Enclosing online wall time includes source loading, cache assembly, cold initialization, explicit compilation, all executed steps, required acceptance transfers/control, and requested device4096 reconstructions. Explicitly timed conditioning, optional output host transfer, reference scoring and archival I/O are excluded and reported separately.',
        time_and_integration_convergence_qualified=False,paired_runtime_comparison=False)
    archive_start=time.perf_counter();save(out/'result.json',result)
    excluded_seconds['archival_io']+=time.perf_counter()-archive_start
    conditioning_start=time.perf_counter();result['gpu_burn']=burn_gpu()
    excluded_seconds['gpu_conditioning']+=time.perf_counter()-conditioning_start
    charges={};compile_times={};step_times=[];codes=[];statuses=[];output_compile={}
    dt=jnp.float64(a.dt);nu=jnp.float64(data.NU)
    station=jnp.float64(a.station_tolerance);update=jnp.float64(a.update_tolerance)
    initial=lambda xy:data.initial(xy,jnp.asarray(initial_parameters))
    start=time.perf_counter()
    base=BaseWeak.prepare(a.L,a.Q,a.M)
    xy=base['xy'];weight=jnp.full(len(xy),1/a.Q**2,dtype=jnp.float64)
    target=initial(xy);norm=jnp.sqrt(jnp.sum(weight*target**2))
    encoder_xy,_=data.grid(16,True);encoder_field=initial(jnp.asarray(encoder_xy))
    allocation_xy,_=data.grid(128,True)
    allocation_target=initial(jnp.asarray(allocation_xy))
    cells=jnp.asarray(A.allocate(a.layout,allocation_xy,np.asarray(allocation_target),mr['seed'],nodes=mr['nodes'],max_depth=3)[:a.nodes])
    A.validate(cells);jax.block_until_ready((target,norm,encoder_field))
    charges['initial_sampling_and_allocation']=time.perf_counter()-start
    cache,charges['weak_cache_assembly']=timer(lambda:W.prepare(params,cells,a.L,a.Q,a.M))
    result['cache']=dict(child_feature_values=sum(int(v.size) for v in [cache['B_stencil']]),
        child_feature_bytes=int(cache['B_stencil'].size*8),feature_assembly_first_use_compilation_included=True)
    cp=dict(params,B=cache['B']);zero=jnp.zeros(root_k+a.nodes*child_k)
    rootfit=F.make_fit(S.apply,60);rootmore=F.make_fit(S.apply,120,newton=True)
    fit=F.make_fit(R.cached,60,jacobian=R.cached_jacobian)
    more=F.make_fit(R.cached,120,jacobian=R.cached_jacobian)
    init=make_initializer(child_k,a.nodes);step=W.make_step(a.L,a.Q,a.iterations,a.theta,root_jacobian=a.root_jacobian)
    def compiled(name,fun,*args):
        begin=time.perf_counter();executable=fun.lower(*args).compile()
        compile_times[name]=time.perf_counter()-begin;return executable
    rf=compiled('root_fit60',rootfit,params['root'],zero[:root_k],xy,target,weight,norm)
    rm=compiled('root_fit120',rootmore,params['root'],zero[:root_k],xy,target,weight,norm)
    ff=compiled('tree_fit60',fit,cp,zero,xy,target,weight,norm)
    fm=compiled('tree_fit120',more,cp,zero,xy,target,weight,norm)
    ii=compiled('residual_initializer',init,params,zero[None,:root_k],cache['B'],target[None],norm[None],xy,weight)
    advance=compiled('weak_step',step,params,zero,cache,dt,nu,station,update)
    start=time.perf_counter()
    near,interp=S.nearest_starts(encoder_field,jnp.asarray(parent['training_encoder']),jnp.asarray(parent['training_codes']))
    attempts=[rf(params['root'],z,xy,target,weight,norm) for z in [near,interp,jnp.zeros(root_k)]]
    rz,rst=min(attempts,key=lambda pair:float(pair[1][0]));root_statuses=[np.asarray(st) for _,st in attempts]
    if not bool(rst[-1]):rz,rst=rm(params['root'],rz,xy,target,weight,norm);root_statuses.append(np.asarray(rst))
    z,initializer_error=ii(params,rz[None],cache['B'],target[None],norm[None],xy,weight)
    z,ist=ff(cp,z[0],xy,target,weight,norm);initial_statuses=[np.asarray(ist)]
    if not bool(ist[-1]):z,ist=fm(cp,z,xy,target,weight,norm);initial_statuses.append(np.asarray(ist))
    jax.block_until_ready(z);charges['cold_embedding_initialization']=time.perf_counter()-start
    result['initialization']=dict(root_attempts=root_statuses,tree_attempts=initial_statuses,
        residual_initializer_error=float(initializer_error[0]),initial_callable_relative_error=float(ist[0]),
        converged=bool(ist[-1]),training_library=parent_proof['parent_job']+' training sampled fields/codes; no development codes')
    result['cells']=np.asarray(cells);codes.append(np.asarray(z));statuses.append(np.asarray(ist)[[0,1,2,4,3,5]])

    def persist():
        # Archive trajectory evidence first. The timing snapshot is taken after
        # this archive and before this call's status JSON write. That final write
        # is outside this snapshot; its measured duration is excluded from every
        # later snapshot. Thus no persist write is omitted or subtracted twice.
        archive_start=time.perf_counter()
        np.savez_compressed(out/'trajectory_codes.npz',codes=np.stack(codes),statuses=np.stack(statuses),
            times=np.arange(len(codes))*a.dt,cells=np.asarray(cells),
            status_columns=np.array(['relative_residual_or_initial_error','stationarity','field_update','iterations_used','accepted_updates','converged']))
        result['trajectory_codes_file']='trajectory_codes.npz'
        result['trajectory_codes_sha256']=digest(out/'trajectory_codes.npz')
        excluded_seconds['archival_io']+=time.perf_counter()-archive_start
        snapshot=time.perf_counter();elapsed=snapshot-query_start
        excluded=sum(excluded_seconds.values())
        assert elapsed>=excluded>=0.,(elapsed,excluded)
        online_wall=elapsed-excluded
        component_sum=source_load_seconds+sum(charges.values())+sum(compile_times.values())+sum(step_times)
        result['timing']=dict(components=charges,explicit_compilation=compile_times,
            step_seconds=step_times,step_seconds_sum=sum(step_times),
            complete_cold_query_seconds=online_wall,online_wall_seconds=online_wall,
            elapsed_to_timing_snapshot_seconds=elapsed,excluded_seconds=dict(excluded_seconds),
            summed_instrumented_components_seconds=component_sum,
            component_sum_note='Incomplete decomposition; excludes uninstrumented control and required acceptance transfers. Complete query is measured by the enclosing wall timer.',
            timing_snapshot_boundary='After current trajectory archival, before current result JSON serialization/write. All earlier status writes are included in cumulative archival exclusion; current write is outside this snapshot.',
            compile_note='Explicit fitting/step/output kernel compilation separated. Spatial feature assembly first-use compilation remains included in assembly/reconstruction.',
            host_transfer_seconds=excluded_seconds['optional_output_host_transfer'],
            reference_scoring_seconds=excluded_seconds['reference_scoring'],
            archival_io_seconds=excluded_seconds['archival_io'])
        archive_start=time.perf_counter()
        try:
            save(out/'result.json',finite_json(result))
        finally:
            excluded_seconds['archival_io']+=time.perf_counter()-archive_start

    optimized_output=None
    def output_field(code):
        nonlocal optimized_output
        if a.output_mode=='cached64':
            assembled_now=optimized_output is None
            if assembled_now:
                begin=time.perf_counter();optimized_output=OutputCache(params,cells,rows=64)
                cache_bytes=optimized_output.build()
                result['output_cache']=dict(build_seconds=time.perf_counter()-begin,
                    logical_cache_bytes=cache_bytes,charged_inside_first_reconstruction=True,
                    build_compilation_seconds=optimized_output.compilation_seconds)
                prior_compile=0.
            else:prior_compile=optimized_output.compilation_seconds
            output=optimized_output.reconstruct(code,reuse=True)
            available_rows=sum(P.feature_rows(packed) for _,packed in optimized_output.tiles)
            rows=available_rows if assembled_now else 0
            result['output_cache']['spatial_feature_rows_built_once']=available_rows
            result['output_cache']['total_compilation_seconds']=optimized_output.compilation_seconds
            return output,rows,optimized_output.compilation_seconds-prior_compile

        # Device output contains all4096² values. Host transfer happens afterwards.
        axis=np.linspace(0.,1.,4096);chunks=[]
        kernel=jax.jit(P.apply);spatial_rows=0;compilation=0.
        for lo in range(0,4096,16):
            points=jnp.asarray(np.stack(np.meshgrid(axis[lo:lo+16],axis,indexing='ij'),-1).reshape(-1,2))
            packed=P.prepare(params['feature'],cells,points);spatial_rows+=P.feature_rows(packed)
            key=tuple(len(v) for v in packed['indices'])
            if key not in output_compile:
                begin=time.perf_counter();output_compile[key]=kernel.lower(params,code,points,packed).compile()
                compilation+=time.perf_counter()-begin
            values=output_compile[key](params,code,points,packed)
            chunks.append(values.reshape(16,4096))
        prediction=jnp.concatenate(chunks,axis=0)
        jax.block_until_ready(prediction)
        return prediction,spatial_rows,compilation

    def score_after_output(prediction,ti):
        # This function is never called before the corresponding model output.
        assert digest(reference_path)==source['sha256']
        with np.load(reference_path) as v:
            assert np.array_equal(v['times'],data.TIMES)
            coefficients=jnp.asarray(v['coefficients'][ti])
        truth=np.asarray(data.synth_grid(coefficients,jnp.linspace(0.,1.,4096)))
        error=prediction-truth
        reference_norm=float(np.linalg.norm(truth))
        relative=float(np.linalg.norm(error)/reference_norm)
        return truth,error,relative,reference_norm

    try:
        if not bool(ist[-1]) or not np.isfinite(np.asarray(ist)).all() or not np.isfinite(np.asarray(z)).all():
            result['status']='failed_initialization';persist();return
        result['status']='running_fixed_tree';persist()
        stride=round(.05/a.dt);nsteps=5*stride
        for it in range(1,nsteps+1):
            (new,st),seconds=timer(lambda:advance(params,z,cache,dt,nu,station,update))
            step_times.append(seconds);codes.append(np.asarray(new));statuses.append(np.asarray(st))
            result['steps_attempted']=it
            if not np.isfinite(np.asarray(new)).all() or not np.isfinite(np.asarray(st)).all() or not bool(st[-1]):
                result.update(status='failed_step',failed_step=it,failed_time=it*a.dt,trajectory_codes_include_failed_attempt=True)
                persist();return
            result['steps_completed']=it
            z=new
            if it%20==0:print('fixed_tree_step',it,float(st[0]),float(st[1]),flush=True)
            if it%stride:continue
            t=it*a.dt;begin=time.perf_counter();prediction,rows,compile_seconds=output_field(z)
            reconstruction=time.perf_counter()-begin
            # Explicit output compilation is separated once, not double-counted.
            compile_times[f'output_{t:.2f}']=compile_seconds
            charges[f'reconstruction_{t:.2f}']=reconstruction-compile_seconds
            begin=time.perf_counter()
            try:
                host=np.asarray(prediction)
            finally:
                transfer=time.perf_counter()-begin
                excluded_seconds['optional_output_host_transfer']+=transfer
            if not np.isfinite(host).all():
                result.update(status='failed_nonfinite_output',failed_time=t);persist();return
            begin=time.perf_counter()
            try:
                truth,error,relative,reference_norm=score_after_output(host,round(t/.025))
            finally:
                score_seconds=time.perf_counter()-begin
                excluded_seconds['reference_scoring']+=score_seconds
            begin=time.perf_counter();file_record=None
            try:
                if not a.no_save_fields:
                    path=out/f'{a.case_label}_{a.layout}_t{t:.2f}_actual4096.npz'
                    np.savez_compressed(path,prediction=host,reference=truth,error=error,time=t,
                        codes=np.asarray(z),cells=np.asarray(cells),role='Fixed-tree PDE output, no future truth in online solve')
                    file_record=dict(file=path.name,sha256=digest(path))
            finally:
                archival=time.perf_counter()-begin
                excluded_seconds['archival_io']+=archival
            allowance=float(reference['allowance'][round(t/.025)])
            result['outputs'].append(dict(time=t,shape=[4096,4096],relative_error=relative,
                reference_allowance=allowance,error_plus_allowance=relative+allowance,
                physical_accuracy_pass=bool(relative+allowance<=.01),reference_l2_discrete=reference_norm,device_reconstruction_seconds=reconstruction,
                explicit_output_compile_seconds=compile_seconds,host_transfer_seconds=transfer,
                reference_scoring_seconds=score_seconds,archival_io_seconds=archival,fields=file_record,
                supported_child_network_rows=rows,dense_child_network_rows=4096**2*a.nodes))
            print('fixed_tree_output',t,relative,allowance,reconstruction,flush=True);persist()
        result.update(status='complete_fixed_tree_diagnostic' if a.diagnostic else 'complete_fixed_tree_solve',
            physical_accuracy_pass=bool(len(result['outputs'])==5 and all(r['physical_accuracy_pass'] for r in result['outputs'])),
            all_steps_converged=True,requested_outputs_complete=len(result['outputs'])==5,
            method_admitted=False,
            admission_note='Time-step/integration refinement and fair paired comparator runs remain required; explicit diagnostic never qualifies the method')
        persist()
    except Exception as exc:
        result.update(status='failed_exception',exception_type=type(exc).__name__,exception=str(exc))
        persist();raise


if __name__=='__main__':main()
