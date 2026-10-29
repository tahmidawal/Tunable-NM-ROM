"""User-authorized diagnostic only; the failed production bank gate stays false."""
from pathlib import Path
import argparse,json,os,resource,sys,time
import numpy as np
import pilot2d as pilot
import stage2d
from fe_adapter import sha_file
from qualification import require_qualification
from frozen_rom2d import read_evidence

JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-diagnostic01'


def checked(item):
    p=Path(item['path'])
    if p.is_symlink() or not p.is_file() or sha_file(p)!=item['sha256']:raise ValueError('bound input differs: '+str(p))
    return p


def verify(plan_path,release_path,inventory_path):
    p=json.loads(Path(plan_path).read_text());r=json.loads(Path(release_path).read_text())
    expected={'schema':'ns-obstacle-neural-diagnostic-release-v1','single_execution':True,
       'plan_sha256':sha_file(plan_path),'launcher_sha256':sha_file(__file__),
       'runtime_inventory_sha256':sha_file(inventory_path),'pilot_source_sha256':pilot.source_digest(),
       'job_directory':JOB,'output_directory':JOB+'/results/diagnostic','allocation_seconds':900,
       'outer_timeout_seconds':885,'outer_kill_grace_seconds':10,'finalization_reserve_seconds':60,
       'diagnostic_gate_exception_authorized':True,'validation_allowed':False,'training_allowed':False,
       'quadrature_fit_allowed':False,'automatic_retry':False,'physical_admission':False,
       'scope_amendment_sha256':p['scope_amendment']['sha256']}
    if any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in expected.items()):raise ValueError('exact user-authorized diagnostic release required')
    if p.get('schema')!='ns-obstacle-neural-diagnostic-plan-v1' or p.get('prefix')!=8 or p.get('development_Re')!=list(pilot.DEV_RE):raise ValueError('only rank8 and the two development cases')
    amendment=json.loads(checked(p['scope_amendment']).read_text())
    if (amendment.get('schema')!='ns-obstacle-user-authorized-neural-diagnostic-amendment-v1' or
        amendment.get('diagnostic_gate_exception_authorized') is not True or amendment.get('original_representation_gate_passed') is not False or
        amendment.get('Re')!=list(pilot.DEV_RE)):raise ValueError('actual user exception required')
    seen=set()
    for row in json.loads(Path(inventory_path).read_text())['files']:
        rel=Path(row['destination']);f=Path(plan_path).parent/rel
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen or f.is_symlink() or f.stat().st_size!=row['bytes'] or sha_file(f)!=row['sha256']:raise ValueError('diagnostic runtime differs')
        seen.add(str(rel))
    candidate,_,inputs=stage2d.verify(checked(p['original_candidate']),checked(p['original_release']))
    if candidate['stage']!='development' or require_qualification(r['reference_qualification'])!=require_qualification(candidate['reference_qualification']):raise ValueError('original FD1 development inputs required')
    metadata,inner=pilot.read_preparation_metadata(inputs['prepared'],inputs['preparation_release'])
    for key,field in [('bank','bank_sha256'),('probes','probes_sha256'),('development','development_manifest_sha256')]:
        if inner.get(field)!=sha_file(inputs[key]):raise ValueError('original input binding differs')
    if sha_file(inputs['bank'])!=amendment['bank_sha256']:raise ValueError('user-authorized bank differs')
    population=json.loads(Path(inputs['development']).read_text())
    if population['reference_admission_sha256']!=metadata['admission_evidence']['reference_admission_sha256']:raise ValueError('actual development truth differs')
    for row in population['cases']:checked({'path':str(Path(inputs['development']).parent/row['array_file']),'sha256':row['array_sha256']})
    prior=Path(p['original_output_directory'])
    if {x['name'] for x in p['original_outputs']}!={s+e for s in ('bank_r8_model','development_floors') for e in ('.json','.npz')}:raise ValueError('original model and failed-floor evidence required')
    for row in p['original_outputs']:
        if Path(row['name']).name!=row['name']:raise ValueError('local evidence name required')
        f=checked({'path':str(prior/row['name']),'sha256':row['sha256']})
        if f.stat().st_size!=row['bytes']:raise ValueError('original evidence size differs')
    floor=read_evidence(prior/'development_floors.json',next(x['sha256'] for x in p['original_outputs'] if x['name']=='development_floors.json'))
    if floor['bank_online_gate'].get('passed') is not False or floor['validation_seen'] is not False:raise ValueError('original failed gate must remain false')
    return p,r,inputs,floor


def diagnose(model,re,reference,boundary,out,guard):
    key='bank_r8_'+str(re);row={'Re':re,'eligible_accuracy':False,'physical_admission':False,'diagnostic_only':True}
    # Only Re and the existing exact operator reach query; its existing initializer is zero.
    try:
        guard();invocation=model.query(re,'full_velocity_pressure_fields',model.operator,False)
    except Exception as exc:
        failure=getattr(exc,'invocation',None)
        if failure is None:failure={'exception':repr(exc),'state_available':False}
        row['raw_invocation']=pilot.save_evidence(out,key+'_raw_invocation',failure);row['failure']=repr(exc)
        return row
    row['raw_invocation']=pilot.save_evidence(out,key+'_raw_invocation',invocation)
    try:
        guard();state=invocation['evidence']['state']
        direct=np.r_[model.rule.direct(state,1./re)];tensor=model.operator.residual(state,1./re,whiten=False)
        terms=model.operator.terms(state,1./re);denominator=max(sum(float(np.linalg.norm(t)) for t in terms.values()),1e-12)
        gradient_fields=model.reconstruct(state,re,gradients=True);boundary_fields=model.reconstruct(state,re,'boundary')
        row['diagnostic_arrays']=pilot.save_evidence(out,key+'_diagnostic_arrays',{'full':gradient_fields,'boundary':boundary_fields,'direct':direct,'tensor':tensor,'terms':terms,'force_coefficient':model.force_output(state,re)})
        scores=pilot.score_state(model,state,reference,boundary,model.rule.weights)
        derivative=pilot.jacobian_check(model.operator,state,1./re,np.ones(len(state)))
        gap=float(np.linalg.norm(direct-tensor)/denominator)
        row.update(scores=scores,solver_status=invocation['work']['status'],full_direct_tensor_relative_gap=gap,
            full_direct_tensor_denominator=denominator,jacobian_check=derivative,
            diagnostic_numerical_targets_passed=bool(invocation['work']['status'] in ('stationary','root_converged') and scores['accuracy_targets_passed'] and derivative['passed'] and gap<=1e-10))
    except Exception as exc:row['diagnostic_failure']=repr(exc)
    return row


def run(plan_path,release_path,inventory_path):
    p,r,inputs,floor=verify(plan_path,release_path,inventory_path)
    if str(Path.cwd().resolve())!=JOB or sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('fresh released cluster allocation required')
    start=float(os.environ['ROM_ALLOCATION_START_UNIX_SECONDS'])
    def guard():
        if time.time()>=start+840:raise TimeoutError('diagnostic deadline; finalization reserved')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>16*2**30:raise MemoryError('existing process guard')
    with Path('DIAGNOSTIC-CLAIM.json').open('x') as f:json.dump({'release_sha256':sha_file(release_path),'job_id':os.environ['SLURM_JOB_ID']},f)
    out=Path(r['output_directory']);out.mkdir(exist_ok=False)
    result={'schema':'ns-obstacle-neural-diagnostic-result-v1','status':'started','bank_gate':floor['bank_online_gate'],
       'diagnostic_gate_exception_authorized':True,'physical_admission':False,'eligible_accuracy':False,
       'validation_seen':False,'training_or_quadrature_fit_performed':False,'reference_fit_initialization_used':False,
       'matched_accuracy_or_speedup_claim':False,'original_operator_bitwise_equality_claim':False,'cases':[]}
    serial=[0]
    def persist():
        result['allocation_elapsed_seconds']=time.time()-start
        serial[0]+=1
        receipt=pilot.save_evidence(out,'diagnostic_status_'+str(serial[0]).zfill(3),result)
        (out/'DIAGNOSTIC.json').write_text(json.dumps({'schema':result['schema'],'status':result['status'],
            'latest_evidence':receipt,'bank_gate_passed':False,'eligible_accuracy':False,'physical_admission':False,
            'completed_case_records':len(result['cases'])},indent=2)+'\n')
    persist()
    try:
        import jax
        if jax.default_backend()!='gpu' or not jax.config.x64_enabled or str(jax.config.jax_default_matmul_precision)!='highest':raise RuntimeError('GPU/f64/highest required')
        arrays,metadata,_=pilot.load_preparation(inputs['prepared'],inputs['preparation_release']);guard()
        probes=pilot._read_arrays(inputs['probes'],2**30);bank=pilot.FrozenBank.load(inputs['bank'])
        model=pilot.build_model('bank',8,arrays,metadata,probes,bank)
        evidence={'source':model.source,'stability':model.stability,'fresh_fingerprint':pilot.model_fingerprint(model),
             'training_coefficients':model.training_coefficients,'operator':vars(model.operator),'force':model.force}
        result['model']=pilot.save_evidence(out,'bank_r8_diagnostic_model',evidence);persist()
        if not model.stability['passed']:raise RuntimeError('unchanged inf-sup gate rejected; representation exception does not waive it')
        rule=pilot.array_digest({k:arrays['volume'][k] for k in ('points_dimensionless','weights_dimensionless')})
        population=pilot.load_reference_population(inputs['development'],sha_file(inputs['development']),'development',rule)
        for reference in population:
            guard();row=diagnose(model,reference['Re'],reference,arrays['boundary'],out,guard)
            result['cases'].append(row);persist()
        result['status']='completed_diagnostic_pending_independent_review';persist()
    except BaseException as exc:
        result.update(status='failed',exception=repr(exc));persist();raise
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('plan','release','inventory'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.verify_only:
        verify(args.plan,args.release,args.inventory);print(json.dumps({'verified':True,'native_or_GPU_imported':False}))
    else:run(args.plan,args.release,args.inventory)
