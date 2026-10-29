"""Finish missing unsampled POD controls; preserve the failed original workload."""
from pathlib import Path
import argparse,gc,hashlib,json,os,resource,shutil,sys,time
import numpy as np
import pilot2d as pilot
import stage2d
from fe_adapter import sha_file
from qualification import require_qualification
from frozen_rom2d import read_evidence,input_identity

JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/pod-recovery01'
METHODS={'exact_tensor_weak_LS','mixed_Galerkin','coefficient_PCHIP'}


def checked(descriptor):
    p=Path(descriptor['path'])
    if p.is_symlink() or not p.is_file() or sha_file(p)!=descriptor['sha256']:
        raise ValueError('bound input differs: '+str(p))
    return p


def verify(plan_path,release_path,inventory_path):
    p=json.loads(Path(plan_path).read_text());r=json.loads(Path(release_path).read_text())
    expected={'schema':'ns-obstacle-pod-control-recovery-release-v1','single_execution':True,
        'plan_sha256':sha_file(plan_path),'launcher_sha256':sha_file(__file__),
        'runtime_inventory_sha256':sha_file(inventory_path),'pilot_source_sha256':pilot.source_digest(),
        'job_directory':JOB,'output_directory':JOB+'/results/recovery','allocation_seconds':1200,
        'outer_timeout_seconds':1185,'outer_kill_grace_seconds':10,'finalization_reserve_seconds':60,
        'failure_reserve_debit_seconds':1200,'validation_allowed':False,'training_allowed':False,
        'quadrature_refit_allowed':False,'automatic_retry':False}
    if any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in expected.items()):
        raise ValueError('exact separate failure-reserve recovery release required')
    if (p.get('schema')!='ns-obstacle-pod-control-recovery-plan-v1' or
        p.get('build_prefixes')!=[2,4,8] or p.get('new_query_prefixes')!=[4,8] or
        p.get('development_Re')!=list(pilot.DEV_RE) or p.get('methods')!=sorted(METHODS)):
        raise ValueError('only missing registered POD controls are authorized')
    for row in json.loads(Path(inventory_path).read_text())['files']:
        rel=Path(row['destination']);f=Path(plan_path).parent/rel
        if rel.is_absolute() or '..' in rel.parts or f.is_symlink() or f.stat().st_size!=row['bytes'] or sha_file(f)!=row['sha256']:
            raise ValueError('recovery runtime changed')
    # Old release is used only to revalidate the unchanged inputs, never to execute again.
    old_candidate=checked(p['original_candidate']);old_release=checked(p['original_release'])
    candidate,_,inputs=stage2d.verify(old_candidate,old_release)
    if candidate['stage']!='development' or require_qualification(r['reference_qualification'])!=require_qualification(candidate['reference_qualification']):
        raise ValueError('original FD1 development inputs required')
    metadata,inner=pilot.read_preparation_metadata(inputs['prepared'],inputs['preparation_release'])
    for key,field in [('bank','bank_sha256'),('probes','probes_sha256'),('development','development_manifest_sha256')]:
        if inner.get(field)!=sha_file(inputs[key]):raise ValueError('original consumer binding changed')
    population=json.loads(Path(inputs['development']).read_text())
    if population['reference_admission_sha256']!=metadata['admission_evidence']['reference_admission_sha256']:
        raise ValueError('actual development truth differs')
    for row in population['cases']:
        path=Path(inputs['development']).parent/row['array_file']
        checked({'path':str(path),'sha256':row['array_sha256']})
    prior=Path(p['original_output_directory'])
    stems=[f'{kind}_r{k}_model' for kind in ('bank','POD') for k in (2,4,8)]
    stems+=['development_floors','POD_r2_quadrature_candidates']+[f'POD_r2_development_{re}' for re in pilot.DEV_RE]
    if {x['name'] for x in p['original_outputs']}!={s+ext for s in stems for ext in ('.json','.npz')}:
        raise ValueError('complete original model/floor/r2 evidence set required')
    for row in p['original_outputs']:
        rel=Path(row['name'])
        if rel.name!=str(rel):raise ValueError('local original evidence member required')
        path=checked({'path':str(prior/rel),'sha256':row['sha256']})
        if path.stat().st_size!=row['bytes']:raise ValueError('original evidence size differs')
    floor=read_evidence(prior/'development_floors.json',next(x['sha256'] for x in p['original_outputs'] if x['name']=='development_floors.json'))
    if floor['bank_online_gate'].get('passed') is not False or floor['validation_seen'] is not False:
        raise ValueError('original bank stop and no-validation evidence required')
    candidates=json.loads((prior/'POD_r2_quadrature_candidates.json').read_text())['record']
    if [row['support_cap'] for row in candidates]!=[320,640,1280,2560]:raise ValueError('original rule attempts incomplete')
    for row in candidates:
        checks=row['development_checks']
        accepted=len(checks)==2 and all(d.get('methods',{}).get('fitted_convection_weak_LS',{}).get(
            'quadrature_reached_state_check',{}).get('passed') is True and
            all(d['methods'].get(method,{}).get('invocation',{}).get('work',{}).get('status') in
                ('stationary','root_converged') for method in ('exact_tensor_weak_LS','fitted_convection_weak_LS')) for d in checks)
        if accepted:raise ValueError('a qualified original rule must not be silently discarded by this no-rule recovery')
    return p,r,inputs,floor


def run(plan_path,release_path,inventory_path):
    p,r,inputs,floor=verify(plan_path,release_path,inventory_path)
    if str(Path.cwd().resolve())!=JOB or sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('fresh separately released cluster allocation required')
    start=float(os.environ['ROM_ALLOCATION_START_UNIX_SECONDS']);deadline=start+1140
    def guard():
        if time.time()>=deadline:raise TimeoutError('recovery science deadline; preserve finalization')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>16*2**30:raise MemoryError('existing sixteen-GiB process guard')
    with Path('RECOVERY-CLAIM.json').open('x') as f:json.dump({'release_sha256':sha_file(release_path),'job_id':os.environ['SLURM_JOB_ID']},f)
    out=Path(r['output_directory']);out.mkdir(exist_ok=False)
    status={'status':'started','models':{},'cases':{},'query_receipts':[],
        'original_job_id':6636731,'original_elapsed_seconds':3346,
        'original_full_operator_fingerprints_available':False,'fresh_fingerprints_are_not_claimed_original':True,
        'bank_gate':floor['bank_online_gate'],'validation_seen':False,'quadrature_refitted':False,
        'missing_cases':[f'POD_r{k}_development_{re}' for k in (4,8) for re in pilot.DEV_RE]}
    def persist():
        status['allocation_elapsed_seconds']=time.time()-start
        (out/'RECOVERY.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    persist();retention_errors=[];records={};models={};choices={}
    try:
        import jax
        if jax.default_backend()!='gpu' or not jax.config.x64_enabled or str(jax.config.jax_default_matmul_precision)!='highest':
            raise RuntimeError('GPU/f64/highest required')
        prior=Path(p['original_output_directory'])
        # Byte-identical reuse preserves all original model/floor/r2 successes and failures.
        for row in p['original_outputs']:
            shutil.copyfile(prior/row['name'],out/row['name'])
            if sha_file(out/row['name'])!=row['sha256']:raise ValueError('original evidence copy differs')
        for path in sorted(out.glob('*.json')):
            if path.name=='RECOVERY.json':continue
            wrapped=json.loads(path.read_text())
            records[path.stem]={'record_file':path.name,'record_sha256':sha_file(path),
                'array_file':wrapped['array_file'],'array_sha256':wrapped['array_sha256']}
        arrays,metadata,inner=pilot.load_preparation(inputs['prepared'],inputs['preparation_release'])
        probes=pilot._read_arrays(inputs['probes'],2**30);bank=pilot.FrozenBank.load(inputs['bank'])
        rule=pilot.array_digest({k:arrays['volume'][k] for k in ('points_dimensionless','weights_dimensionless')})
        development=pilot.load_reference_population(inputs['development'],inner['development_manifest_sha256'],'development',rule)
        identity={'manifest_sha256':inner['development_manifest_sha256'],
            'reference_admission_sha256':metadata['admission_evidence']['reference_admission_sha256'],'volume_rule_sha256':rule}
        for prefix in (2,4,8):
            guard();name=f'POD_r{prefix}';status['phase']='build_'+name;persist()
            model=pilot.build_model('POD',prefix,arrays,metadata,probes,bank)
            original=read_evidence(out/(name+'_model.json'),records[name+'_model']['record_sha256'])
            comparable={k:v for k,v in model.source.items() if k!='offline_build_seconds'}
            original_source={k:v for k,v in original['source'].items() if k!='offline_build_seconds'}
            if comparable!=original_source or model.stability!=original['stability'] or not np.array_equal(model.training_coefficients,original['training_coefficients']):
                raise ValueError('available original source/stability/training coefficient bindings differ')
            fingerprint=pilot.model_fingerprint(model);models[name]={'kind':'POD','prefix':prefix,'fingerprint':fingerprint}
            fingerprint_arrays={'training_coefficients':model.training_coefficients}
            fingerprint_arrays.update({'operator_'+k:v for k,v in vars(model.operator).items() if isinstance(v,np.ndarray)})
            fingerprint_arrays.update({'force_'+k:v for k,v in model.force.items() if isinstance(v,np.ndarray)})
            if pilot.array_digest(fingerprint_arrays)!=fingerprint:raise ValueError('fresh operator evidence differs')
            operator_receipt=pilot.save_evidence(out,name+'_recovery_operator',{
                'fresh_fingerprint':fingerprint,'fingerprint_arrays':fingerprint_arrays,
                'original_full_operator_comparison_available':False,'source':model.source,'stability':model.stability})
            choices[name]={'selected_support':None,'selection_population':'development','validation_seen':False,
                'quadrature_status':'measured_rejection' if prefix==2 else 'incomplete_original_timeout_not_refitted'}
            status['models'][name]={'fresh_fingerprint':fingerprint,'available_original_bindings_exact':True,
                'original_operator_fingerprint_comparison':'unavailable, not claimed',
                'offline_build_seconds':model.source['offline_build_seconds'],'operator_evidence':operator_receipt};persist()
            if prefix==2:
                for re in pilot.DEV_RE:status['cases'][f'{name}_development_{re}']={'reused':True,'receipt':records[f'{name}_development_{re}']}
            else:
                original_query=model.query
                def logged(re,contract,operator=None,interpolate=False):
                    method='coefficient_PCHIP' if interpolate else ('exact_tensor_weak_LS' if operator is model.operator else 'mixed_Galerkin')
                    key=f'{name}_{re}_{method}_raw_invocation'
                    try:
                        guard();invocation=original_query(re,contract,operator,interpolate)
                    except Exception as exc:
                        failure={'exception':repr(exc),'state_available':False,'model':name,'Re':re,'method':method}
                        try:receipt=pilot.save_evidence(out,key,failure)
                        except Exception as save_error:retention_errors.append(repr(save_error));raise
                        status['query_receipts'].append({'key':key,'failed':True,'receipt':receipt});persist();raise
                    try:receipt=pilot.save_evidence(out,key,invocation)
                    except Exception as save_error:retention_errors.append(repr(save_error));raise
                    status['query_receipts'].append({'key':key,'failed':False,'receipt':receipt});persist()
                    return invocation
                model.query=logged
                for reference in development:
                    guard();key=f"{name}_development_{reference['Re']}";status['phase']=key;persist()
                    row=pilot.evaluate_case(model,reference,arrays['boundary'],floor['bank_online_gate'],fitted=None)
                    records[key]=pilot.save_evidence(out,key,row)
                    if set(row.get('methods',{}))!=METHODS:raise ValueError('required unsampled controls missing')
                    status['cases'][key]={'reused':False,'receipt':records[key]}
                    status['missing_cases'].remove(key);persist()
                del model.query
                original_query=None
            model=None;gc.collect()
        guard()
        if retention_errors or status['missing_cases'] or len(status['cases'])!=6 or set(models)!={'POD_r2','POD_r4','POD_r8'}:
            raise RuntimeError('incomplete required recovery evidence; freeze forbidden')
        frozen={'schema':'ns-obstacle-frozen-rom-development-v1','source_sha256':pilot.source_digest(),
            'reference_qualification':metadata['reference_qualification'],'development_identity':identity,
            'input_identity':input_identity(metadata,probes,bank),'models':models,'choices':choices,
            'bank_gate':floor['bank_online_gate'],'validation_seen':False,'fitted_operators':{name:None for name in models},
            'recovery_driver_sha256':sha_file(__file__),'original_operator_fingerprints_available':False,
            'original_r2_cases_reused':True,'quadrature_scope_incomplete':True}
        frozen_receipt=pilot.save_evidence(out,'frozen_models',frozen)
        summary={'source_sha256':pilot.source_digest(),'reference_qualification':metadata['reference_qualification'],
            'records':records,'construction_failures':{},'bank_development_gate':floor['bank_online_gate'],
            'frozen_choices':choices,'frozen_models':frozen_receipt,'model_fingerprints':models,'validation_seen':False,
            'timing_comparison_performed':False,'physical_admission':False,'recovery_driver_sha256':sha_file(__file__),
            'original_r2_records_reused':True,'quadrature_scope_incomplete':True,'original_failed_job_preserved':True}
        pilot.save_evidence(out,'development_summary',summary)
        status.update(status='completed_unsampled_controls_pending_independent_review',frozen_models=frozen_receipt)
    except Exception as exc:status.update(status='failed_incomplete',exception=repr(exc));raise
    finally:
        status['retention_errors']=retention_errors
        status['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        persist()
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','release','inventory'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true');a=parser.parse_args()
    if a.verify_only:verify(a.plan,a.release,a.inventory);print('recovery inputs verified; no native/GPU work')
    else:run(a.plan,a.release,a.inventory)
