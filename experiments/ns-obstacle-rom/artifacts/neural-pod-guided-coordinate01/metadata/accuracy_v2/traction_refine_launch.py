"""Supervised trace plus boundary-traction warm refinement; no Adam, validation or exact continuation."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
from fe_adapter import sha_file,CAPS
from pilot2d import read_preparation_metadata,source_digest as pilot_digest
from capacity_context2d import consumer_caps
from qualification import require_qualification

JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-traction01'

def verify(plan_path,release_path,inventory_path):
    plan=json.loads(Path(plan_path).read_text());r=json.loads(Path(release_path).read_text());root=Path(plan_path).parent
    expected={'schema':'ns-obstacle-supervised-traction-refinement-release-v1','authorized':True,'single_execution':True,'job_directory':JOB,'output_directory':JOB+'/results/fit','plan_sha256':sha_file(plan_path),'runtime_inventory_sha256':sha_file(inventory_path),'launcher_sha256':sha_file(__file__),'pilot_source_sha256':pilot_digest(),'allocation_seconds':3600,'outer_timeout_seconds':3585,'outer_kill_grace_seconds':10,'resume_allowed':False,'validation_allowed':False,'new_reference_solves_allowed':False,'automatic_retry':False,'raw_plus_archive_reservation_bytes':6*2**30,'supervised_divergence_weight':1.0,'traction_weight':1.0,'parent_base_objective_replay_required':True,'new_adam_steps':0,'warm_restart':True,'exact_optimizer_continuation_claim':False,'estimated_peak_host_bytes':16*2**30,'allocation_memory_bytes':64*2**30,'resource_check_sha256':sha_file(root/'RESOURCE-CHECK.json')}
    if any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in expected.items()):raise ValueError('exact separate successor execution release required')
    if plan['schema']!='ns-obstacle-supervised-traction-refinement-plan-v1' or set(plan['inputs'])!={'prepared','preparation_release','parent_bank','parent_receipt','parent_review','training_coefficients_source'}:raise ValueError('training-only input schema')
    if sha_file(root/'CONFIG.json')!=plan['config_sha256'] or sha_file(root/'SCOPE-AMENDMENT.json')!=plan['scope_amendment_sha256']:raise ValueError('recipe or user scope changed')
    scope=json.loads((root/'SCOPE-AMENDMENT.json').read_text())
    if scope.get('schema')!='ns-obstacle-accuracy-first-successor-amendment-v1' or scope.get('successor_training_and_architecture_changes_authorized') is not True:raise ValueError('genuine successor scope required')
    config=json.loads((root/'CONFIG.json').read_text())
    if config!=plan['config'] or config['split']!='training' or config['rank']!=4:raise ValueError('frozen first recipe differs')
    seen=set()
    for row in json.loads(Path(inventory_path).read_text())['files']:
        rel=Path(row['destination']);p=root/rel
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen or p.is_symlink() or p.stat().st_size!=row['bytes'] or sha_file(p)!=row['sha256']:raise ValueError('runtime closure changed')
        seen.add(str(rel))
    inputs={}
    for key,desc in plan['inputs'].items():
        path=Path(desc['path']);inputs[key]=path
        if path.is_symlink():raise ValueError('immutable input symlink forbidden')
        if desc['kind']=='file':
            if path.stat().st_size!=desc['bytes'] or sha_file(path)!=desc['sha256']:raise ValueError('input file changed')
        elif desc['kind']=='directory':
            actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
            if actual!=set(desc['files']):raise ValueError('input directory file set differs')
            for name,row in desc['files'].items():
                rel=Path(name);p=path/rel
                if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or p.stat().st_size!=row['bytes'] or sha_file(p)!=row['sha256']:raise ValueError('immutable prepared input changed')
        else:raise ValueError('unsupported input kind')
    meta,consumer=read_preparation_metadata(inputs['prepared'],inputs['preparation_release']);consumer_caps(meta,inputs['preparation_release'],consumer,CAPS)
    if require_qualification(meta['reference_qualification'])!=require_qualification(r['reference_qualification']) or meta['admission_evidence']['reference_admission_sha256']!=r['reference_admission_sha256']:raise ValueError('actual truth binding differs')
    if meta['split']!='training' or meta['Re']!=[15+1.25*j for j in range(9)]:raise ValueError('actual nine training cases required')
    if meta['pod']['velocity']['small_R_singular_values_padded'][:4]!=plan['teacher_singular_values']:raise ValueError('actual POD teacher ordering/weights differ')
    receipt=json.loads(inputs['parent_receipt'].read_text());review=json.loads(inputs['parent_review'].read_text())
    if receipt.get('bank_sha256')!=sha_file(inputs['parent_bank']) or review.get('passed') is not True or review.get('model_sha256')!=sha_file(inputs['parent_bank']) or review.get('model_receipt_sha256')!=sha_file(inputs['parent_receipt']):raise ValueError('reviewed actual parent model required')
    from accuracy_v2.spatial_bank import SpatialBank
    from accuracy_v2.train import digest
    import numpy as np
    bank=SpatialBank.load(inputs['parent_bank']);teacher=bank.metadata['teacher_identity']
    if bank.metadata.get('objective_variant')!='supervised_trace_error_lambda_1':raise ValueError('reviewed trace parent required')
    if inputs['training_coefficients_source']!=inputs['prepared']/'training.npz' or sha_file(inputs['training_coefficients_source'])!=meta['coefficient_files']['training.npz']:raise ValueError('actual original TRAIN coefficient source required')
    theta=np.concatenate([a.ravel() for layer in bank.layers for a in layer])
    if bank.rank!=4 or not np.array_equal(bank.transform,np.eye(4)) or bank.metadata['config']!=config or digest(theta)!=receipt['selected_theta_sha256']:raise ValueError('actual unrotated parent/config/theta differs')
    if teacher['split']!='training' or teacher['Re']!=meta['Re'] or teacher['prepared_manifest_sha256']!=plan['prepared_manifest_sha256'] or teacher['training_array_sha256']!=meta['training_array_sha256']:raise ValueError('parent training provenance differs')
    if type(review.get('allocated_seconds')) is not int or review['allocated_seconds']<0 or review.get('adam_steps_recorded')!=bank.metadata['actual_applied_adam_steps'] or review.get('accepted_lbfgs_steps')!=bank.metadata['accepted_lbfgs_steps'] or review.get('selected_theta_sha256')!=receipt['selected_theta_sha256']:raise ValueError('actual parent cost/count/selection review missing')
    expected_protocol={'max_iterations':10000,'checkpoint_every':250,'history_size':10,'line_search_steps':20,'gradient_tolerance':1e-10,'warm_restart':True,'new_adam_steps':0,'supervised_divergence_weight':1.0,'traction_weight':1.0}
    if plan['refinement_protocol']!=expected_protocol or r['refinement_protocol']!=expected_protocol:raise ValueError('frozen compact refinement protocol required')
    return plan,r,inputs,config

def run(plan_path,release_path,inventory_path):
    plan,r,inputs,config=verify(plan_path,release_path,inventory_path)
    if str(Path.cwd().resolve())!=JOB or sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('fresh released GPU allocation required')
    with Path('TRAINING-CLAIM.json').open('x') as f:json.dump({'release_sha256':sha_file(release_path),'job_id':os.environ['SLURM_JOB_ID']},f)
    status={'status':'started','training_only':True,'validation_seen':False,'physical_admission':False,'automatic_resume':False};start=float(os.environ['ROM_ALLOCATION_START_UNIX_SECONDS'])
    try:
        from accuracy_v2.teacher import load
        from accuracy_v2.traction_refine import refine
        x,w,v,g,metadata,identity=load(inputs['prepared'],inputs['preparation_release'])
        from accuracy_v2.boundary_teacher import load as load_boundary
        from accuracy_v2.training_coefficients import certify
        boundary,boundary_record=load_boundary(inputs['prepared'],metadata,identity,Path('results/boundary-teacher'))
        coefficient_record,_=certify(inputs['training_coefficients_source'],plan['inputs']['training_coefficients_source']['sha256'],x,w,v,identity,Path('results/training-coefficients'))
        if time.time()>=start+3480:raise TimeoutError('preparation consumed operational training allocation')
        bank=refine(x,w,v,g,identity,boundary,boundary_record,coefficient_record,inputs['parent_bank'],inputs['parent_receipt'],inputs['parent_review'],plan['refinement_protocol'],r['output_directory'],start+3570)
        receipt=Path(r['output_directory'])/'MODEL-RECEIPT.json'
        if not receipt.is_file():raise RuntimeError('missing model export receipt')
        status.update(status='training_checkpoint_exported_pending_independent_review',model_receipt_sha256=sha_file(receipt),optimizer_status=bank.metadata['status'])
    except BaseException as exc:status.update(status='failed',exception=repr(exc));raise
    finally:
        status['allocation_elapsed_seconds']=time.time()-start
        Path('results/TRAINING-RESULT.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','release','inventory'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify(a.plan,a.release,a.inventory);print('source/recipe/actual training inputs verified; no JAX or native imports')
    else:run(a.plan,a.release,a.inventory)
