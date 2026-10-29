"""First accuracy-first training allocation. Fresh only; validation and resume forbidden."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
from fe_adapter import sha_file,CAPS
from pilot2d import read_preparation_metadata,source_digest as pilot_digest
from capacity_context2d import consumer_caps
from qualification import require_qualification

JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided01'

def verify(plan_path,release_path,inventory_path):
    plan=json.loads(Path(plan_path).read_text());r=json.loads(Path(release_path).read_text());root=Path(plan_path).parent
    expected={'schema':'ns-obstacle-pod-guided-training-release-v2','authorized':True,'single_execution':True,'job_directory':JOB,'output_directory':JOB+'/results/fit','plan_sha256':sha_file(plan_path),'runtime_inventory_sha256':sha_file(inventory_path),'launcher_sha256':sha_file(__file__),'pilot_source_sha256':pilot_digest(),'allocation_seconds':3600,'outer_timeout_seconds':3585,'outer_kill_grace_seconds':10,'resume_allowed':False,'validation_allowed':False,'new_reference_solves_allowed':False,'automatic_retry':False,'raw_plus_archive_reservation_bytes':2**30,'estimated_peak_host_bytes':16*2**30,'allocation_memory_bytes':64*2**30,'resource_check_sha256':sha_file(root/'RESOURCE-CHECK.json')}
    if any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in expected.items()):raise ValueError('exact separate successor execution release required')
    if plan['schema']!='ns-obstacle-pod-guided-training-plan-v2' or set(plan['inputs'])!={'prepared','preparation_release'}:raise ValueError('training-only input schema')
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
    return plan,r,inputs,config

def run(plan_path,release_path,inventory_path):
    plan,r,inputs,config=verify(plan_path,release_path,inventory_path)
    if str(Path.cwd().resolve())!=JOB or sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('fresh released GPU allocation required')
    with Path('TRAINING-CLAIM.json').open('x') as f:json.dump({'release_sha256':sha_file(release_path),'job_id':os.environ['SLURM_JOB_ID']},f)
    status={'status':'started','training_only':True,'validation_seen':False,'physical_admission':False,'automatic_resume':False};start=float(os.environ['ROM_ALLOCATION_START_UNIX_SECONDS'])
    try:
        from accuracy_v2.teacher import load
        from accuracy_v2.train import train
        x,w,v,g,metadata,identity=load(inputs['prepared'],inputs['preparation_release'])
        if time.time()>=start+3480:raise TimeoutError('preparation consumed operational training allocation')
        bank=train(x,w,v,g,plan['teacher_singular_values'],metadata['reference_qualification'],config,r['output_directory'],start+3540,teacher_identity=identity)
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
