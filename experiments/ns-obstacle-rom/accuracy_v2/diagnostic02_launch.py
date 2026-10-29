"""Release-bound diagnostic of one accepted successor checkpoint, never a refit."""
from pathlib import Path
import argparse,json,os,sys,time,resource
from fe_adapter import sha_file,CAPS
from pilot2d import source_digest,read_preparation_metadata
from capacity_context2d import consumer_caps
from qualification import require_qualification
JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-diagnostic02'

def verify(plan_path,release_path,inventory_path):
    p=json.loads(Path(plan_path).read_text());r=json.loads(Path(release_path).read_text());root=Path(plan_path).parent
    expected={'schema':'ns-obstacle-pod-guided-diagnostic-release-v3','authorized':True,'single_execution':True,
      'plan_sha256':sha_file(plan_path),'runtime_inventory_sha256':sha_file(inventory_path),
      'launcher_sha256':sha_file(__file__),'pilot_source_sha256':source_digest(),'job_directory':JOB,
      'output_directory':JOB+'/results/diagnostic','allocation_seconds':900,'outer_timeout_seconds':885,
      'outer_kill_grace_seconds':10,'finalization_reserve_seconds':60,'numpy_derivative_contraction':'optimized_einsum','fast_evaluator_sha256':sha_file(Path(__file__).parent/'fast_spatial_bank.py'),'training_allowed':False,
      'quadrature_fit_allowed':False,'validation_allowed':False,'automatic_retry':False,'physical_admission':False}
    if any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in expected.items()):raise ValueError('exact separate diagnostic release required')
    if p.get('schema')!='ns-obstacle-pod-guided-diagnostic-plan-v3' or p.get('rank')!=4:raise ValueError('first successor rank4 plan required')
    if set(p['inputs'])!={'prepared','preparation_release','bank','model_receipt','model_review','probes','development'}:raise ValueError('only declared development inputs permitted')
    seen=set()
    for row in json.loads(Path(inventory_path).read_text())['files']:
        d=Path(row['destination']);f=root/d
        if d.is_absolute() or '..' in d.parts or str(d) in seen or f.is_symlink() or f.stat().st_size!=row['bytes'] or sha_file(f)!=row['sha256']:raise ValueError('runtime closure mismatch')
        seen.add(str(d))
    inputs={}
    for key,desc in p['inputs'].items():
        f=Path(desc['path']);inputs[key]=f
        if f.is_symlink():raise ValueError('input symlink forbidden')
        if desc['kind']=='file':
            if f.stat().st_size!=desc['bytes'] or sha_file(f)!=desc['sha256']:raise ValueError('input identity mismatch')
        elif desc['kind']=='directory' and key=='prepared':
            if {str(x.relative_to(f)) for x in f.rglob('*') if x.is_file()}!=set(desc['files']):raise ValueError('prepared file set differs')
            for name,row in desc['files'].items():
                rel=Path(name);q=f/rel
                if rel.is_absolute() or '..' in rel.parts or q.is_symlink() or q.stat().st_size!=row['bytes'] or sha_file(q)!=row['sha256']:raise ValueError('prepared member mismatch')
        else:raise ValueError('unsupported input kind')
    receipt=json.loads(inputs['model_receipt'].read_text());review=json.loads(inputs['model_review'].read_text())
    if receipt.get('schema')!='ns-obstacle-supervised-neural-model-v2' or receipt['bank_sha256']!=sha_file(inputs['bank']):raise ValueError('actual successor model receipt required')
    if review.get('passed') is not True or review.get('model_sha256')!=sha_file(inputs['bank']) or review.get('model_receipt_sha256')!=sha_file(inputs['model_receipt']):raise ValueError('independent actual checkpoint review required')
    meta,consumer=read_preparation_metadata(inputs['prepared'],inputs['preparation_release']);consumer_caps(meta,inputs['preparation_release'],consumer,CAPS)
    if require_qualification(meta['reference_qualification'])!=require_qualification(r['reference_qualification']):raise ValueError('truth scope mismatch')
    if meta['admission_evidence']['reference_admission_sha256']!=r['reference_admission_sha256']:raise ValueError('actual truth admission differs')
    dev=json.loads(inputs['development'].read_text())
    if dev.get('schema')!='ns-obstacle-pilot-reference-population-v1' or dev.get('passed') is not True or dev.get('split')!='development' or [x['Re'] for x in dev['cases']]!=[16.875,23.125]:raise ValueError('only actual two-case DEV population permitted')
    if require_qualification(dev)!=require_qualification(meta['reference_qualification']):raise ValueError('development truth differs')
    for row in dev['cases']:
        rel=Path(row['array_file']);f=inputs['development'].parent/rel
        if rel.is_absolute() or '..' in rel.parts or f.is_symlink() or sha_file(f)!=row['array_sha256']:raise ValueError('development field identity mismatch')
    # Complete semantic rule verification remains in the unchanged loader.
    if sha_file(inputs['development'])!=r['development_manifest_sha256'] or sha_file(inputs['probes'])!=r['probes_sha256']:raise ValueError('development binding differs')
    return p,r,inputs

def run(plan,release,inventory):
    p,r,inputs=verify(plan,release,inventory)
    if str(Path.cwd().resolve())!=JOB or sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('released fresh cluster allocation required')
    with Path('DIAGNOSTIC-CLAIM.json').open('x') as f:json.dump({'job_id':os.environ['SLURM_JOB_ID'],'release_sha256':sha_file(release)},f)
    import jax
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or str(jax.config.jax_default_matmul_precision)!='highest':raise RuntimeError('GPU/f64/highest required')
    start=float(os.environ['ROM_ALLOCATION_START_UNIX_SECONDS'])
    def guard():
        if time.time()>=start+840:raise TimeoutError('diagnostic finalization reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>16*2**30:raise MemoryError('conservative RSS rejection')
    from accuracy_v2.diagnostic import run_verified
    from accuracy_v2.fast_spatial_bank import FastSpatialBank
    return run_verified(inputs,r['output_directory'],sha_file(inputs['bank']),guard,bank_loader=FastSpatialBank.load)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    for name in ('plan','release','inventory'):a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--verify-only',action='store_true');a=a.parse_args()
    if a.verify_only:verify(a.plan,a.release,a.inventory);print('actual source/model/DEV bindings verified without native/JAX import')
    else:run(a.plan,a.release,a.inventory)
