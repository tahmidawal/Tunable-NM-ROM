"""Released FD1 handoff builders. No reference admission or model fitting is created.

Training consumes separately reviewed lossless materializations. Development and
validation reconstruct only their own admitted primary cases, in a fresh GPU process.
Validation source arrays stay unopened until the exact frozen-method review/release.
"""
from pathlib import Path
import argparse,hashlib,json,os,resource,shutil,sys,time
import numpy as np
import fe_adapter as fe
import materialize_reference2d as materializer
from qualification import require_qualification
from pilot2d import DEV_RE,VALIDATION_RE,array_digest

ROOT=Path(__file__).parent
POPULATIONS={'training':fe.REYNOLDS,'development':list(DEV_RE),'validation':list(VALIDATION_RE)}


def source_digest():
    return hashlib.sha256(json.dumps({'entry':fe.sha_file(__file__),'materializer':materializer.source_digest(),
        'adapter':fe.adapter_digest(),'validation_guard':fe.sha_file(ROOT/'validation2d.py')},sort_keys=True).encode()).hexdigest()


def read_bound(item):
    path=Path(item['path']).resolve()
    if fe.sha_file(path)!=item['sha256']:raise ValueError('declared input changed: '+str(path))
    return path,json.loads(path.read_text())


def verify(plan_path,release_path):
    plan=json.loads(Path(plan_path).read_text());release=json.loads(Path(release_path).read_text());split=plan['split']
    if (split not in POPULATIONS or plan.get('schema')!='ns-obstacle-fd1-handoff-plan-v1' or
        release.get('schema')!='ns-obstacle-fd1-handoff-release-v1' or release.get('source_sha256')!=source_digest() or
        release.get('plan_sha256')!=fe.sha_file(plan_path) or release.get('split')!=split or release.get('single_execution') is not True):
        raise ValueError('exact split/source/plan release required')
    truth_path,truth=read_bound(plan['reference_admission']);q=require_qualification(truth)
    if release.get('reference_admission_sha256')!=fe.sha_file(truth_path) or require_qualification(release.get('reference_qualification'))!=q:
        raise ValueError('actual released FD1 admission required')
    expected={re:('train' if key=='training' else key) for key,values in POPULATIONS.items() for re in values}
    rows=truth.get('admitted_references',[])
    if len(rows)!=15 or {row['Re']:row['split'] for row in rows}!=expected:raise ValueError('complete genuine fifteen-case admission required')
    if [row['Re'] for row in plan['cases']]!=POPULATIONS[split]:raise ValueError('exact split population/order required')
    if split=='validation':
        from validation2d import verify_review
        frozen_path,_=read_bound(plan['frozen_models']);review_path,_=read_bound(plan['rom_review'])
        review=verify_review({'frozen_models':str(frozen_path),'rom_review':str(review_path)})
        if (review['reference_admission_sha256']!=fe.sha_file(truth_path) or require_qualification(review['reference_qualification'])!=q or
            release.get('all_development_decisions_frozen') is not True or release.get('future_validation_driven_selection_changes_forbidden') is not True):
            raise ValueError('frozen methods and explicit validation materialization release required')
    return plan,release,truth,q


def adopted_case(truth,re,folder):
    row=next(row for row in truth['admitted_references'] if row['Re']==re);folder=Path(folder).resolve()
    pairs={'export_manifest_sha256':'EXPORT-MANIFEST.json','config_sha256':'config.json','mesh_sha256':'mesh.vol.bin','solution_sha256':'solution.sol'}
    if any(fe.sha_file(folder/name)!=row[key] for key,name in pairs.items()):raise ValueError('actual adopted runtime differs')
    if json.loads((folder/'fe_layout.json').read_text())['sha256']!=row['layout_sha256']:raise ValueError('adopted internal layout differs')
    return row


def training(plan,truth,q,out):
    lift_path,_=read_bound(plan['common_lift_admission'])
    lift=fe.read_shared_lift_admission(lift_path,fe.sha_file(lift_path))
    truth_sha=plan['reference_admission']['sha256']
    if lift['reference_qualification']!=q or lift['reference_admission_sha256']!=truth_sha:raise ValueError('common lift truth differs')
    adapter_review=Path(plan['adapter_review']['path'])
    if fe.sha_file(adapter_review)!=plan['adapter_review']['sha256']:raise ValueError('adapter review changed')
    rows=[]
    for item in plan['cases']:
        folder=Path(item['directory']).resolve();receipt=folder/'MATERIALIZATION.json';meta=json.loads(receipt.read_text())
        _,review=read_bound(item['materialization_review']);adopted=next(row for row in truth['admitted_references'] if row['Re']==item['Re'])
        if (review.get('schema')!='ns-obstacle-fd1-materialization-review-v1' or review.get('passed') is not True or
            review.get('materialization_sha256')!=fe.sha_file(receipt) or review.get('export_manifest_sha256')!=fe.sha_file(folder/'EXPORT-MANIFEST.json') or
            review.get('reference_admission_sha256')!=truth_sha or
            meta.get('source_sha256')!=materializer.source_digest() or meta.get('status')!='exact_content_replay_passed_pending_independent_admission' or meta.get('Re')!=item['Re'] or
            meta.get('reference_admission_sha256')!=truth_sha or require_qualification(meta['reference_qualification'])!=q or
            meta.get('original_export_sha256')!=adopted['export_manifest_sha256'] or meta.get('original_config_sha256')!=adopted['config_sha256']):
            raise ValueError('actual independently reviewed adopted materialization required')
        export=json.loads((folder/'EXPORT-MANIFEST.json').read_text())
        if any(fe.sha_file(folder/name)!=export['files'].get(name) for name in fe.FILES if name!='EXPORT-MANIFEST.json'):
            raise ValueError('reviewed derived export member changed')
        if fe.sha_file(folder/'full_rule.npz')!=meta['new_full_rule_archive_sha256'] or fe.sha_file(folder/'config.json')!=meta['derived_config_sha256']:
            raise ValueError('derived fields/config changed')
        for key,name in [('mesh_sha256','mesh.vol.bin'),('solution_sha256','solution.sol')]:
            if fe.sha_file(folder/name)!=adopted[key]:raise ValueError('immutable adopted state changed')
        rows.append({'Re':item['Re'],'split':'training','directory':os.path.relpath(folder,out),'files':{name:fe.sha_file(folder/name) for name in fe.FILES},
            'materialization_sha256':fe.sha_file(receipt),'materialization_review_sha256':item['materialization_review']['sha256']})
    contract=json.loads((ROOT/'MODEL-CONTRACT.json').read_text())
    manifest=dict(q,schema=fe.SCHEMA,split='training',adapter_source_sha256=fe.adapter_digest(),adapter_review_sha256=plan['adapter_review']['sha256'],
        reference_admission_sha256=truth_sha,campaign_sha256=contract['central_campaign_sha256'],mesh_filename=fe.MESH_FILENAME,
        mesh_format=fe.MESH_FORMAT,mesh_reload=fe.RELOAD_POLICY,cases=rows,
        common_stokes_lift={'admission_path':os.path.relpath(lift_path,out),'admission_sha256':fe.sha_file(lift_path)})
    target=out/'training.json';target.write_text(json.dumps(manifest,indent=2)+'\n');fe.read_training_manifest(target)
    return target


def population(plan,truth,q,out,release):
    # Exact validation freeze is checked by verify() before this function or native import.
    if any(name[:-3] in sys.modules for name in json.loads((materializer.NATIVE/'SOURCE.json').read_text())['files']):
        raise RuntimeError('fresh native reconstruction interpreter required')
    if (sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID') or
        os.environ.get('JAX_ENABLE_X64')!='true' or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest' or
        Path(release['job_directory']).resolve()!=Path.cwd().resolve()):raise RuntimeError('fresh released cluster GPU process required')
    import jax
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or jax.config.jax_default_matmul_precision!='highest':raise RuntimeError('GPU/f64/highest required')
    print('jax_backend=gpu',flush=True)
    sys.path.insert(0,str(materializer.NATIVE))
    from reference_primary import load_primary_case,reconstruct_primary
    from immutable_rule import array_digest as physical_digest
    import ngsolve as ng
    cases=[];rule=None
    for item in plan['cases']:
        folder=Path(item['directory']).resolve();adopted=adopted_case(truth,item['Re'],folder)
        if adopted['storage_mode']!='coefficient_primary':raise ValueError('held-out adopted primary storage required')
        cfg,primary,_,_=load_primary_case(folder)
        if cfg['U']!=.2 or cfg['D']!=.1 or abs(cfg['nu']-.02/item['Re'])>1e-15:raise ValueError('fixed scales/parameter differ')
        descriptors=primary['derived_full_rule_arrays'];expanded=sum(int(np.prod(d['shape']))*np.dtype(d['dtype']).itemsize for d in descriptors.values())
        reserve=release['other_active_disk_reservation_bytes']
        if type(reserve) is not int or reserve<0:raise ValueError('explicit concurrent disk reservation required')
        if expanded>2**30 or shutil.disk_usage(out).free<10*2**30+reserve+4*expanded+64*2**20:raise MemoryError('raw/reconstructed/durable output disk reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+4*expanded+2*2**30>16*2**30:raise MemoryError('reconstruction process cap')
        case=out/str(item['Re']);case.mkdir();fields,replay=reconstruct_primary(folder,ng);volume=fields['volume']
        raw={key:volume[key] for key in descriptors};np.savez_compressed(case/'raw-full-rule.npz',**raw)
        actual={key:{'dtype':a.dtype.str,'shape':list(a.shape),'content_sha256':physical_digest(a)} for key,a in raw.items()}
        (case/'REPLAY.json').write_text(json.dumps({'reconstruction':replay,'actual':actual,'expected':descriptors,
            'original_export_sha256':adopted['export_manifest_sha256'],'raw_archive_sha256':fe.sha_file(case/'raw-full-rule.npz')},indent=2)+'\n')
        if actual!=descriptors:raise ValueError('actual primary replay differs; raw counterexample retained')
        current=array_digest({'points_dimensionless':volume['points']/.1,'weights_dimensionless':volume['weights']/.1**2})
        if current!=plan['volume_rule_sha256'] or (rule is not None and current!=rule):raise ValueError('exact prepared common rule differs')
        rule=current
        scaled={'velocity':volume['velocity']/.2,'pressure':volume['pressure']/.2**2,'velocity_gradient':volume['velocity_gradient']*.1/.2}
        if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in scaled.values()):raise ValueError('finite original f64 fields required')
        arrays=case/'fields.npz';np.savez_compressed(arrays,**scaled)
        result=json.loads((folder/'result.json').read_text());force=result['metrics']['forces']['physical_symmetric_order16']['coefficient']
        if np.asarray(force).shape!=(2,) or not np.isfinite(force).all():raise ValueError('actual finite symmetric force vector required')
        cases.append({'Re':item['Re'],'split':plan['split'],'reference_admitted':True,'force_definition':'physical_symmetric',
            'force_coefficient':force,'array_file':str(arrays.relative_to(out)),'array_sha256':fe.sha_file(arrays),
            'original_export_sha256':adopted['export_manifest_sha256'],'replay_sha256':fe.sha_file(case/'REPLAY.json')})
        del fields,volume,raw,scaled
    target=out/(plan['split']+'.json')
    target.write_text(json.dumps(dict(q,schema='ns-obstacle-pilot-reference-population-v1',split=plan['split'],
        reference_admission_sha256=plan['reference_admission']['sha256'],volume_rule_sha256=rule,
        velocity_gradient_units='d(u/U)/d(x/D)',cases=cases),indent=2)+'\n')
    return target


def run(plan_path,release_path,out):
    started=time.perf_counter();plan,release,truth,q=verify(plan_path,release_path);out=Path(out).resolve()
    if str(out)!=release.get('output_directory'):raise ValueError('released fresh output path required')
    out.mkdir(parents=True,exist_ok=False)
    record={'physical_admission_created':False,'source_sha256':source_digest(),'plan_sha256':fe.sha_file(plan_path),
        'release_sha256':fe.sha_file(release_path),'reference_admission_sha256':plan['reference_admission']['sha256'],'split':plan['split']}
    try:
        target=training(plan,truth,q,out) if plan['split']=='training' else population(plan,truth,q,out,release)
        record.update(status='completed_pending_independent_handoff_review',manifest_sha256=fe.sha_file(target))
    except Exception as exc:record.update(status='failed',exception=repr(exc));raise
    finally:
        record.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            retained_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
        (out/'HANDOFF.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.plan,a.release,a.out),indent=2))
