"""Released exact FD1 training-field reconstruction; never creates admission.

Run in a fresh cluster interpreter. Original primary/runtime files remain immutable;
new ZIP/config/export containers have new hashes and explicit derivation provenance.
"""
from pathlib import Path
import argparse,hashlib,json,os,resource,shutil,sys,time
import numpy as np
from qualification import require_qualification,EXPORT_RELEASE,SCOPE_SHA256

ROOT=Path(__file__).parent;NATIVE=ROOT/'primary_native'
COPIED=('mesh.vol.bin','boundary_rule.json','solution.sol','result.json','fe_layout.json',
        'reload_checks.json','mesh_identity.json','normal_orientation.json','reload_normal_orientation.json')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def native_digest():
    manifest=json.loads((NATIVE/'SOURCE.json').read_text())
    expected={'fom.py','reference_export.py','shared_lift.py','preflight.py','scalars.py',
              'oriented_normals.py','immutable_rule.py','reference_primary.py','reference_scope.py'}
    if set(manifest['files'])!=expected:raise ValueError('exact primary source closure required')
    for name,digest in manifest['files'].items():
        if sha(NATIVE/name)!=digest:raise ValueError('primary helper source changed: '+name)
    return sha(NATIVE/'SOURCE.json')


def source_digest():
    return hashlib.sha256(json.dumps({'entry':sha(__file__),'native':native_digest(),
        'qualification':sha(ROOT/'qualification.py'),'scope':SCOPE_SHA256,
        'adapter':sha(ROOT/'fe_adapter.py')},sort_keys=True).encode()).hexdigest()


def verify(original,truth_path,release_path):
    original=Path(original).resolve();release=json.loads(Path(release_path).read_text())
    if (release.get('schema')!='ns-obstacle-fd1-primary-materialization-release-v1' or
        release.get('authorized_stage')!='training_primary_materialization' or
        release.get('source_sha256')!=source_digest() or release.get('split')!='training' or
        release.get('single_execution') is not True):raise ValueError('exact reviewed training materialization release required')
    if sha(truth_path)!=release['reference_admission_sha256']:raise ValueError('actual truth admission differs')
    qualification=require_qualification(json.loads(Path(truth_path).read_text()))
    if qualification!=require_qualification(release.get('reference_qualification')):raise ValueError('truth scope differs')
    if sha(original/'EXPORT-MANIFEST.json')!=release['original_export_sha256']:raise ValueError('released primary case differs')
    config=json.loads((original/'config.json').read_text());re=release['Re'];mode=release.get('source_storage_mode')
    if mode not in ('coefficient_primary','legacy_expanded_Re20'):raise ValueError('explicit source storage mode required')
    if mode=='legacy_expanded_Re20' and (re!=20. or config.get('release')!='G2_steady_control_candidate' or (original/'PRIMARY-REFERENCE.json').exists()):
        raise ValueError('only the separately released legacy expanded Re20 baseline is supported')
    if (re not in [15+1.25*j for j in range(9)] or abs(config['nu']-.02/re)>1e-15 or
        (config.get('scope_contract_sha256')!=SCOPE_SHA256 if mode=='coefficient_primary' else config.get('scope_contract_sha256') not in (None,SCOPE_SHA256))):raise ValueError('exact scoped training state required')
    return original,release,qualification


def derive(original,out,config,primary,fields,replay,release,qualification,digest_array):
    """Write all actual arrays before applying exact original-content gates."""
    descriptors=primary['derived_full_rule_arrays'];volume=fields['volume']
    selected={k:volume[k] for k in descriptors};archive=out/'full_rule.npz'
    if sum(a.nbytes for a in selected.values())>2**30:raise MemoryError('existing expanded case archive cap')
    np.savez_compressed(archive,**selected)
    actual={k:{'dtype':a.dtype.str,'shape':list(a.shape),'content_sha256':digest_array(a)} for k,a in selected.items()}
    record={'schema':'ns-obstacle-fd1-primary-materialization-v1','physical_admission':False,
        'status':'pending_exact_content_check','reference_qualification':qualification,
        'reference_admission_sha256':release['reference_admission_sha256'],
        'source_sha256':source_digest(),'original_directory':str(original),
        'original_export_sha256':sha(original/'EXPORT-MANIFEST.json'),
        'original_config_sha256':sha(original/'config.json'),'source_storage_mode':release.get('source_storage_mode','coefficient_primary'),
        'original_derived_array_descriptors':descriptors,'actual_derived_array_descriptors':actual,
        'new_full_rule_archive_sha256':sha(archive),'historical_ZIP_identity_claimed':False,
        'original_files_mutated':False,'reconstruction':replay,'Re':release['Re'],
        'role_only_configuration_changes':{'release':{'before':config.get('release'),'after':EXPORT_RELEASE},
                                           'split':{'before':config.get('split'),'after':'training'}}}
    if release.get('source_storage_mode','coefficient_primary')=='coefficient_primary':
        record.update(primary_manifest_sha256=sha(original/'PRIMARY-REFERENCE.json'),primary_coefficients_sha256=primary['primary_sha256'],immutable_geometry_rules=primary['immutable_geometry_rules'])
    else:
        record['original_full_rule_archive_sha256']=sha(original/'full_rule.npz')
        record['role_only_configuration_changes']['scope_contract_sha256']={'before':config.get('scope_contract_sha256'),'after':SCOPE_SHA256}
    path=out/'MATERIALIZATION.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    if actual!=descriptors:raise ArithmeticError('regenerated field bits differ; retain actual arrays for scoped review')
    with np.load(archive,allow_pickle=False) as saved:
        if set(saved.files)!=set(selected) or any(digest_array(saved[k])!=actual[k]['content_sha256'] for k in selected):
            raise ArithmeticError('new lossless container replay differs')
    for name in COPIED:
        shutil.copyfile(original/name,out/name)
        if sha(out/name)!=sha(original/name):raise ArithmeticError('immutable producer certificate copy differs')
    derived=dict(config,release=EXPORT_RELEASE,split='training',scope_contract_sha256=SCOPE_SHA256)
    (out/'config.json').write_text(json.dumps(derived,indent=2)+'\n')
    record.update(status='exact_content_replay_passed_pending_independent_admission',
                  derived_config_sha256=sha(out/'config.json'),original_config_preserved=True)
    path.write_text(json.dumps(record,indent=2)+'\n')
    source_export=json.loads((original/'EXPORT-MANIFEST.json').read_text())
    export={'schema':'ns-obstacle-derived-fd1-fe-export-v1','physical_admission':False,
        'mesh_filename':source_export['mesh_filename'],'mesh_format':source_export['mesh_format'],
        'reload_policy':source_export['reload_policy'],'case':{'Re':release['Re'],'split':'training','release':EXPORT_RELEASE},
        'original_export_sha256':release['original_export_sha256'],'original_storage_contract_preserved_at_source':True,
        'materialization_sha256':sha(path),'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'EXPORT-MANIFEST.json').write_text(json.dumps(export,indent=2)+'\n')
    return record


def materialize(original,truth_path,release_path,out):
    original,release,qualification=verify(original,truth_path,release_path)
    if (sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID') or
        os.environ.get('JAX_ENABLE_X64')!='true' or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest' or
        Path(release['job_directory']).resolve()!=Path.cwd().resolve()):raise RuntimeError('released fresh cluster GPU execution required')
    import jax
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled:raise RuntimeError('GPU/f64 preflight failed')
    out=Path(out)
    if out.resolve()!=Path(release['output_directory']).resolve():raise ValueError('released materialization output differs')
    with (Path.cwd()/f"MATERIALIZATION-CLAIM-Re{release['Re']}.json").open('x') as claim:
        json.dump({'release_sha256':sha(release_path),'original_export_sha256':release['original_export_sha256'],'job_id':os.environ['SLURM_JOB_ID']},claim)
    out.mkdir(exist_ok=False)
    started=time.perf_counter();final={'status':'started','release_sha256':sha(release_path),'physical_admission':False}
    try:
        names=json.loads((NATIVE/'SOURCE.json').read_text())['files']
        if any(name[:-3] in sys.modules for name in names):raise RuntimeError('fresh uncontaminated reconstruction interpreter required')
        sys.path.insert(0,str(NATIVE))
        from reference_primary import load_primary_case,reconstruct_primary
        from immutable_rule import array_digest
        if release['source_storage_mode']=='coefficient_primary':
            config,primary,_,_=load_primary_case(original)
            expanded=sum(np.prod(d['shape'],dtype=np.int64)*np.dtype(d['dtype']).itemsize for d in primary['derived_full_rule_arrays'].values())
        else:
            import zipfile
            export=json.loads((original/'EXPORT-MANIFEST.json').read_text())
            for name in (*COPIED,'config.json','full_rule.npz'):
                if sha(original/name)!=export['files'].get(name):raise ValueError('legacy runtime input differs: '+name)
            config=json.loads((original/'config.json').read_text())
            fixed={'kind':'dfg','outlet':'gradient','xmin':0.,'xmax':2.2,'H':.41,'center':[.2,.2],'radius':.05,'U':.2,'D':.1,'precision':'float64'}
            if any(config.get(k)!=v for k,v in fixed.items()):raise ValueError('legacy fixed geometry/scales differ')
            reload=json.loads((original/'reload_checks.json').read_text())
            execution=reload.get('execution',{})
            if reload.get('passed') is not True or execution.get('fresh_interpreter') is not True or execution.get('pid')==execution.get('producer_pid'):
                raise ValueError('legacy fresh-process replay certificate required')
            with zipfile.ZipFile(original/'full_rule.npz') as z:expanded=sum(x.file_size for x in z.infolist())
        if expanded>2**30 or shutil.disk_usage(out).free<10*2**30+2*expanded+64*2**20:
            raise MemoryError('materialization and durable-copy disk reserve or case archive cap exceeded')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+4*expanded+2*2**30>16*2**30:
            raise MemoryError('prospective reconstruction working reserve exceeds16GiB')
        if release['source_storage_mode']=='coefficient_primary':
            import ngsolve as ng
            fields,replay=reconstruct_primary(original,ng)
        else:
            with np.load(original/'full_rule.npz',allow_pickle=False) as archive:volume={k:archive[k] for k in archive.files}
            if any(a.dtype.kind not in 'fiu' or not np.isfinite(a).all() for a in volume.values()):raise ValueError('finite numeric legacy arrays required')
            primary={'derived_full_rule_arrays':{k:{'dtype':a.dtype.str,'shape':list(a.shape),'content_sha256':array_digest(a)} for k,a in volume.items()}}
            fields={'volume':volume};replay={'no_PDE_solve':True,'new_native_replay_performed':False,
                'method':'lossless original expanded arrays; original independent native replay certificate retained unchanged'}
        derive(original,out,config,primary,fields,replay,release,qualification,array_digest)
        final['status']='exact_reconstruction_completed_pending_independent_review'
    except Exception as exc:final.update(status='failed',exception=repr(exc));raise
    finally:
        final.update(elapsed_seconds=time.perf_counter()-started,peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     actual_output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
        (out/'MATERIALIZATION-EXECUTION.json').write_text(json.dumps(final,indent=2)+'\n')
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','truth','release','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();materialize(a.original,a.truth,a.release,a.out)
