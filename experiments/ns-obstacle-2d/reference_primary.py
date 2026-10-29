"""Lossless coefficient-primary evidence; derived fields remain reproducible on demand.

No existing artifact is retired here. Only the current producer's temporary full-rule
export may be removed, after its successful independent fresh-process replay and after
all primary/state/result/manifest evidence is written. Producer exceptions before finalization keep it; later physical assessment may still fail.
"""
from pathlib import Path
import json,os,shutil
import numpy as np
import reference_export as rex
from immutable_rule import array_digest
from reference_scope import validate_geometry


def prepare_primary(out,cfg,layout,solution,stokes,prescribed_trace,reload_evidence):
    validate_geometry(cfg)
    if not cfg.get('immutable_mesh') or not cfg.get('immutable_rule_cache') or reload_evidence.get('passed') is not True:
        raise ValueError('coefficient-primary storage requires verified immutable geometry and fresh native replay')
    out=Path(out);vectors={'solution_coefficients':np.asarray(solution),'stokes_coefficients':np.asarray(stokes),'prescribed_FE_trace_coefficients':np.asarray(prescribed_trace)}
    n=layout['layout']['velocity_ndof']+layout['layout']['pressure_ndof']
    expected_sizes={'solution_coefficients':n,'stokes_coefficients':n,'prescribed_FE_trace_coefficients':layout['layout']['velocity_ndof']}
    if any(v.dtype!=np.dtype('float64') or v.shape!=(expected_sizes[k],) or not np.isfinite(v).all() for k,v in vectors.items()):raise ValueError('primary vectors must already be finite exact-layout float64')
    primary=out/'primary_coefficients.npz';np.savez_compressed(primary,**vectors)
    with np.load(primary,allow_pickle=False) as saved:
        if any(array_digest(saved[k])!=array_digest(v) for k,v in vectors.items()):raise ArithmeticError('lossless primary write/read failed')
    with np.load(out/'full_rule.npz',allow_pickle=False) as saved:
        derived={k:{'dtype':saved[k].dtype.str,'shape':list(saved[k].shape),'content_sha256':array_digest(saved[k])} for k in saved.files}
    if type(cfg.get('retain_primary_control_oracle',False)) is not bool:raise TypeError('primary proof oracle switch must be boolean')
    if cfg.get('retain_primary_control_oracle',False):
        oracle=out.parent/(out.name+'-derived-control-oracle.npz')
        if oracle.exists():raise FileExistsError('refusing to overwrite primary proof oracle')
        shutil.copyfile(out/'full_rule.npz',oracle)
        if rex.sha_file(oracle)!=rex.sha_file(out/'full_rule.npz'):raise ValueError('primary proof oracle copy differs')
    portable={}
    for region,descriptor in cfg['immutable_rule_cache'].items():
        manifest=Path(descriptor['manifest_path']).resolve()
        if rex.sha_file(manifest)!=descriptor['manifest_sha256']:raise ValueError('cache closure changed before retention')
        cache=json.loads(manifest.read_text());target=out/'geometry-cache'/region;target.mkdir(parents=True,exist_ok=False)
        members={manifest.name:descriptor['manifest_sha256'],cache['arrays_file']:cache['arrays_sha256'],**cache.get('support_files',{})}
        for name,digest in members.items():
            if Path(name).name!=name or rex.sha_file(manifest.parent/name)!=digest:raise ValueError('cache closure member mismatch')
            source=manifest.parent/name;destination=target/name
            try:os.link(source,destination)
            except OSError:shutil.copyfile(source,destination)
            if rex.sha_file(destination)!=digest:raise ValueError('portable retained cache differs')
        portable[region]={'manifest_path':str((target/manifest.name).relative_to(out)),'manifest_sha256':descriptor['manifest_sha256']}
    record={'schema':'ns-obstacle-coefficient-primary-v1','physical_admission':False,
            'mode':'coefficient_primary','primary_file':primary.name,'primary_sha256':rex.sha_file(primary),
            'primary_arrays':{k:{'dtype':v.dtype.str,'shape':list(v.shape),'content_sha256':array_digest(v)} for k,v in vectors.items()},
            'mesh_file':rex.MESH_FILENAME,'mesh_sha256':rex.sha_file(out/rex.MESH_FILENAME),'layout_sha256':layout['sha256'],
            'config_sha256':rex.sha_file(out/'config.json'),'immutable_geometry_rules':portable,'original_geometry_rule_descriptors':cfg['immutable_rule_cache'],'portable_cache_resolution':'manifest_path is relative to the retained case directory; all declared members are included in the recursive export manifest',
            'derived_full_rule_original_archive_sha256':rex.sha_file(out/'full_rule.npz'),'derived_full_rule_arrays':derived,
            'fresh_process_replay_sha256':rex.sha_file(out/'reload_checks.json'),'fresh_process_replay_passed':True,
            'primary_retention':'bitwise lossless f64 coefficient arrays and exact binary mesh/layout/config/immutable-rule identity',
            'derived_retention':'full-rule fields are numerically reproducible from primary coefficients and exact rules; original ZIP bytes are not claimed recoverable',
            'derived_numerical_replay_tolerance':rex.RELOAD_TOL,'derived_temporary_full_rule_retained':False,
            'raw_Jacobian_force_constraint_and_failure_evidence':'retained unchanged; no scientific threshold or coverage reduction',
            'source_sha256':{name:rex.sha_file(Path(__file__).with_name(name)) for name in ('reference_primary.py','immutable_rule.py','reference_export.py','reference_scope.py')},
            'scope_contract_sha256':cfg['scope_contract_sha256']}
    rex.dump(out/'PRIMARY-REFERENCE.json',record)
    return record


def load_primary_case(folder):
    """Verify a self-contained relocated primary case; never follow old absolute paths."""
    folder=Path(folder).resolve();export=json.loads((folder/'EXPORT-MANIFEST.json').read_text())
    for relative,digest in export['files'].items():
        path=(folder/relative).resolve()
        if not path.is_relative_to(folder) or rex.sha_file(path)!=digest:raise ValueError('primary export closure mismatch')
    record=json.loads((folder/'PRIMARY-REFERENCE.json').read_text())
    if record!=export.get('storage_contract') or record.get('schema')!='ns-obstacle-coefficient-primary-v1':raise ValueError('explicit primary storage record required')
    cfg=json.loads((folder/'config.json').read_text());validate_geometry(cfg)
    if rex.sha_file(folder/'config.json')!=record['config_sha256'] or rex.sha_file(folder/rex.MESH_FILENAME)!=record['mesh_sha256']:raise ValueError('primary configuration/mesh identity mismatch')
    archive=folder/record['primary_file']
    if rex.sha_file(archive)!=record['primary_sha256'] or archive.stat().st_size>2**30:raise ValueError('primary archive hash/size mismatch')
    import zipfile
    with zipfile.ZipFile(archive) as zipped:
        if sum(x.file_size for x in zipped.infolist())>2**30:raise MemoryError('primary expanded archive cap')
    with np.load(archive,allow_pickle=False) as saved:vectors={k:saved[k] for k in saved.files}
    if set(vectors)!=set(record['primary_arrays']):raise ValueError('primary array inventory changed')
    for key,value in vectors.items():
        if value.dtype!=np.dtype('float64') or not np.isfinite(value).all() or array_digest(value)!=record['primary_arrays'][key]['content_sha256']:raise ValueError('primary coefficient bits changed')
    descriptors={}
    for region,descriptor in record['immutable_geometry_rules'].items():
        path=(folder/descriptor['manifest_path']).resolve()
        if not path.is_relative_to(folder) or rex.sha_file(path)!=descriptor['manifest_sha256']:raise ValueError('portable geometry closure escaped or changed')
        descriptors[region]={**descriptor,'manifest_path':str(path)}
    return cfg,record,vectors,descriptors


def reconstruct_primary(folder,ng):
    """Actual fresh-mesh CF evaluation, not a stored-field lookup or PDE solve."""
    from immutable_rule import NativeRuleCache
    from oriented_normals import fluid_outward_normal
    folder=Path(folder).resolve();cfg,record,vectors,descriptors=load_primary_case(folder)
    mesh=ng.Mesh(str(folder/rex.MESH_FILENAME));V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
    layout,digest=rex.layout_record(mesh,V,Q,ng)
    if digest!=record['layout_sha256'] or json.loads((folder/'fe_layout.json').read_text())!={'layout':layout,'sha256':digest}:raise ValueError('fresh primary FE layout mismatch')
    state=ng.GridFunction(X);state.vec.FV().NumPy()[:]=vectors['solution_coefficients']
    if array_digest(state.vec.FV().NumPy())!=array_digest(vectors['solution_coefficients']):raise ValueError('fresh native state differs from primary bits')
    normal,normal_certificate=fluid_outward_normal(mesh,cfg['center'],order=cfg['diagnostic_order'])
    caches={region:NativeRuleCache(ng,mesh,folder/rex.MESH_FILENAME,digest,cfg,descriptor,space=X,normal=normal if region=='boundary' else None,normal_factors=normal_certificate['factors'] if region=='boundary' else None) for region,descriptor in descriptors.items()}
    fields={};certificates={}
    for region,cache in caches.items():
        changing,work=cache.evaluate(state);fields[region]={**cache.arrays,**changing};certificates[region]={'mapping':cache.certificate,'offline_seconds':cache.offline_seconds,'evaluation':work}
    fields['volume'].update(vectors)
    distances,_=rex.obstacle_quadrature(fields['volume']['points'],fields['volume']['weights'],cfg)
    if distances is not None:fields['volume']['signed_obstacle_distance']=distances
    return fields,{'layout_sha256':digest,'mesh_sha256':record['mesh_sha256'],'no_PDE_solve':True,'fresh_native_state_from_primary':True,'portable_closure_only':True,'rules':certificates}
