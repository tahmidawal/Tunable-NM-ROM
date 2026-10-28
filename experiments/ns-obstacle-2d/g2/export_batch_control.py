"""Same-invocation native readout equivalence; retain repeated arrays losslessly."""
from pathlib import Path
import argparse,hashlib,json,resource,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from compare_fields import load_candidate
from oriented_normals import fluid_outward_normal
import reference_export as rex


def retain_array(folder,array):
    array=np.ascontiguousarray(array)
    header=json.dumps({'dtype':array.dtype.str,'shape':list(array.shape)},sort_keys=True).encode()
    digest=hashlib.sha256(header+memoryview(array).cast('B')).hexdigest()
    path=folder/(digest+'.npz')
    if not path.exists():np.savez_compressed(path,array=array)
    return {'path':str(path.relative_to(folder.parent)),'content_sha256':digest,
            'file_sha256':rex.sha_file(path),'dtype':array.dtype.str,'shape':list(array.shape)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    arrays_folder=args.out/'arrays';arrays_folder.mkdir();plan=json.loads(args.plan.read_text());folder,cfg,result=load_candidate(plan['case'])
    import ngsolve as ng
    import jax,jax.numpy as jnp
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or jax.config.jax_default_matmul_precision!='highest':raise RuntimeError('required GPU/f64/highest preflight')
    burn_array=jnp.arange(256*256,dtype=jnp.float64).reshape(256,256)/65536
    def burn_gpu():
        for _ in range(3):jnp.matmul(burn_array,burn_array.T).block_until_ready()
    ng.SetNumThreads(cfg['threads']);records=[];timings=np.full((2,3,2),np.nan)
    record={'passed':False,'physical_admission':False,'case_manifest_sha256':plan['case']['export_manifest_sha256'],'records':records,'same_original_coefficients':True,'no_PDE_solve':True,'timing_claim':'diagnostic readout only; not complete producer/query cost or cross-job speedup','physics_thresholds_and_rule_unchanged':True,'repetitions':3,'GPU_burn_before_each_timed_block':True,'array_retention':'lossless content-addressed deduplication; every invocation references its actual returned arrays'}
    def save():
        np.savez_compressed(args.out/'timing_repetitions.npz',seconds=timings,kinds=np.asarray(['volume','boundary']),modes=np.asarray(['scalar','batched']))
        rex.dump(args.out/'control.json',record)
    with ng.TaskManager():
        mesh=ng.Mesh(str(folder/rex.MESH_FILENAME));V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
        if mesh.ne>cfg['max_elements'] or X.ndof>cfg['max_dofs']:raise MemoryError('retained mesh/layout cap')
        layout,digest=rex.layout_record(mesh,V,Q,ng);assert json.loads((folder/'fe_layout.json').read_text())=={'layout':layout,'sha256':digest};record['layout_sha256']=digest
        state=ng.GridFunction(X);state.Load(str(folder/'solution.sol'));u,p=state.components
        normal,certificate=fluid_outward_normal(mesh,cfg['center'],order=cfg['diagnostic_order'],evidence_path=args.out/'normals.json');assert certificate['passed']
        with np.load(folder/'full_rule.npz',allow_pickle=False) as archive:original_volume={k:archive[k] for k in ('points','weights','velocity','velocity_gradient','pressure','jacobians')}
        original_boundary=json.loads((folder/'boundary_rule.json').read_text());keys={'points':'point','weights':'weight','velocity':'velocity','velocity_gradient':'velocity_gradient','pressure':'pressure','normals':'normal','raw_normals':'raw_normal'}
        original_boundary_arrays={k:np.asarray([row[v] for row in original_boundary],dtype=np.float64) for k,v in keys.items()}
        for kind,boundary in enumerate((False,True)):
            tag='boundary' if boundary else 'volume';expected=original_boundary_arrays if boundary else original_volume
            original_refs={key:retain_array(arrays_folder,value) for key,value in expected.items()}
            for repetition in range(3):
                for mode in ((0,1) if repetition%2==0 else (1,0)):
                    if int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024>48*1024**3:raise MemoryError('readout memory guard exceeded')
                    burn_gpu();tick=time.perf_counter()
                    actual=rex._mapped_state(mesh,u,p,cfg['diagnostic_order'],boundary,ng,normal=normal,volume_point_limit=rex.volume_point_cap(cfg),batched=bool(mode))
                    seconds=time.perf_counter()-tick;timings[kind,repetition,mode]=seconds;checks={};actual_refs={};retention_start=time.perf_counter()
                    for key,original in expected.items():
                        actual_refs[key]=retain_array(arrays_folder,actual[key])
                        scale={'points':cfg['D'],'velocity':cfg['U'],'velocity_gradient':cfg['U']/cfg['D'],'pressure':cfg['U']**2}.get(key,1.)
                        checks[key]=rex.relative_max_record(actual[key],original,scale)
                    labels_match=True
                    if boundary:
                        labels=np.asarray([row['label'] for row in original_boundary]);labels_match=actual['labels']==labels.tolist()
                        original_refs['labels']=retain_array(arrays_folder,labels);actual_refs['labels']=retain_array(arrays_folder,np.asarray(actual['labels']))
                    passed=labels_match and all(row['value']<=rex.RELOAD_TOL for row in checks.values())
                    records.append({'kind':tag,'repetition':repetition,'mode':'batched' if mode else 'scalar','checks':checks,'seconds_diagnostic_only':seconds,'retention_seconds':time.perf_counter()-retention_start,'passed':passed,'labels_match':labels_match,'original_arrays':original_refs,'actual_arrays':actual_refs});save()
                    if not passed:raise ArithmeticError('native readout differs from original retained evidence')
                    del actual
        record['passed']=all(row['passed'] for row in records);record['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024;record['peak_rss_scope']='process lifetime high-water';save()

if __name__=='__main__':main()
