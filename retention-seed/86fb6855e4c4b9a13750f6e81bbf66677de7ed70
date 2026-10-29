"""Actual coefficient readout on the immutable common rule; no solve or admission."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import reference_export as rex
from reference_primary import load_primary_case
from immutable_rule import CommonPointCache,array_digest
from oriented_normals import fluid_outward_normal
from scalars import scalar_value

LEGACY_RUNTIME=('config.json','mesh.vol.bin','fe_layout.json','solution.sol','result.json','reload_checks.json','mesh_identity.json','normal_orientation.json','reload_normal_orientation.json')

def open_case(folder,entry):
    export_path=folder/'EXPORT-MANIFEST.json';export=json.loads(export_path.read_text())
    if entry['retained']:
        if rex.sha_file(export_path)!=entry['export_manifest_sha256']:raise ValueError('retained export changed')
        for name in LEGACY_RUNTIME:
            if rex.sha_file(folder/name)!=export['files'][name]:raise ValueError('legacy actual-read member changed')
        cfg=json.loads((folder/'config.json').read_text());vectors=None
    else:
        cfg,primary,vectors,descriptors=load_primary_case(folder)
    if json.loads((folder/'reload_checks.json').read_text()).get('passed') is not True:raise ValueError('retained reload failed')
    return cfg,vectors

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--group',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--baseline-only',action='store_true');a=parser.parse_args()
    plan=json.loads(a.plan.read_text());group=next(g for g in plan['groups'] if g['id']==a.group);a.output.mkdir(parents=True,exist_ok=False)
    import ngsolve as ng
    import jax
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or jax.config.jax_default_matmul_precision!='highest':raise RuntimeError('GPU/f64/highest required')
    report={'admission':False,'Re':group['Re'],'split':group['split'],'states':{},'complete_common_rule':plan['common_rule'],'baseline_only':a.baseline_only}
    start=time.perf_counter();owners={};common_weights=None;common_points=None
    for entry in group['states']:
        if a.baseline_only and entry['variant']!='baseline':continue
        folder=Path('retained')/entry['id'] if entry['retained'] else Path('results/states')/entry['id']
        cfg,vectors=open_case(folder,entry);ng.SetNumThreads(cfg['threads']);space=entry['space'];tick=time.perf_counter()
        with ng.TaskManager():
            if space not in owners:
                mesh_path=folder/rex.MESH_FILENAME;mesh=ng.Mesh(str(mesh_path));V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
                layout,digest=rex.layout_record(mesh,V,Q,ng)
                if {'layout':layout,'sha256':digest}!=json.loads((folder/'fe_layout.json').read_text()):raise ValueError('native layout differs')
                cache_path=Path('shared/caches')/space/'common/cache.json';descriptor={'manifest_path':str(cache_path),'manifest_sha256':plan['common_cache_manifests'][space]}
                cache=CommonPointCache(ng,mesh,mesh_path,digest,cfg,descriptor,space=X)
                normal,normal_record=fluid_outward_normal(mesh,cfg['center'],order=16)
                owners[space]=(mesh,X,cache,normal,rex.sha_file(mesh_path),digest)
            mesh,X,cache,normal,mesh_sha,layout_sha=owners[space]
            if rex.sha_file(folder/rex.MESH_FILENAME)!=mesh_sha or json.loads((folder/'fe_layout.json').read_text())['sha256']!=layout_sha:raise ValueError('shared native owner identity differs')
            state=ng.GridFunction(X)
            if vectors is None:state.Load(str(folder/'solution.sol'))
            else:state.vec.FV().NumPy()[:]=vectors['solution_coefficients']
            fields,work=cache.evaluate(state,gradient=False)
            if common_weights is None:
                common_weights=cache.arrays['weights'];common_points=cache.arrays['points'];np.savez_compressed(a.output/'common-rule.npz',points=cache.arrays['points'],weights=common_weights)
            elif not np.array_equal(common_weights,cache.arrays['weights']) or not np.array_equal(common_points,cache.arrays['points']):raise ValueError('common rule weights differ')
            result=json.loads((folder/'result.json').read_text());forces=result['metrics']['forces'];force=np.asarray(forces['physical_symmetric_order16']['coefficient'])
            if entry['variant']=='baseline':
                u,p=state.components;gradient=ng.BoundaryFromVolumeCF(ng.Grad(u));stress=cfg['nu']*(gradient+gradient.trans)-p*ng.Id(2)
                replay=np.asarray([-scalar_value(ng.Integrate((stress*normal)[i],mesh,definedon=mesh.Boundaries('obstacle'),order=16),'force replay')/(.5*cfg['U']**2*cfg['D']) for i in range(2)])
            else:replay=None
            destination=a.output/(entry['id']+'.npz');np.savez_compressed(destination,**fields)
            report['states'][entry['id']]={'space':space,'variant':entry['variant'],'source_export_sha256':rex.sha_file(folder/'EXPORT-MANIFEST.json'),'solution_sha256':rex.sha_file(folder/'solution.sol'),'field_file':destination.name,'field_sha256':rex.sha_file(destination),'field_content_sha256':{k:array_digest(v) for k,v in fields.items()},'force_order16':force.tolist(),'force_replay_order16':None if replay is None else replay.tolist(),'other_force_study':forces,'mapping':cache.certificate,'work':work,'readout_seconds':time.perf_counter()-tick}
            rex.dump(a.output/'readout.json',report)
    report.update(complete=True,total_seconds=time.perf_counter()-start);rex.dump(a.output/'readout.json',report)

if __name__=='__main__':main()
