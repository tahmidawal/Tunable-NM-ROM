"""Exercise production immutable-input reads without invoking native mesh math."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import reference_export as rex
from immutable_rule import load_cache
from population_readout import LEGACY_RUNTIME

class NativeMeshBoundaryReached(Exception):pass
def stop_before_native_mesh(*args,**kwargs):raise NativeMeshBoundaryReached()

def confined(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('staged input escapes package')
    return path

def validate(root,plan,group_id):
    root=Path(root).resolve();group=next(x for x in plan['groups']if x['id']==group_id)
    pinned=json.loads((root/'DEPENDENCIES.json').read_text())['candidates']['ngsolve']['version']
    sentinel=SimpleNamespace(__version__=pinned,Mesh=stop_before_native_mesh)
    representatives={};new_configs=[];legacy=[]
    for entry in group['states']:
        if entry['retained']:
            folder=root/'retained'/entry['id'];manifest=folder/'EXPORT-MANIFEST.json'
            if rex.sha_file(manifest)!=entry['export_manifest_sha256']:raise ValueError('legacy export manifest differs')
            export=json.loads(manifest.read_text())
            for name in LEGACY_RUNTIME:
                if rex.sha_file(folder/name)!=export['files'][name]:raise ValueError('legacy actual-read dependency differs')
            cfg=json.loads((folder/'config.json').read_text());legacy.append(entry['id'])
        else:
            path=confined(root,entry['config'])
            if rex.sha_file(path)!=entry['config_sha256']:raise ValueError('producer configuration differs')
            cfg=json.loads(path.read_text());new_configs.append((entry['id'],cfg))
        representatives.setdefault(entry['space'],cfg)
    checked_meshes=[];checked_caches=[]
    def actual_mesh_reads(cfg,spec,label):
        local=copy.deepcopy(cfg);local['immutable_mesh']={**spec,'mesh_path':str(confined(root,spec['mesh_path'])),'identity_path':str(confined(root,spec['identity_path']))}
        with tempfile.TemporaryDirectory(prefix='mesh-read-preflight-')as directory:
            try:rex.make_or_reuse_mesh(local,Path(directory),None,sentinel)
            except NativeMeshBoundaryReached:pass
            else:raise AssertionError('production mesh reader did not reach native boundary')
        identity=json.loads(Path(local['immutable_mesh']['identity_path']).read_text());checked_meshes.append({'label':label,'mesh_sha256':spec['mesh_sha256'],'identity_sha256':spec['identity_sha256'],'reload_checks_sha256':identity['reload_checks_sha256']})
        return local['immutable_mesh']['mesh_path'],identity['layout_sha256']
    # Exercise all five staged mesh/identity/reload triples, including the fine
    # reference mesh whose Re20 coefficient states are reused rather than solved.
    shared={}
    for space,cfg in representatives.items():
        folder=root/'shared/meshes'/space;spec={'mesh_path':str((folder/rex.MESH_FILENAME).relative_to(root)),'mesh_sha256':rex.sha_file(folder/rex.MESH_FILENAME),'identity_path':str((folder/'mesh_identity.json').relative_to(root)),'identity_sha256':rex.sha_file(folder/'mesh_identity.json')}
        shared[space]=actual_mesh_reads(cfg,spec,space)
    for name,cfg in new_configs:
        mesh_path,layout=actual_mesh_reads(cfg,cfg['immutable_mesh'],name)
        for region,descriptor in cfg['immutable_rule_cache'].items():
            descriptor={**descriptor,'manifest_path':str(confined(root,descriptor['manifest_path']))}
            metadata,arrays=load_cache(descriptor,mesh_path,layout,cfg,sentinel)
            checked_caches.append({'label':name+'/'+region,'manifest_sha256':descriptor['manifest_sha256'],'arrays_sha256':metadata['arrays_sha256'],'point_count':metadata['point_count']});del arrays
    for space,cfg in representatives.items():
        path=root/'shared/caches'/space/'common/cache.json';descriptor={'manifest_path':str(path),'manifest_sha256':plan['common_cache_manifests'][space]};mesh_path,layout=shared[space]
        metadata,arrays=load_cache(descriptor,mesh_path,layout,cfg,sentinel);checked_caches.append({'label':space+'/common','manifest_sha256':descriptor['manifest_sha256'],'arrays_sha256':metadata['arrays_sha256'],'point_count':metadata['point_count']});del arrays
    return {'passed':True,'group':group_id,'actual_production_mesh_reader_exercised':True,'native_mesh_constructor_called':False,'actual_production_cache_loader_exercised':True,'mesh_reads':checked_meshes,'cache_reads':checked_caches,'legacy_actual_read_closures':legacy}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--plan',type=Path,required=True);p.add_argument('--group',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=validate(a.root,json.loads(a.plan.read_text()),a.group);a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
