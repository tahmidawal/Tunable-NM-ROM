"""Prospective complete-rule cache and portable primary retention proof."""
from pathlib import Path
import argparse,copy,json,resource,shutil,subprocess,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from compare_fields import load_candidate
from export_batch_control import retain_array
from immutable_rule import NativeRuleCache,CommonPointCache,write_cache,array_digest
from oriented_normals import fluid_outward_normal
import reference_export as rex
from reference_scope import FD1_CONTRACT_SHA256
ACTIVE=None


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);arrays=a.out/'arrays';arrays.mkdir()
    plan=json.loads(a.plan.read_text());records=[];report={'passed':False,'physical_admission':False,'records':records,'one_new_producer_invocation':True,'timings':'complete phase resource measurements, no cross-job speedup claim'}
    global ACTIVE
    ACTIVE=(a.out,report)
    import ngsolve as ng
    import jax,jax.numpy as jnp
    if jax.default_backend()!='gpu' or not jax.config.x64_enabled or jax.config.jax_default_matmul_precision!='highest':raise RuntimeError('GPU/f64/highest required')
    burn_matrix=jnp.arange(65536,dtype=jnp.float64).reshape(256,256)/65536
    def burn():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3:raise MemoryError('48GiB readout guard')
        for _ in range(3):jnp.matmul(burn_matrix,burn_matrix.T).block_until_ready()
    def save():rex.dump(a.out/'control.json',report)
    def compare(tag,actual,expected,cfg,extra):
        checks={};refs={}
        for key,original in expected.items():
            value=np.asarray(actual[key]);original=np.asarray(original);refs[key]={'actual':retain_array(arrays,value),'expected':retain_array(arrays,original)}
            if original.dtype.kind in 'USbiu':checks[key]={'value':0. if np.array_equal(value,original) else 1.}
            else:checks[key]=rex.relative_max_record(value,original,{'points':cfg['D'],'velocity':cfg['U'],'velocity_gradient':cfg['U']/cfg['D'],'pressure':cfg['U']**2}.get(key,1.))
        passed=all(x['value']<=rex.RELOAD_TOL for x in checks.values());records.append({'tag':tag,'passed':passed,'checks':checks,'arrays':refs,**extra});save()
        if not passed:raise ArithmeticError('cache proof differs: '+tag)
    fine=None
    for entry in plan['cases']:
        folder,cfg,result=load_candidate(entry['case']);ng.SetNumThreads(cfg['threads']);mesh_path=folder/rex.MESH_FILENAME
        with ng.TaskManager():
            mesh=ng.Mesh(str(mesh_path));V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
            layout,digest=rex.layout_record(mesh,V,Q,ng)
            if {'layout':layout,'sha256':digest}!=json.loads((folder/'fe_layout.json').read_text()):raise ValueError('retained layout mismatch')
            state=ng.GridFunction(X);state.Load(str(folder/'solution.sol'));u,pres=state.components
            normal,normal_record=fluid_outward_normal(mesh,cfg['center'],order=16)
            base={'mesh_sha256':rex.sha_file(mesh_path),'layout_sha256':digest,'geometry':rex.geometry_signature(cfg),'ngsolve_version':ng.__version__}
            descriptors={}
            for order in entry['orders']:
                local_cfg={**cfg,'diagnostic_order':order,'max_volume_points':2000000 if order==16 else 1300000};pair={}
                for boundary in (False,True):
                    region='boundary' if boundary else 'volume';legacy_seconds=None
                    if order==cfg['diagnostic_order']:
                        if boundary:
                            rows=json.loads((folder/'boundary_rule.json').read_text());keys={'points':'point','weights':'weight','velocity':'velocity','pressure':'pressure','velocity_gradient':'velocity_gradient','normals':'normal','raw_normals':'raw_normal'}
                            expected={k:np.asarray([x[v] for x in rows],dtype=np.float64) for k,v in keys.items()};expected['labels']=np.asarray([x['label'] for x in rows])
                        else:
                            with np.load(folder/'full_rule.npz',allow_pickle=False) as z:expected={k:z[k] for k in ('points','weights','jacobians','velocity','pressure','velocity_gradient')}
                    else:
                        burn();tick=time.perf_counter();expected=rex._mapped_state(mesh,u,pres,order,boundary,ng,normal=normal,volume_point_limit=2000000,batched=False);legacy_seconds=time.perf_counter()-tick
                        expected={k:np.asarray(v) for k,v in expected.items()}
                    geometry={k:np.asarray(v) for k,v in expected.items() if k not in ('velocity','pressure','velocity_gradient')}
                    metadata={**base,'mode':'native_element_rule','region':region,'order':order,'point_count':len(geometry['weights']),'element_rule_record':[[int(el.nr),str(el.type),len(ng.IntegrationRule(el.type,order))] for el in mesh.Elements(ng.BND if boundary else ng.VOL)]}
                    if boundary:metadata['normal_orientation_factors']=normal_record['factors']
                    tick=time.perf_counter();descriptor=write_cache(a.out/'caches'/entry['name']/str(order)/region,metadata,geometry);serialization=time.perf_counter()-tick;pair[region]=descriptor
                    burn();tick=time.perf_counter();cache=NativeRuleCache(ng,mesh,mesh_path,digest,local_cfg,descriptor,space=X,normal=normal if boundary else None,normal_factors=normal_record['factors'] if boundary else None);build=time.perf_counter()-tick
                    burn();tick=time.perf_counter();changing,work=cache.evaluate(state);elapsed=time.perf_counter()-tick
                    compare(f"{entry['name']}/{order}/{region}",{**geometry,**changing},expected,local_cfg,{'legacy_scalar_seconds':legacy_seconds,'geometry_serialization_seconds':serialization,'cache_construction_seconds':build,'actual_cached_evaluation_seconds':elapsed,'mapping':cache.certificate,'work':work})
                    if entry['name']=='p4_g4_h3' and order==12 and not boundary:
                        changed=ng.GridFunction(X);changed.vec.data=state.vec;delta=1e-5*np.sin(np.arange(X.ndof,dtype=np.float64)+1)*np.asarray(X.FreeDofs(),dtype=bool);changed.vec.FV().NumPy()[:]+=delta
                        burn();tick=time.perf_counter();oracle=rex._mapped_state(mesh,*changed.components,order,False,ng,volume_point_limit=1300000,batched=False);scalar_seconds=time.perf_counter()-tick
                        burn();actual,changed_work=cache.evaluate(changed)
                        if np.array_equal(actual['velocity'],changing['velocity']):raise ArithmeticError('changed coefficients did not change readout')
                        compare('deterministic_changed_state',actual,{k:oracle[k] for k in ('velocity','pressure','velocity_gradient')},local_cfg,{'coefficients':retain_array(arrays,changed.vec.FV().NumPy().copy()),'legacy_scalar_seconds':scalar_seconds,'work':changed_work})
                    del cache,expected,geometry,changing
                descriptors[str(order)]=pair
            if entry.get('common'):
                paths=entry['common'];z=np.load(paths['mapped'],allow_pickle=False);rule=np.load(paths['rule'],allow_pickle=False)
                geometry={'points':rule['points'],'weights':rule['weights'],**{k:z[k] for k in ('reference_coordinates','barycentric_coordinates','element_numbers','valid')}}
                descriptor=write_cache(a.out/'caches'/entry['name']/'common',{**base,'mode':'certified_common_points','region':'volume','point_count':len(rule['weights']),'original_lookup_evidence':{'filename':'original-validity.json','sha256':rex.sha_file(paths['validity'])}},geometry,{'original-validity.json':Path(paths['validity']).read_bytes()})
                burn();cache=CommonPointCache(ng,mesh,mesh_path,digest,cfg,descriptor,space=X);burn();changing,work=cache.evaluate(state,gradient=False)
                compare(entry['name']+'/common',changing,{k:z[k] for k in ('velocity','pressure')},cfg,{'mapping':cache.certificate,'cache_construction_seconds':cache.offline_seconds,'work':work});z.close();rule.close();del cache,changing
            if entry['name']=='p4_g4_h3':fine=(folder,cfg,descriptors['12'])
    folder,cfg,descriptors=fine;producer_cfg=copy.deepcopy(cfg);producer_cfg.update(retain_primary_control_oracle=True,reference_storage_mode='coefficient_primary',scope_contract_sha256=FD1_CONTRACT_SHA256,immutable_rule_cache=descriptors,immutable_mesh={'mesh_path':str(folder/rex.MESH_FILENAME),'mesh_sha256':rex.sha_file(folder/rex.MESH_FILENAME),'identity_path':str(folder/'mesh_identity.json'),'identity_sha256':rex.sha_file(folder/'mesh_identity.json')})
    rex.dump(a.out/'producer-config.json',producer_cfg);burn();tick=time.perf_counter()
    with (a.out/'producer.stdout').open('w') as stdout,(a.out/'producer.stderr').open('w') as stderr:subprocess.run([sys.executable,str(ROOT/'fom.py'),'--config',str(a.out/'producer-config.json'),'--out',str(a.out/'producer')],stdout=stdout,stderr=stderr,check=True,timeout=300,cwd=ROOT)
    report['complete_producer_seconds']=time.perf_counter()-tick;report['producer_result']=json.loads((a.out/'producer/result.json').read_text());save()
    shutil.copytree(a.out/'producer',a.out/'relocated-primary');burn();tick=time.perf_counter()
    with (a.out/'portable.stdout').open('w') as stdout,(a.out/'portable.stderr').open('w') as stderr:subprocess.run([sys.executable,str(ROOT/'g2/primary_replay.py'),'--case',str(a.out/'relocated-primary'),'--out',str(a.out/'portable-replay')],cwd=a.out/'relocated-primary',stdout=stdout,stderr=stderr,check=True,timeout=180)
    report['portable_fresh_process_seconds']=time.perf_counter()-tick
    with np.load(a.out/'producer-derived-control-oracle.npz',allow_pickle=False) as original,np.load(a.out/'portable-replay/volume.npz',allow_pickle=False) as reconstructed:compare('same_invocation_portable_replay',{k:reconstructed[k] for k in original.files},{k:original[k] for k in original.files},cfg,{'same_actual_producer_state':True})
    with np.load(folder/'full_rule.npz',allow_pickle=False) as original,np.load(a.out/'portable-replay/volume.npz',allow_pickle=False) as reconstructed:compare('complete_producer_portable_replay',{k:reconstructed[k] for k in original.files},{k:original[k] for k in original.files},cfg,{'independent_new_producer_vs_retained_original':True})
    report.update(passed=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if ACTIVE is not None:
            output,record=ACTIVE;record.update(passed=False,failure={'exception':repr(exc),'traceback':traceback.format_exc(),'raw_arrays_preserved':True});rex.dump(output/'control.json',record)
        raise
