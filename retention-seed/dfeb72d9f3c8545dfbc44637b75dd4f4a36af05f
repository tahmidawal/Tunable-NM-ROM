"""Freeze the complete FD1 population; retained states are never silently rerun."""
from pathlib import Path
import copy, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_scope import FD1_CONTRACT_SHA256,FD1_SCOPE,validate_geometry

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    resource=ROOT/'g2/POPULATION-NOISE-RESOURCE-PLAN.json';proposal=json.loads(resource.read_text())
    target=ROOT/'g2/population01';proof=ROOT/'artifacts/g2-cache-primary01/results/cache_primary_control'
    templates={x['state']:x for x in proposal['baseline_states']};groups=[]
    for split,parameters in proposal['population'].items():
        for reynolds in parameters:
            tag=str(reynolds).replace('.','p');group={'id':'re'+tag,'Re':reynolds,'split':split,'states':[],'runner_seconds':900 if reynolds==20 else 1400,'allocation_seconds':1000 if reynolds==20 else 1500}
            for name,entry in templates.items():
                baseline=ROOT/entry['baseline_directory'];template=json.loads((baseline/'config.json').read_text())
                for variant in ('baseline','solver_100x','quadrature_plus4'):
                    retained=reynolds==20 and (variant=='baseline' or name=='p4_g4_h3')
                    state={'id':name+'_'+variant,'space':name,'variant':variant,'retained':retained}
                    if retained:
                        folder=baseline if variant=='baseline' else ROOT/'artifacts/g2-solverquad01/results'/variant
                        state.update(source_directory=str(folder.relative_to(ROOT)),export_manifest_sha256=sha(folder/'EXPORT-MANIFEST.json'))
                    else:
                        cfg=copy.deepcopy(template);cfg.update(release='G2_fixed_finite_channel_control_candidate',scope_contract_sha256=FD1_CONTRACT_SHA256,reference_storage_mode='coefficient_primary',purpose='direct FD1 parameter numerical qualification; not reference admission',split=split)
                        cfg['nu']=template['nu'] if reynolds==20 else cfg['U']*cfg['D']/reynolds
                        if variant=='solver_100x':
                            for key in ('linear_relative_tolerance','nonlinear_relative_tolerance'):cfg[key]/=100
                        if variant=='quadrature_plus4':cfg['assembly_bonus_order']+=4;cfg['diagnostic_order']+=4
                        cfg['max_volume_points']=2000000 if variant=='quadrature_plus4' else 1300000
                        cfg['immutable_mesh']={'mesh_path':'shared/meshes/'+name+'/mesh.vol.bin','mesh_sha256':sha(baseline/'mesh.vol.bin'),'identity_path':'shared/meshes/'+name+'/mesh_identity.json','identity_sha256':sha(baseline/'mesh_identity.json')}
                        cache_name='p4_g4_h3_q16' if name=='p4_g4_h3' and cfg['diagnostic_order']==16 else name
                        cfg['immutable_rule_cache']={region:{'manifest_path':f'shared/caches/{cache_name}/{cfg["diagnostic_order"]}/{region}/cache.json','manifest_sha256':sha(proof/'caches'/cache_name/str(cfg['diagnostic_order'])/region/'cache.json')} for region in ('volume','boundary')}
                        validate_geometry(cfg);path=target/'configs'/group['id']/(state['id']+'.json');dump(path,cfg)
                        state.update(config=str(path.relative_to(ROOT)),config_sha256=sha(path))
                    group['states'].append(state)
            groups.append(group)
    if len(groups)!=15 or sum(len(g['states']) for g in groups)!=225 or sum(s['retained'] for g in groups for s in g['states'])!=7:raise ValueError('frozen coverage mismatch')
    dump(target/'PLAN.json',{'schema':'FD1-complete-population-source-plan-v1','execution_released':False,'qualification_scope':FD1_SCOPE,'resource_plan_sha256':sha(resource),'groups':groups,'shared_training_geometry':templates['p4_g4_h3']['immutable_mesh_sha256'],'common_rule':proposal['common_rule'],'common_cache_manifests':{name:sha(proof/'caches'/name/'common/cache.json') for name in templates},'force_noise_rule':proposal['force_noise_rule'],'numerical_coverage':'all five baseline states and both S/Q controls at every parameter; no endpoint inheritance','execution_bounds':{'max_live_jobs':2,'allocation_seconds_by_group':'1000 for retained Re20 completion; 1500 for each new parameter','runner_seconds_by_group':'900 for Re20; 1400 otherwise','reference_infrastructure_total_seconds':28800,'campaign_total_seconds':43200,'automatic_retry':False}})
    print(json.dumps({'groups':len(groups),'states':225,'retained':7,'new_PDE_states':218,'plan_sha256':sha(target/'PLAN.json')}))

if __name__=='__main__':main()
