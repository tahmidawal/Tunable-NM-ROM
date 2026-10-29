"""Freeze only the first actual FD1 population batch, not an execution release."""
from pathlib import Path
import hashlib,json,shutil,sys,tempfile
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from dependencies import validate_install_metadata
sys.path.insert(0,str(B/'g2'))
from population_readout import LEGACY_RUNTIME
G=B/'g2/population-re20-01';G.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
rows=[]
def add(src,dst=None):
    p=B/src;rows.append({'source':src,'destination':dst or src,'sha256':sha(p),'bytes':p.stat().st_size})
plan_path=B/'g2/population01/PLAN.json';plan=json.loads(plan_path.read_text());group=next(g for g in plan['groups'] if g['Re']==20)
resource=json.loads((B/'g2/POPULATION-NOISE-RESOURCE-PLAN.json').read_text());templates={x['state']:x for x in resource['baseline_states']}
proof='artifacts/g2-cache-primary01/results/cache_primary_control';caches=set()
for state in group['states']:
    name=state['space'];base=templates[name]['baseline_directory']
    if state['retained']:
        folder=state['source_directory'];manifest=json.loads((B/folder/'EXPORT-MANIFEST.json').read_text())
        for member in ('EXPORT-MANIFEST.json',*LEGACY_RUNTIME):
            if member!='EXPORT-MANIFEST.json' and sha(B/folder/member)!=manifest['files'][member]:raise ValueError('legacy actual-read closure changed')
            add(folder+'/'+member,'retained/'+state['id']+'/'+member)
    else:
        add(state['config']);cfg=json.loads((B/state['config']).read_text())
        for descriptor in cfg['immutable_rule_cache'].values():caches.add(str(Path(descriptor['manifest_path']).parent.relative_to('shared/caches')))
    caches.add(name+'/common')
for name,entry in templates.items():
    for member in ('mesh.vol.bin','mesh_identity.json'):add(entry['baseline_directory']+'/'+member,'shared/meshes/'+name+'/'+member)
for cache in sorted(caches):
    directory=proof+'/caches/'+cache;manifest=json.loads((B/directory/'cache.json').read_text())
    for member,digest in {'cache.json':sha(B/directory/'cache.json'),manifest['arrays_file']:manifest['arrays_sha256'],**manifest.get('support_files',{})}.items():
        if sha(B/directory/member)!=digest:raise ValueError('immutable cache closure changed')
        add(directory+'/'+member,'shared/caches/'+cache+'/'+member)
dump(G/'RUNTIME-BOUND.json',{'status':'source_bound_requires_external_coordinator_release','population_plan_sha256':sha(plan_path),'allowed_groups':['re20'],'reference_infrastructure_seconds_remaining':28800-5575,'campaign_seconds_remaining':43200-5575-14400,'allocation_seconds':1000,'runner_seconds':900,'finalization_margin_seconds':20,'automatic_retry':False})
dump(G/'PROVISION.json',{'fresh_attempt':'g2-population-re20-01','purpose':'eight missing direct Re20 S/Q states and complete original-rule readout; no reference admission','no_shared_environment_mutation':True,'automatic_retry':False})
batch=(B/'g2/cache-primary01/run.sh').read_text().replace('cache-primary01','population-re20-01').replace('00:20:00','00:16:40').replace('set -euo pipefail','set -euo pipefail\nexport NSOBS_ALLOCATION_STARTED=$(date +%s)').replace('run_stage cache_primary_proof "$PY" g2/population-re20-01/run.py','run_stage population "$PY" g2/population_run.py --plan g2/population01/PLAN.json --group re20 --release g2/population-re20-01/RUNTIME-BOUND.json')
(G/'run.sh').write_text(batch);shutil.copyfile(B/'g2/cache-primary01/summary.py',G/'summary.py')
for name in ['fom.py','reference_export.py','reference_primary.py','reference_scope.py','immutable_rule.py','shared_lift.py','oriented_normals.py','scalars.py','dependencies.py','preflight.py','DEPENDENCIES.json','benchmark.json','manufactured_witness.json','g2/campaign-A6.json','g2/population_run.py','g2/population_readout.py','g2/population01/PLAN.json','g2/RE20-REPLAY-NOISE.json',*[f'g2/population-re20-01/{x}' for x in ('run.sh','summary.py','RUNTIME-BOUND.json')]]:add(name)
add('g2/population-re20-01/PROVISION.json','REPAIR.json')
if len({x['destination'] for x in rows})!=len(rows):raise ValueError('duplicate package destination')
record={'status':'source_candidate_no_execution_release','files':rows,'new_PDE_states':8,'retained_states':7,'parameter':20,'admission':False,'input_bytes':sum(x['bytes'] for x in rows),'unique_input_bytes_by_sha256':sum(next(x['bytes'] for x in rows if x['sha256']==h) for h in {x['sha256'] for x in rows}),'legacy_input_policy':'Only explicitly listed actual-read members, each bound to unchanged complete original export inventory; unused audit evidence remains in its retained archive. No missing dependency silently accepted.','bounds':{'allocation_seconds':1000,'runner_seconds':900,'allocation_GiB':64,'memory_guard_GiB':48,'cpu_threads':16,'producer_timeout_seconds':300,'max_live_campaign_jobs':2,'minimum_local_reserve_GiB':10,'local_collection_planning_GiB':14,'collection_policy':'Do not collect until actual archive+expanded+chunks+new Git objects and other active writes fit while preserving10GiB. Keep remote original meanwhile; no assumed compression or deduplication.'},'source_closure':{'scope_contract_sha256':plan['qualification_scope']['scope_contract_sha256'],'resource_plan_sha256':sha(B/'g2/POPULATION-NOISE-RESOURCE-PLAN.json'),'population_plan_sha256':sha(plan_path)}}
dump(G/'SOURCE-CANDIDATE.json',record)
with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary);shutil.copyfile(B/'DEPENDENCIES.json',folder/'DEPENDENCIES.json');shutil.copyfile(G/'PROVISION.json',folder/'REPAIR.json');metadata=validate_install_metadata(folder,Path('/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/g2-population-re20-01/deps'))
dump(G/'METADATA-PREFLIGHT.json',metadata);print(json.dumps({'files':len(rows),'input_bytes':record['input_bytes'],'candidate_sha256':sha(G/'SOURCE-CANDIDATE.json')}))
