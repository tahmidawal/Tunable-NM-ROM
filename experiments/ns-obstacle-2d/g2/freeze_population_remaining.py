"""Freeze one of the remaining fourteen actual FD1 population groups, not an execution release."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,tempfile
parser=argparse.ArgumentParser();parser.add_argument("--group",required=True,choices=[g["id"]for g in json.loads((Path(__file__).resolve().parents[1]/"g2/population01/PLAN.json").read_text())["groups"]if g["id"]!="re20"]);args=parser.parse_args()
if args.group=="re20":raise ValueError("Re20 already has its separate attempt")
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from dependencies import validate_install_metadata
sys.path.insert(0,str(B/'g2'))
from population_readout import LEGACY_RUNTIME
G=B/'g2/population-rest01'/args.group;G.mkdir(parents=True,exist_ok=True);relative=G.relative_to(B);attempt='g2-population-'+args.group+'-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
rows=[]
def add(src,dst=None):
    p=B/src;row={'source':src,'destination':dst or src,'sha256':sha(p),'bytes':p.stat().st_size}
    if p.stat().st_size>50*1024**2 and src.startswith('artifacts/g2-cache-primary01/'):
        proof='artifacts/g2-cache-primary01/GIT-COLD-ARCHIVE.json'
        row['archive_source']={'manifest_source':proof,'manifest_sha256':sha(B/proof),'member':'g2-cache-primary01/'+str(Path(src).relative_to('artifacts/g2-cache-primary01'))}
    rows.append(row)
plan_path=B/'g2/population01/PLAN.json';plan=json.loads(plan_path.read_text());group=next(g for g in plan['groups'] if g['id']==args.group)
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
    for member in ('mesh.vol.bin','mesh_identity.json','reload_checks.json'):add(entry['baseline_directory']+'/'+member,'shared/meshes/'+name+'/'+member)
for cache in sorted(caches):
    directory=proof+'/caches/'+cache;manifest=json.loads((B/directory/'cache.json').read_text())
    for member,digest in {'cache.json':sha(B/directory/'cache.json'),manifest['arrays_file']:manifest['arrays_sha256'],**manifest.get('support_files',{})}.items():
        if sha(B/directory/member)!=digest:raise ValueError('immutable cache closure changed')
        add(directory+'/'+member,'shared/caches/'+cache+'/'+member)
dump(G/'RUNTIME-BOUND.json',{'status':'source_bound_requires_external_coordinator_release','population_plan_sha256':sha(plan_path),'allowed_groups':[args.group],'reference_infrastructure_seconds_remaining':28800-5917,'campaign_seconds_remaining':43200-5917-14400,'allocation_seconds':group['allocation_seconds'],'runner_seconds':group['runner_seconds'],'finalization_margin_seconds':20,'automatic_retry':False})
dump(G/'PROVISION.json',{'fresh_attempt':attempt,'purpose':'all15 direct states, complete original-rule readout and five fresh baseline replays; no reference admission','no_shared_environment_mutation':True,'automatic_retry':False})
batch=(B/'g2/cache-primary01/run.sh').read_text().replace('ctol_nsobs2d_cache-primary01','ctol_nsobs2d_population-'+args.group+'-01').replace('g2/cache-primary01',str(relative)).replace('00:20:00','00:25:00').replace('set -euo pipefail','set -euo pipefail\nexport NSOBS_ALLOCATION_STARTED=$(date +%s)')
batch=batch.replace('run_stage cache_primary_proof "$PY" '+str(relative/'run.py'),'run_stage population "$PY" g2/population_run.py --plan g2/population01/PLAN.json --group '+args.group+' --release '+str(relative/'RUNTIME-BOUND.json'))
batch=batch.replace('run_stage api_and_inverse_preflight','run_stage immutable_input_metadata "$PY" g2/validate_population_inputs.py --plan g2/population01/PLAN.json --group '+args.group+' --out results/immutable_input_preflight.json\nrun_stage api_and_inverse_preflight')
(G/'run.sh').write_text(batch);shutil.copyfile(B/'g2/cache-primary01/summary.py',G/'summary.py')
for name in ['fom.py','reference_export.py','reference_primary.py','reference_scope.py','immutable_rule.py','shared_lift.py','oriented_normals.py','scalars.py','dependencies.py','preflight.py','DEPENDENCIES.json','benchmark.json','manufactured_witness.json','g2/campaign-A6.json','g2/delta_retention.py','g2/git_archive_recovery.py','g2/population-rest01/SHARED-REUSE-INDEX.json','g2/population_run.py','g2/population_readout.py','g2/validate_population_inputs.py','g2/population01/PLAN.json',*[str(relative/x) for x in ('run.sh','summary.py','RUNTIME-BOUND.json')]]:add(name)
add(str(relative/'PROVISION.json'),'REPAIR.json')
if len({x['destination'] for x in rows})!=len(rows):raise ValueError('duplicate package destination')
record={'status':'source_candidate_no_execution_release','files':rows,'new_PDE_states':15,'retained_states':0,'fresh_baseline_replays':5,'parameter':group['Re'],'population_group':args.group,'admission':False,'input_bytes':sum(x['bytes'] for x in rows),'unique_input_bytes_by_sha256':sum(next(x['bytes'] for x in rows if x['sha256']==h) for h in {x['sha256'] for x in rows}),'legacy_input_policy':'Only explicitly listed actual-read members, each bound to unchanged complete original export inventory; unused audit evidence remains in its retained archive. No missing dependency silently accepted.','bounds':{'allocation_seconds':group['allocation_seconds'],'runner_seconds':group['runner_seconds'],'allocation_GiB':64,'memory_guard_GiB':48,'cpu_threads':16,'producer_timeout_seconds':300,'max_live_campaign_jobs':2,'minimum_local_reserve_GiB':10,'local_collection_planning_GiB':12,'collection_policy':'Remote delta packing by exact hash/size catalogue. Serial local recovery then cold-base cleanup; verified chunk creation and transport retirement BEFORE Git writes. Reserve both phases separately plus10GiB; actual measured gates and all other active writes remain mandatory. No compression credit.'},'source_closure':{'scope_contract_sha256':plan['qualification_scope']['scope_contract_sha256'],'resource_plan_sha256':sha(B/'g2/POPULATION-NOISE-RESOURCE-PLAN.json'),'population_plan_sha256':sha(plan_path),'reuse_index_sha256':sha(B/'g2/population-rest01/SHARED-REUSE-INDEX.json')}}
dump(G/'SOURCE-CANDIDATE.json',record)
with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary);shutil.copyfile(B/'DEPENDENCIES.json',folder/'DEPENDENCIES.json');shutil.copyfile(G/'PROVISION.json',folder/'REPAIR.json');metadata=validate_install_metadata(folder,Path('/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027')/attempt/'deps')
dump(G/'METADATA-PREFLIGHT.json',metadata);print(json.dumps({'files':len(rows),'input_bytes':record['input_bytes'],'candidate_sha256':sha(G/'SOURCE-CANDIDATE.json')}))
