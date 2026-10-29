"""Conservative first-block collection envelope from measured immutable inputs."""
from pathlib import Path
import hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def size(p):return sum(x.stat().st_size for x in p.rglob('*')if x.is_file())
plan=read(B/'g2/population01/PLAN.json');resource=read(B/'g2/POPULATION-NOISE-RESOURCE-PLAN.json');templates={x['state']:x for x in resource['baseline_states']};group=next(g for g in plan['groups']if g['Re']==20);cases=[]
for state in group['states']:
    if state['retained']:continue
    folder=B/templates[state['space']]['baseline_directory'];export=read(folder/'EXPORT-MANIFEST.json');bound=0
    for name in export['files']:
        if name=='full_rule.npz':continue
        path=folder/name
        if path.suffix=='.npz':
            with zipfile.ZipFile(path)as archive:bound+=sum(int(x.file_size*1.01)+4096 for x in archive.infolist())
        elif path.suffix=='.json' and name not in ('fe_layout.json','mesh_identity.json'):bound+=8*path.stat().st_size+2**20
        else:bound+=path.stat().st_size
    layout=read(folder/'fe_layout.json')['layout'];bound+=(3*layout['velocity_ndof']+2*layout['pressure_ndof'])*8+3*4096
    cfg=read(B/state['config'])
    for descriptor in cfg['immutable_rule_cache'].values():
        cache=B/'artifacts/g2-cache-primary01/results/cache_primary_control/caches'/str(Path(descriptor['manifest_path']).parent.relative_to('shared/caches'));bound+=size(cache)
    cases.append({'state':state['id'],'baseline_manifest_sha256':sha(folder/'EXPORT-MANIFEST.json'),'prospective_logical_case_bytes':bound})
candidate=read(B/'g2/population-re20-01/SOURCE-CANDIDATE.json');case_bytes=sum(x['prospective_logical_case_bytes']for x in cases)
field_bytes=15*(plan['common_rule']['point_count']*3*8+3*4096)
rule_bytes=plan['common_rule']['point_count']*3*8+3*4096
measurement=B/'g2/population-re20-01/DISK-PLAN-INITIAL-MEASUREMENT.json'
environment_bytes=read(measurement)['measured_dependency_tree_bytes']
auxiliary=256*1024**2
expanded=case_bytes+field_bytes+rule_bytes+candidate['input_bytes']+environment_bytes+auxiliary
archive=int(expanded*1.01)+16*1024**2
# Existing input files are already staged at the post-staging guard. Do not credit
# tar hardlinks, compression, shared meshes or Git deduplication in collection.
collection=expanded+3*archive+case_bytes+256*1024**2
record={'status':'prospective_first_block_only_not_execution_release','candidate_sha256':sha(B/'g2/population-re20-01/SOURCE-CANDIDATE.json'),'cases':cases,'case_policy':'Same immutable spaces and fixed Jacobian array schema: charge uncompressed NPZ bytes plus1percent and per-entry headers; variable JSON8times measured bytes plus1MiB each; new primary vectors and each geometry closure counted separately. No compression/dedup credit.','common_fields_bytes':field_bytes,'common_rule_bytes':rule_bytes,'dependency_measurement_sha256':sha(measurement),'measured_dependency_tree_bytes':environment_bytes,'input_bytes':candidate['input_bytes'],'auxiliary_allowance_bytes':auxiliary,'expanded_logical_bound_bytes':expanded,'archive_bound_bytes':archive,'complete_collection_bound_bytes':collection,'collection_components':'expanded plus local archive plus chunk working copies plus new Git objects plus all new case warm blobs plus256MiB transient producer allowance','declared_collection_reservation_bytes':14*1024**3,'fits_declared_reservation':collection<=14*1024**3,'untouched_local_reserve_bytes':10*1024**3,'effective_post_staging_submission_guard_bytes':24*1024**3,'free_bytes_at_generation':shutil.disk_usage(B).free,'condition':'Measure actual remote archive/expanded closure before collection. If it exceeds this envelope or other reservations consume headroom, retain remote originals and stop collection for review. Never delete unique evidence to fit.'}
if not record['fits_declared_reservation']:raise MemoryError('first block exceeds declared collection envelope')
out=B/'g2/population-re20-01/DISK-PLAN.json';out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
