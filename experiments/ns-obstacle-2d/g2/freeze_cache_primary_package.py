"""Freeze one prospective native cache/portable-primary proof, not admission."""
from pathlib import Path
import hashlib,json,shutil,sys
B=Path(__file__).resolve().parents[1];G=B/'g2/cache-primary01';G.mkdir(exist_ok=True);remote='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/g2-cache-primary01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
rows=[]
def add(src,dst):
 p=B/src;rows.append({'source':src,'destination':dst,'sha256':sha(p),'bytes':p.stat().st_size})
cases=[]
for name,source,mapping,orders in [
 ('p4_g4_h1','g2-p4trend/results/dfg_h1','g2-p4trend/results/comparison_dfg/dfg_h1',[12,16]),
 ('p4_g4_h2','g2-p4trend/results/dfg_h2','g2-p4trend/results/comparison_dfg/dfg_h2',[12,16]),
 ('p4_g4_h3','g2-p4trend/results/dfg_h3','g2-p4trend/results/comparison_dfg/dfg_h3',[12]),
 ('p3_g3_h3','g2-h3/results/dfg_h3','g2-crossp02/results/comparison_p_geometry/p3_g3_h3',[12,16]),
 ('p3_g4_h3','g2-p4controls/results/dfg_p3_g4_h3','g2-crossp02/results/comparison_p_geometry/p3_g4_h3',[12,16]),
 ('p4_g4_h3_q16','g2-solverquad01/results/quadrature_plus4',None,[16])]:
 folder='artifacts/'+source;manifest=B/folder/'EXPORT-MANIFEST.json';entry={'name':name,'case':{'directory':remote+'/inputs/'+name,'export_manifest_sha256':sha(manifest)},'orders':orders}
 for item in ['EXPORT-MANIFEST.json',*json.loads(manifest.read_text())['files']]:add(folder+'/'+item,'inputs/'+name+'/'+item)
 if mapping:
  source_base=Path('artifacts')/mapping;entry['common']={}
  for key,path in [('mapped',source_base.with_suffix('.npz')),('validity',source_base.with_name(source_base.name+'-validity.json')),('rule',source_base.parent/'common_rule.npz')]:
   dst='inputs/common/'+name+'-'+key+path.suffix;add(str(path),dst);entry['common'][key]=remote+'/'+dst
 cases.append(entry)
dump(G/'plan.json',{'cases':cases,'scope':'native equivalence only; no reference admission; unchanged physical equations and numerical thresholds','complete_producer_repetitions':1,'proof_process_seconds':900,'primary_scope_contract_sha256':'31ec75738dc47f8d960acf2f51dd355caf497a2681a3d499cc00faa138b28f76'})
dump(G/'PROVISION.json',{'fresh_attempt':'g2-cache-primary01','purpose':'immutable live-mesh readout and portable coefficient-primary proof','no_shared_environment_mutation':True,'automatic_retry':False})
(G/'run.sh').write_text((B/'g2/export-batch01/run.sh').read_text().replace('export-batch01','cache-primary01').replace('saved_state_batch_readout','cache_primary_proof'))
(G/'summary.py').write_text((B/'g2/export-batch01/summary.py').read_text())
(G/'run.py').write_text((B/'g2/export-batch01/run.py').read_text().replace("'new_PDE_solves':0","'new_PDE_solves':1").replace('export_batch_control.py','cache_primary_control.py').replace('export-batch01','cache-primary01').replace('export_batch_control','cache_primary_control'))
contract=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/FD1-SCOPE-CONTRACT.json');assert sha(contract)=='31ec75738dc47f8d960acf2f51dd355caf497a2681a3d499cc00faa138b28f76';shutil.copyfile(contract,G/contract.name)
for name in ['fom.py','reference_export.py','reference_primary.py','reference_scope.py','immutable_rule.py','shared_lift.py','oriented_normals.py','scalars.py','dependencies.py','preflight.py','DEPENDENCIES.json','benchmark.json','manufactured_witness.json','g2/campaign-A6.json','g2/compare_fields.py','g2/exact_lookup.py','g2/export_batch_control.py','g2/cache_primary_control.py','g2/primary_replay.py',*[f'g2/cache-primary01/{x}' for x in ('run.py','run.sh','summary.py','plan.json','FD1-SCOPE-CONTRACT.json')]]:add(name,name)
add('g2/cache-primary01/PROVISION.json','REPAIR.json')
record={'status':'source_review_candidate_no_execution_release','files':rows,'physical_admission':False,'new_PDE_solves':1,'bounds':{'allocation_seconds':1200,'process_seconds':900,'producer_seconds':300,'portable_replay_seconds':180,'allocation_GiB':64,'memory_stop_GiB':48,'volume_points_hard_cap':2000000,'cpus':16,'proof_output_planning_GiB':4,'complete_collection_planning_GiB':6,'other_lane_collection_reservation_GiB':3,'local_collection_policy':'Measure actual remote archive/expanded bytes before collection; retain remote originals if local reservation is insufficient. Full local archive+expanded+Git copies would require18GiB beyond existing data; do not allocate these blindly. Canonical directory move avoids a fourth copy; no unmeasured compression/dedup credit.'},'input_bytes':sum(x['bytes'] for x in rows),'resource_planning':'Scalar order16 readouts on four saved spaces, deterministic changed fine state oracle, all native/cache/common readouts, one original P4h3 producer with unchanged factor guards, and fresh relocated primary replay. No credited speedup; hard process/allocation stops. Canonical input bytes remain; content deduplication is prospective until measured.', 'scientific_scope':'All five baseline spaces, native rules12/16, full original P4 common mappings, actual changed coefficients, complete producer and portable fresh-process reconstruction. Same coordinates/weights/order, no dropped points, no threshold or parameter coverage change.'}
dump(G/'SOURCE-CANDIDATE.json',record)
# Exercise the actual installer metadata function against the staged metadata map.
sys.path.insert(0,str(B));from dependencies import validate_install_metadata
import tempfile
with tempfile.TemporaryDirectory() as temporary:
 root=Path(temporary);shutil.copyfile(B/'DEPENDENCIES.json',root/'DEPENDENCIES.json');shutil.copyfile(G/'PROVISION.json',root/'REPAIR.json');metadata=validate_install_metadata(root,Path(remote)/'deps')
dump(G/'METADATA-PREFLIGHT.json',metadata);print(len(rows),sha(G/'SOURCE-CANDIDATE.json'),record['input_bytes'])
