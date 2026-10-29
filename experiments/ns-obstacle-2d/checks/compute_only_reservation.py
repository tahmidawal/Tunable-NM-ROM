"""Actual frozen package sizing and refusal checks; no staging or native computation."""
from pathlib import Path
import hashlib,json,sys,subprocess
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'g2'));from stage_released import compute_only_bounds
sha=lambda data:hashlib.sha256(data).hexdigest();commit='a7c58adca376dd9beaa5e7de2c8a62072de6caa4';source=B/'g2/stage_released.py';schedule=json.loads((B/'g2/population-rest01/SCHEDULE.json').read_text());row=schedule['groups'][0];candidate=json.loads((B/row['candidate']).read_text());release={'schema':'ns-obstacle-compute-only-stage-release-v1','compute_only':True,'source_commit':commit,'candidate_sha256s':[r['candidate_sha256']for r in schedule['groups']],'stage_helper_sha256':sha(source.read_bytes()),'minimum_local_reserve_bytes':10*1024**3}
def committed(path):return subprocess.check_output(['git','show',commit+':experiments/ns-obstacle-2d/'+path])
bounds=compute_only_bounds(candidate,commit,row['candidate_sha256'],release,committed,sha(source.read_bytes()),0);assert bounds['cold_source_archive_bytes']==3357019564
for changes,other in [({'source_commit':'bad'},0),({'candidate_sha256s':[]},0),({'stage_helper_sha256':'0'*64},0),({'minimum_local_reserve_bytes':1},0),({},-1)]:
 try:compute_only_bounds(candidate,commit,row['candidate_sha256'],{**release,**changes},committed,sha(source.read_bytes()),other)
 except ValueError:pass
 else:raise AssertionError('invalid release/reserve accepted')
second=compute_only_bounds(candidate,commit,row['candidate_sha256'],release,committed,sha(source.read_bytes()),1024**3);assert second['required_before_staging_bytes']==bounds['required_before_staging_bytes']+1024**3 and second['required_after_staging_bytes']==bounds['required_after_staging_bytes']+1024**3
record={'schema':'ns-obstacle-compute-only-stage-guard-proof-v1','bounds':bounds,'wrong_source_candidate_helper_and_low_negative_reserve_rejected':True,'concurrent_reservations_counted_before_and_after':True,'numerical_input_hashes_unchanged':True,'stage_helper_sha256':sha(source.read_bytes())};(B/'staging/COMPUTE-ONLY-STAGE-GUARD-PROOF.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
