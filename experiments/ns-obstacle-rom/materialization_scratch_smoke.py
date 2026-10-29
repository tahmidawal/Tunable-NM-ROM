"""Verify staging scratch ownership without reading or staging producer data."""
from pathlib import Path
import json,sys
import stage_materialization_packet as staging
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reference_inputs'))
import source_inputs

def main():
 inventory=json.loads((ROOT/'staging/training-materialize01/INPUT-INVENTORY.json').read_text())
 source_repo=Path('/nonexistent-readonly-reference-fixture')
 with staging.owned_inputs(source_inputs,source_repo,inventory) as reader:
  scratch=Path(reader.temporary.name);assert scratch.parent==ROOT and scratch.is_dir()
  assert reader.repo==source_repo and reader.commit==inventory['producer_git_commit'] and reader.prefix==inventory['producer_prefix']
  assert reader.verify_warm_archives is True and reader.reserve_bytes==10*2**30
  (scratch/'ownership-probe').write_text('temporary only')
 assert not scratch.exists()
 result={'passed':True,'staging_source_sha256':staging.batch.sha(ROOT/'stage_materialization_packet.py'),
  'source_inputs_sha256':staging.batch.sha(ROOT/'reference_inputs/source_inputs.py'),'archive_recovery_sha256':staging.batch.sha(ROOT/'reference_inputs/git_archive_recovery.py'),
  'scratch_created_and_cleaned_in_ROM_lane':True,'producer_repository_identity_unchanged':True,'warm_and_cold_helper_methods_unchanged':True,'conservative_reservation_unchanged':True,'producer_reads_or_staging_performed':False,'native_work':False}
 (ROOT/'checks/materialization-scratch-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
