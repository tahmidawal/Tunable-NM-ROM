"""Pure metadata/launch tests using actual admitted plan; no native work or release."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy,json,subprocess,sys,tempfile
import materialize_batch2d as batch
import stage_materialization_packet as staging
ROOT=Path(__file__).parent

def main():
 packet=ROOT/'staging/training-materialize01';plan_path=packet/'BATCH-PLAN.json';inventory=packet/'INPUT-INVENTORY.json';truth=packet/'FD1-REFERENCE-ADMISSION.json';plan=json.loads(plan_path.read_text())
 # In-memory validation fixture only; never written as approval and never executed.
 fixture={'schema':'ns-obstacle-nine-training-materialization-batch-release-v1','source_sha256':batch.source_digest(),'plan_sha256':batch.sha(plan_path),
  'input_inventory_sha256':batch.sha(inventory),'reference_admission_sha256':batch.sha(truth),'allocation_seconds':450,'remaining_fine_preparation_seconds':1350,'single_execution':True,'authorize_exact_nine_child_releases':True}
 assert len(batch.released_cases(plan,fixture,plan_path,inventory,truth))==9
 for field,value in [('source_sha256','0'*64),('reference_admission_sha256','0'*64),('allocation_seconds',451),('authorize_exact_nine_child_releases',False)]:
  bad=dict(fixture);bad[field]=value
  try:batch.released_cases(plan,bad,plan_path,inventory,truth)
  except ValueError:pass
  else:raise AssertionError(field)
 bad=copy.deepcopy(plan);bad['cases'][1]['release_candidate']['Re']=15.
 try:batch.released_cases(bad,fixture,plan_path,inventory,truth)
 except ValueError:pass
 else:raise AssertionError('cross-case binding')
 assert batch.remaining(100.,160.)==360 and batch.remaining(100.,520.)==0
 for start,now in [(float('nan'),1.),(2.,1.)]:
  try:batch.remaining(start,now)
  except ValueError:pass
  else:raise AssertionError('invalid allocation clock')
 with tempfile.TemporaryDirectory() as folder:
  folder=Path(folder);stdout=folder/'stdout';stderr=folder/'stderr'
  def fail(argv,**kwargs):
   assert argv[0]==sys.executable and argv[1]==str(ROOT/'materialize_reference2d.py') and kwargs['timeout']==12
   kwargs['stdout'].write(b'partial child output');kwargs['stderr'].write(b'actual failure');return SimpleNamespace(returncode=1)
  with stdout.open('wb') as out,stderr.open('wb') as err,patch.object(batch.subprocess,'run',fail):assert batch.launch(plan['cases'][0],truth,folder/'unused',12,out,err).returncode==1
  assert stdout.read_bytes()==b'partial child output' and stderr.read_bytes()==b'actual failure'
  bad_release=folder/'invalid.json';bad_release.write_text('{}')
  try:staging.stage(packet,bad_release,folder)
  except ValueError:pass
  else:raise AssertionError('unreleased staging accepted')
  assert not (packet/'payload').exists()
 result={'passed':True,'actual_reference_admission_sha256':batch.sha(truth),'plan_sha256':batch.sha(plan_path),'source_sha256':batch.source_digest(),
  'exact_nine_training_metadata_checked':True,'mismatched_release_rejected':True,'cumulative_setup_time_and_finalization_checked':True,'fresh_child_command_and_failure_streams_checked':True,
  'staging_without_release_rejected_before_output':True,'native_execution':False,'approval_written':False,'training_or_validation':False}
 (ROOT/'checks/materialization-batch-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
