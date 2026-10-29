"""Bind genuine adopted training inputs to an unapproved single-allocation packet."""
from pathlib import Path
import argparse,hashlib,json,subprocess,shutil
import materialize_batch2d as batch
import materialize_reference2d as worker
from qualification import require_qualification
ROOT=Path(__file__).parent
JOB='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/training-materialize01'
def sha(p):return worker.sha(p)
def git(commit,path):return subprocess.check_output(['git','-c','gc.auto=0','show',commit+':'+path])
def build(args):
 truth=json.loads(args.truth.read_text());review=json.loads(args.review.read_text());qualification=require_qualification(truth)
 if (sha(args.truth)!=args.truth_sha256 or truth.get('schema')!='ns-obstacle-FD1-reference-admission-v1' or len(truth.get('admitted_references',[]))!=15
     or review.get('schema')!='ns-obstacle-FD1-population-independent-review-v1' or review.get('passed') is not True
     or truth.get('independent_review_sha256')!=sha(args.review) or truth.get('candidate_sha256')!=review.get('candidate_sha256')):raise ValueError('actual published and independently reviewed full admission required')
 pre=json.loads(args.precheck.read_text());assert pre['schema']=='ns-obstacle-nine-training-materialization-input-precheck-v1' and not pre['missing_runtime_bytes'] and not pre['pending_closure_declarations']
 assert sha(ROOT/'reference_inputs/source_inputs.py')=='bdde5ce4f17448235fa2e37d8d34c19f399d72b0294b0458127b196799e842cc'
 rows=[];cases=[];out=args.out.resolve();out.mkdir(exist_ok=False)
 for item in pre['rows']:
  re=item['Re'];identifier='re'+str(re).removesuffix('.0').replace('.','p');adopted=next(r for r in truth['admitted_references'] if r['Re']==re)
  for key in ('Re','split','export_manifest_sha256','config_sha256','mesh_sha256','layout_sha256','solution_sha256','storage_mode'):
   if adopted[key]!=(item['adopted_reference'].get(key) if key not in ('Re','split') else item[key]):raise ValueError('precheck is not actual adopted training member')
  if item['closure']:
   path=Path(item['closure']['path']);assert sha(path)==item['closure']['sha256'];source_rows=json.loads(path.read_text())['files']
  else:source_rows=[dict(f,destination=f['name']) for f in item['runtime_files']]
  expected={f['name']:f['sha256'] for f in item['runtime_files']};assert {r['destination']:r['sha256'] for r in source_rows}==expected
  for source in source_rows:
   row={k:source[k] for k in ('source','destination','sha256','bytes')};row['destination']='original/'+identifier+'/'+row['destination']
   p=args.reference_root/source['source'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==source['bytes'] and sha(p)==source['sha256']
   if 'archive_source' in source:
    row['archive_source']=source['archive_source'];a=source['archive_source']
    for commit in (args.reference_source,args.reference_mirror):assert hashlib.sha256(git(commit,'experiments/ns-obstacle-2d/'+a['manifest_source'])).hexdigest()==a['manifest_sha256']
   else:
    for commit in (args.reference_source,args.reference_mirror):assert hashlib.sha256(git(commit,'experiments/ns-obstacle-2d/'+row['source'])).hexdigest()==row['sha256']
   rows.append(row)
  candidate={'schema':'ns-obstacle-fd1-primary-materialization-release-candidate-v1','source_sha256':worker.source_digest(),'split':'training','single_execution':True,
   'reference_admission_sha256':sha(args.truth),'reference_qualification':qualification,'original_export_sha256':adopted['export_manifest_sha256'],
   'Re':re,'source_storage_mode':item['materialization_route'],'job_directory':JOB,'output_directory':JOB+'/results/'+identifier}
  cases.append({'id':identifier,'Re':re,'original_directory':'original/'+identifier,'release_candidate':candidate})
 assert [c['Re'] for c in cases]==batch.TRAIN and len(rows)==156
 resources=dict(pre['resources']);resources.update(native_environment_bound_bytes=2**30,remote_workspace_envelope_bytes=4*2**30,remote_workspace_plus_archive_envelope_bytes=8*2**30,
  conservative_recovery_reservations_unchanged=True,local_runtime_and_metadata_allowance_bytes=64*2**20)
 plan={'schema':'ns-obstacle-nine-training-materialization-plan-v1','source_sha256':batch.source_digest(),'builder_sha256':sha(__file__),
  'reference_admission_path':'FD1-REFERENCE-ADMISSION.json','reference_admission_sha256':sha(args.truth),'independent_reference_review_sha256':sha(args.review),
  'reference_qualification':qualification,'allocation_seconds':450,'finalization_reserve_seconds':30,'fine_preparation_remaining_seconds':1350,'job_directory':JOB,
  'cases':cases,'resources':resources,'execution_released':False,'source_precheck_sha256':sha(args.precheck)}
 (out/'BATCH-PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');shutil.copyfile(args.truth,out/'FD1-REFERENCE-ADMISSION.json');shutil.copyfile(args.review,out/'FD1-REFERENCE-REVIEW.json')
 old=ROOT/'staging/candidate-lift01'
 for name in ('dependencies.py','DEPENDENCIES.json'):shutil.copyfile(old/name,out/name)
 (out/'REPAIR.json').write_text(json.dumps({'fresh_attempt':'training-materialize01','purpose':'released exact nine training field materializations; no solves or training'})+'\n')
 script=(old/'native.sh').read_text().replace('ctol_nsobsrom_candlift','ctol_nsobsrom_materialize').replace('00:04:00','00:07:30').replace('candidate-lift01','training-materialize01')
 script=script.replace('12*2**30','18*2**30').replace('ten-GiB reserve plus two-GiB exact control staging envelope','ten-GiB reserve plus eight-GiB workspace/archive envelope')
 script=script.replace('results/control/RESULT.json','results/BATCH-RESULT.json').replace("'interface_passed'","'materialization_completed'")
 script=script.replace('"$PY" candidate_lift_control.py --plan CASES.json --release CONTROL-RELEASE.json --out results/control','"$PY" materialize_batch2d.py --plan BATCH-PLAN.json --release BATCH-RELEASE.json --out results')
 (out/'native.sh').write_text(script)
 sources=['materialize_batch2d.py','materialize_reference2d.py','qualification.py','FD1-SCOPE-CONTRACT.json','MODEL-CONTRACT.json','fe_adapter.py','primary_native/SOURCE.json']
 sources+=['primary_native/'+name for name in json.loads((ROOT/'primary_native/SOURCE.json').read_text())['files']]
 runtime=[{'source':name,'destination':name,'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size} for name in sources]
 for name in ('BATCH-PLAN.json','FD1-REFERENCE-ADMISSION.json','FD1-REFERENCE-REVIEW.json','native.sh','dependencies.py','DEPENDENCIES.json','REPAIR.json'):
  p=out/name;runtime.append({'source':str(p.relative_to(ROOT)),'destination':name,'sha256':sha(p),'bytes':p.stat().st_size})
 inventory={'schema':'ns-obstacle-nine-training-materialization-inventory-v1','runtime':runtime,'producer':rows,'producer_git_commit':args.reference_source,'producer_mirror_commit':args.reference_mirror,
  'producer_prefix':'experiments/ns-obstacle-2d','source_inputs_sha256':sha(ROOT/'reference_inputs/source_inputs.py'),'archive_recovery_sha256':sha(ROOT/'reference_inputs/git_archive_recovery.py'),
  'staging_source_sha256':sha(ROOT/'stage_materialization_packet.py'),'verify_warm_archives':True,'conservative_reservations_unchanged':True,'staging_performed':False}
 (out/'INPUT-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
 print(json.dumps({'source_sha256':batch.source_digest(),'plan_sha256':sha(out/'BATCH-PLAN.json'),'inventory_sha256':sha(out/'INPUT-INVENTORY.json'),'runtime_files':len(runtime),'producer_files':len(rows),'resources':resources},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('truth','review','precheck','reference-root','out'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('truth-sha256','reference-source','reference-mirror'):p.add_argument('--'+name,required=True)
 build(p.parse_args())
