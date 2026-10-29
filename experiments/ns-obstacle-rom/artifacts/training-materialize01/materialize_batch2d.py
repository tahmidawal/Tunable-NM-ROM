"""One released 450-second allocation, nine fresh per-case interpreters; no admission."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,math
import materialize_reference2d as worker
ROOT=Path(__file__).parent
TRAIN=[15+1.25*j for j in range(9)]
def sha(p):return worker.sha(p)
def source_digest():
 return hashlib.sha256(json.dumps({'entry':sha(__file__),'worker':worker.source_digest()},sort_keys=True).encode()).hexdigest()
def released_cases(plan,release,plan_path,inventory_path,truth):
 if (release.get('schema')!='ns-obstacle-nine-training-materialization-batch-release-v1' or release.get('source_sha256')!=source_digest()
     or release.get('plan_sha256')!=sha(plan_path) or release.get('input_inventory_sha256')!=sha(inventory_path)
     or release.get('reference_admission_sha256')!=sha(truth) or release.get('allocation_seconds')!=450
     or release.get('remaining_fine_preparation_seconds')!=1350 or release.get('single_execution') is not True
     or release.get('authorize_exact_nine_child_releases') is not True):raise ValueError('exact reviewed nine-case allocation release required')
 if (plan.get('schema')!='ns-obstacle-nine-training-materialization-plan-v1' or plan.get('source_sha256')!=source_digest()
     or [r['Re'] for r in plan.get('cases',[])]!=TRAIN or plan['reference_admission_sha256']!=sha(truth)
     or plan['allocation_seconds']!=450 or plan['finalization_reserve_seconds']!=30):raise ValueError('fixed genuine nine-case plan differs')
 if len({r['id'] for r in plan['cases']})!=9:raise ValueError('unique case IDs required')
 for r in plan['cases']:
  c=r['release_candidate']
  if (c['Re']!=r['Re'] or c['reference_admission_sha256']!=sha(truth) or c['job_directory']!=plan['job_directory']
      or c['output_directory']!=str(Path(plan['job_directory'])/'results'/r['id'])
      or r['original_directory']!='original/'+r['id'] or '/' in r['id'] or '..' in r['id']):raise ValueError('case identity/path binding differs')
 return plan['cases']
def remaining(start,now):
 if not all(math.isfinite(v) for v in (start,now)) or now<start:raise ValueError('valid cumulative allocation clock required')
 return 450-(now-start)-30

def launch(item,truth,child_path,timeout,stdout,stderr):
 return subprocess.run([sys.executable,str(ROOT/'materialize_reference2d.py'),'--original',item['original_directory'],'--truth',str(truth),'--release',str(child_path),'--out',item['release_candidate']['output_directory']],stdout=stdout,stderr=stderr,timeout=timeout)

def run(plan_path,release_path,out):
 plan_path=Path(plan_path);release_path=Path(release_path);plan=json.loads(plan_path.read_text());release=json.loads(release_path.read_text())
 truth=Path(plan['reference_admission_path']);inventory=Path('INPUT-INVENTORY.json')
 cases=released_cases(plan,release,plan_path,inventory,truth)
 if (sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID')
     or str(Path.cwd())!=release.get('job_directory') or str(Path.cwd())!=plan['job_directory']
     or os.environ.get('JAX_ENABLE_X64')!='true' or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest'):
  raise RuntimeError('released fresh cluster GPU allocation required')
 start=float(os.environ['ROM_ALLOCATION_START_UNIX_SECONDS']);out=Path(out)
 if out.resolve()!=Path(plan['job_directory'],'results'):raise ValueError('fixed result directory required')
 out.mkdir(exist_ok=True)
 with Path('BATCH-CLAIM.json').open('x') as f:json.dump({'release_sha256':sha(release_path),'job_id':os.environ['SLURM_JOB_ID']},f)
 record={'schema':'ns-obstacle-nine-training-materialization-result-v1','passed':False,'physical_admission_created':False,
  'source_sha256':source_digest(),'plan_sha256':sha(plan_path),'release_sha256':sha(release_path),'reference_admission_sha256':sha(truth),
  'allocation_seconds':450,'fine_preparation_remaining_seconds':1350,'records':[]}
 try:
  for item in cases:
   timeout=remaining(start,time.time())
   if timeout<=0:raise TimeoutError('cumulative allocation finalization reserve reached; no budget reset')
   child=dict(item['release_candidate']);assert child.pop('schema')=='ns-obstacle-fd1-primary-materialization-release-candidate-v1'
   child.update(schema='ns-obstacle-fd1-primary-materialization-release-v1',authorized_stage='training_primary_materialization',parent_batch_release_sha256=sha(release_path))
   child_path=out/(item['id']+'-RELEASE.json');child_path.write_text(json.dumps(child,indent=2)+'\n')
   worker.verify(item['original_directory'],truth,child_path)
   timeout=remaining(start,time.time())
   if timeout<=0:raise TimeoutError('verification consumed remaining allocation budget')
   entry={'Re':item['Re'],'release_sha256':sha(child_path),'started_allocation_seconds':time.time()-start,'status':'started'};record['records'].append(entry)
   with (out/(item['id']+'.stdout')).open('wb') as stdout,(out/(item['id']+'.stderr')).open('wb') as stderr:
    try:
     proc=launch(item,truth,child_path,timeout,stdout,stderr)
     entry.update(returncode=proc.returncode,status='process_completed' if proc.returncode==0 else 'failed')
     if proc.returncode:raise RuntimeError('materialization child failed; raw child output retained')
    except BaseException as exc:entry.update(status='failed',exception=repr(exc));raise
    finally:
     entry['finished_allocation_seconds']=time.time()-start
     record['allocation_elapsed_seconds']=time.time()-start;(out/'BATCH-RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
   execution=Path(child['output_directory'])/'MATERIALIZATION-EXECUTION.json';entry['execution_sha256']=sha(execution)
   if json.loads(execution.read_text()).get('status')!='exact_reconstruction_completed_pending_independent_review':raise ValueError('child did not complete exact reconstruction')
   entry['status']='completed'
  record['passed']=True
 except BaseException as exc:record['exception']=repr(exc);raise
 finally:
  record['allocation_elapsed_seconds']=time.time()-start;record['remaining_cases']=[r['Re'] for r in cases if r['Re'] not in {x['Re'] for x in record['records'] if x['status']=='completed'}]
  (out/'BATCH-RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
 return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.plan,a.release,a.out)
