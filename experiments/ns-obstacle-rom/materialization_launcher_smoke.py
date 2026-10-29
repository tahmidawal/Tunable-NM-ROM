"""Subsecond process-group deadline checks; no numerical or native computation."""
from pathlib import Path
import json,subprocess,tempfile,time
import materialize_batch2d as batch
ROOT=Path(__file__).parent

def main():
 packet=ROOT/'staging/training-materialize01';actual=(packet/'native.sh').read_text();body=(packet/'native-body.sh').read_text()
 assert 'exec /usr/bin/timeout --signal=TERM --kill-after=10s 435s /usr/bin/bash ./native-body.sh' in actual
 assert 'python' not in actual and 435+10<450
 assert body.index('trap finish EXIT')<body.index('export ROM_ALLOCATION_START_UNIX_SECONDS=')<body.index('JOB_PHASE=gpu_preflight')
 records=[]
 with tempfile.TemporaryDirectory() as directory:
  for mode in ('term','kill'):
   p=Path(directory)/mode;p.mkdir();(p/'native.sh').write_text(actual.replace('10s','0.2s').replace('435s','0.2s'))
   trap="trap 'echo finalized > finalized; exit 143' TERM" if mode=='term' else "trap '' TERM"
   (p/'native-body.sh').write_text('#!/usr/bin/env bash\n'+trap+"\necho partial-setup\nsleep 20 &\necho $! > descendant.pid\nwait $!\n")
   started=time.monotonic()
   with (p/'stdout').open('wb') as out,(p/'stderr').open('wb') as err:result=subprocess.run(['/usr/bin/bash','native.sh'],cwd=p,stdout=out,stderr=err,timeout=3)
   elapsed=time.monotonic()-started;assert elapsed<2 and result.returncode in (124,137,-9)
   pid=int((p/'descendant.pid').read_text());proc=Path('/proc')/str(pid)/'stat';state=proc.read_text().split()[2] if proc.exists() else 'absent';assert state in ('Z','absent')
   assert b'partial-setup' in (p/'stdout').read_bytes()
   if mode=='term':assert (p/'finalized').read_text().strip()=='finalized'
   records.append({'case':mode,'returncode':result.returncode,'elapsed_seconds':elapsed,'descendant_state':state,'partial_stdout_retained':True,'finalization_marker':(p/'finalized').exists()})
 record={'passed':True,'native_sh_sha256':batch.sha(packet/'native.sh'),'native_body_sha256':batch.sha(packet/'native-body.sh'),'source_plan_sha256':batch.sha(packet/'BATCH-PLAN.json'),
  'real_outer_term_seconds':435,'real_kill_grace_seconds':10,'real_ceiling_seconds':445,'allocation_reservation_seconds':450,'test_duration_substitutions_only':True,'numerical_or_native_work':False,'records':records}
 (ROOT/'checks/materialization-launcher-smoke.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
