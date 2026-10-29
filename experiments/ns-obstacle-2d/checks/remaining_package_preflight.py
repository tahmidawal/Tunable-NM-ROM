"""Verify all14 frozen bytes and exercise actual producer reads on one exact package."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile
B=Path(__file__).resolve().parents[1];PY='/home/tahmid/Dev/.venv/bin/python'
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
records=[]
for path in sorted((B/'g2/population-rest01').glob('*/SOURCE-CANDIDATE.json')):
 data=json.loads(path.read_text());seen=set()
 for row in data['files']:
  p=B/row['source'];assert p.stat().st_size==row['bytes']and sha(p)==row['sha256'];assert row['destination']not in seen;seen.add(row['destination'])
 subprocess.run(['bash','-n',str(path.with_name('run.sh'))],check=True)
 records.append({'group':path.parent.name,'candidate_sha256':sha(path),'inputs':len(data['files']),'all_runtime_bytes_verified':True})
path=B/'g2/population-rest01/re15/SOURCE-CANDIDATE.json';data=json.loads(path.read_text())
with tempfile.TemporaryDirectory(dir=B/'staging',prefix='remaining-metadata-')as temp:
 root=Path(temp)
 for row in data['files']:
  p=root/row['destination'];p.parent.mkdir(parents=True,exist_ok=True);os.link(B/row['source'],p)
 receipt=B/'staging/REMAINING-ACTUAL-PRODUCER-READ-PREFLIGHT.json'
 result=subprocess.run([PY,'-B',str(root/'g2/validate_population_inputs.py'),'--root',str(root),'--plan',str(root/'g2/population01/PLAN.json'),'--group','re15','--out',str(receipt)],capture_output=True,text=True)
 if result.returncode:raise RuntimeError(result.stdout+result.stderr)
out={'schema':'ns-obstacle-remaining14-exact-input-preflight-v1','groups':records,'representative_actual_reader_receipt_sha256':sha(receipt),'native_mesh_math_not_executed':True}
(B/'staging/REMAINING-EXACT-PACKAGE-PREFLIGHT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'groups':len(records),'inputs_each':records[0]['inputs'],'actual_reader':json.loads(receipt.read_text())}))
