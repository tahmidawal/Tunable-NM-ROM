from pathlib import Path
import subprocess,json,datetime
p=Path(__file__).with_name('ledger.json');rows=json.loads(p.read_text())
ids=','.join(str(r['id']) for r in rows)
out=subprocess.check_output(['ssh','tufts-login',f'sacct -j {ids} -X --format=JobID,State,ElapsedRaw,AllocCPUS -P'],text=True)
for line in out.splitlines()[1:]:
 z=line.split('|')
 for r in rows:
  if z[0]==str(r['id']):r.update(state=z[1],elapsed_seconds=int(z[2]),allocated_cpus=int(z[3]))
p.write_text(json.dumps(rows,indent=2)+'\n')
print(out);print('elapsed allocated GPU hours',sum(r.get('elapsed_seconds',0) for r in rows)/3600)
