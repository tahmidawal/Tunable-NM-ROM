"""Selective streaming checksum retention, with explicit remote-only derived arrays."""
from pathlib import Path
import subprocess as s,json,hashlib,sys
H=Path(__file__).resolve().parent;NS='/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117'
ledger=json.loads((H/'cluster/ledger.json').read_text())
for job in sys.argv[1:]:
 assert job.isalnum()
 row=next(r for r in ledger if r['name']==job);assert row.get('state')=='COMPLETED'
 root=NS+'/'+job;o=H/'results'/job;o.mkdir(parents=True,exist_ok=True)
 manifest=s.check_output(['ssh','tufts-login',f'cat {root}/OUTPUTS.sha256'],text=True);(o/'OUTPUTS.sha256').write_text(manifest)
 records=[];missing=[]
 for line in manifest.splitlines():
  digest,name=line.split('  ',1);p=Path(name);assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='output'
  full=job in ('c1data','c1familyb','c1ref32','c1classical','c2frame','c3befficiency')
  local=full or p.suffix in ('.json','.pkl') or p.name in ('teacher.npz','normalization.npz','projection.npz') or (job=='c1manifold' and p.name=='lift.npz') or (job=='c1bbank' and p.name=='pod.npz') or (job=='c1cost' and ('_block' in p.name or p.name=='final4096_case9.npz')) or (job in ('c1fast','c1compress','c3globalcompress','c3globalcapacity') and '_q' in p.name) or (job=='c1cubature' and '_cap' in p.name)
  if local and (not (o/p).exists() or hashlib.sha256((o/p).read_bytes()).hexdigest()!=digest):missing.append(name)
  records.append(dict(relative=name,sha256=digest,local_path=str(o/p) if local else None,remote_path=root+'/'+name,retention='local+remote verified' if local else 'remote raw retained; not downloaded'))
 if missing:
  remote=s.Popen(['ssh','tufts-login',f'cd {root} && tar -cf - --null -T -'],stdin=s.PIPE,stdout=s.PIPE)
  extract=s.Popen(['tar','-xf','-','-C',str(o)],stdin=remote.stdout);remote.stdout.close();remote.stdin.write(('\0'.join(missing)+'\0').encode());remote.stdin.close()
  assert extract.wait()==0 and remote.wait()==0
 for record in records:
  if record['local_path']:assert hashlib.sha256(Path(record['local_path']).read_bytes()).hexdigest()==record['sha256']
 s.run(['scp','-qr',f'tufts-login:{root}/logs',str(o)],check=True)
 (H/'evidence'/f'{job}-retention.json').write_text(json.dumps(dict(job=job,records=records),indent=2)+'\n')
 print(job,len(records),'verified selected outputs',flush=True)
