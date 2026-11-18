"""Selective checksum-verified retention; explicit remote-only derived arrays."""
from pathlib import Path
import subprocess as s,json,hashlib,sys
H=Path(__file__).resolve().parent;NS='/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117'
ledger=json.loads((H/'cluster/ledger.json').read_text())
for job in sys.argv[1:]:
 row=next(r for r in ledger if r['name']==job);assert row.get('state')=='COMPLETED'
 root=NS+'/'+job;o=H/'results'/job;o.mkdir(parents=True,exist_ok=True)
 manifest=s.check_output(['ssh','tufts-login',f'cat {root}/OUTPUTS.sha256'],text=True);(o/'OUTPUTS.sha256').write_text(manifest)
 records=[]
 for line in manifest.splitlines():
  digest,name=line.split('  ',1);p=Path(name);assert not p.is_absolute() and '..' not in p.parts
  full=job in ('c1data','c1familyb','c1ref32')
  local=full or p.suffix in ('.json','.pkl') or (job=='c1cost' and ('_block' in p.name or p.name=='final4096_case9.npz')) or (job=='c1fast' and '_q' in p.name) or (job=='c1cubature' and '_cap' in p.name)
  if local:
   target=o/p;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:s.run(['scp','-q',f'tufts-login:{root}/{name}',str(target)],check=True)
   assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
  records.append(dict(relative=name,sha256=digest,local_path=str(o/p) if local else None,remote_path=root+'/'+name,retention='local+remote verified' if local else 'remote raw retained; not downloaded'))
 s.run(['scp','-qr',f'tufts-login:{root}/logs',str(o)],check=True)
 (H/'evidence'/f'{job}-retention.json').write_text(json.dumps(dict(job=job,records=records),indent=2)+'\n')
 print(job,len(records),'verified selected outputs',flush=True)
