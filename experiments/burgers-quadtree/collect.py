"""Collect only an owned completed job; preserve remote evidence until durable archival."""
from pathlib import Path
import subprocess,json,hashlib,sys,shutil
H=Path(__file__).resolve().parent;name=sys.argv[1];rows=json.loads((H/'cluster/ledger.json').read_text());row=next(r for r in rows if r['name']==name)
remote=row['remote'];assert remote.startswith('/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117/')
root=H/'results'/name;root.mkdir(parents=True,exist_ok=True)
for sub in ['output','logs']:
 subprocess.run(['rsync','-a',f'tufts-login:{remote}/{sub}/',str(root/sub)+'/'],check=True)
for file in ['OUTPUTS.sha256','MANIFEST.sha256','run.sbatch']:
 subprocess.run(['scp','-q',f'tufts-login:{remote}/{file}',str(root/file)],check=True)
subprocess.run(['sha256sum','-c','OUTPUTS.sha256','--quiet'],cwd=root,check=True)
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file():files.append(dict(path=str(p.relative_to(H)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(H/'evidence'/f'{name}-retention.json').write_text(json.dumps(dict(local=str(root),remote=remote,files=files),indent=2)+'\n')
for f in ['result.json','spectral.json']:
 if (root/'output'/f).exists():shutil.copyfile(root/'output'/f,H/'evidence'/f'{name}-{f}')
print(name,'verified',len(files),'files',sum(f['bytes'] for f in files),'bytes')
