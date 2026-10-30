from pathlib import Path
import subprocess,json
remote=r'''
from pathlib import Path
import subprocess,hashlib,json
base=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-galerkin01');archive=base.with_suffix('.tar.gz')
assert not archive.exists() and not (base/'COLLECTION-SHA256SUMS').exists()
status=subprocess.check_output(['sacct','-j','6659611','-n','-P','--format=JobID,State,ElapsedRaw,ExitCode'],text=True);assert any(x.split('|')[0]=='6659611' and x.split('|')[1]=='COMPLETED' and x.split('|')[3]=='0:0' for x in status.splitlines() if x.strip())
assert '6659611' not in subprocess.check_output(['squeue','-u','tawal01','-h','-o','%A'],text=True).split()
roots=[base.parent,Path('/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027'),Path('/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027')]
usage={str(p):int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in roots};assert sum(usage.values())+15*2**30+256*2**20<50*2**30
files=sorted(p for p in base.rglob('*') if p.is_file());assert files and not any(p.is_symlink() for p in base.rglob('*'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ledger=base/'COLLECTION-SHA256SUMS';ledger.write_text(''.join(sha(p)+'  '+str(p.relative_to(base))+'\n' for p in files));files.append(ledger)
subprocess.run(['tar','--hard-dereference','-czf',str(archive),'--null','-T','-'],cwd=base.parent,input=b'\0'.join(str(p.relative_to(base.parent)).encode() for p in files)+b'\0',check=True)
print(json.dumps({'job_id':6659611,'terminal_status':status,'archive_path':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'regular_files':len(files),'expanded_bytes':sum(p.stat().st_size for p in files),'ledger_sha256':sha(ledger),'all_originals_preserved':True,'diagnostic_only':True,'result_admitted':False,'prepack_namespaces':usage}))
'''
r=subprocess.run(['ssh','tufts-login','/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python','-'],input=remote,text=True,capture_output=True);Path(__file__).with_suffix('.last-stdout.txt').write_text(r.stdout);Path(__file__).with_suffix('.last-stderr.txt').write_text(r.stderr);r.check_returncode();record=json.loads(r.stdout);p=Path(__file__).resolve().parent/'neural-pod-guided-galerkin01-archive-measurement.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
