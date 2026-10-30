# Prospective namespace template; bind/review actual terminal job before use.
EXPECTED_JOB_ID=6660183
assert type(EXPECTED_JOB_ID) is int and EXPECTED_JOB_ID>0, 'actual reviewed diagnostic08 job is not bound'
from pathlib import Path
import subprocess,json
remote=r'''
from pathlib import Path
import subprocess,hashlib,json
base=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-diagnostic08');archive=base.with_suffix('.tar.gz')
assert not archive.exists() and not (base/'COLLECTION-SHA256SUMS').exists()
status=subprocess.check_output(['sacct','-j','__JOB_ID__','-n','-P','--format=JobID,State,ElapsedRaw,ExitCode'],text=True);assert any(x.split('|')[0]=='__JOB_ID__' and x.split('|')[1]=='COMPLETED' and x.split('|')[3]=='0:0' for x in status.splitlines() if x.strip())
assert '__JOB_ID__' not in subprocess.check_output(['squeue','-u','tawal01','-h','-o','%A'],text=True).split()
roots=[base.parent,Path('/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027'),Path('/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027')]
usage={str(p):int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in roots};assert sum(usage.values())+15*2**30+256*2**20<50*2**30
files=sorted(p for p in base.rglob('*') if p.is_file());assert files and not any(p.is_symlink() for p in base.rglob('*'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ledger=base/'COLLECTION-SHA256SUMS';ledger.write_text(''.join(sha(p)+'  '+str(p.relative_to(base))+'\n' for p in files));files.append(ledger)
subprocess.run(['tar','--hard-dereference','-czf',str(archive),'--null','-T','-'],cwd=base.parent,input=b'\0'.join(str(p.relative_to(base.parent)).encode() for p in files)+b'\0',check=True)
print(json.dumps({'job_id':__JOB_ID__,'terminal_status':status,'archive_path':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'regular_files':len(files),'expanded_bytes':sum(p.stat().st_size for p in files),'ledger_sha256':sha(ledger),'all_originals_preserved':True,'diagnostic_only':True,'result_admitted':False,'prepack_namespaces':usage}))
'''
remote=remote.replace('__JOB_ID__',str(EXPECTED_JOB_ID))
r=subprocess.run(['ssh','tufts-login','/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python','-'],input=remote,text=True,capture_output=True);Path(__file__).with_suffix('.last-stdout.txt').write_text(r.stdout);Path(__file__).with_suffix('.last-stderr.txt').write_text(r.stderr);r.check_returncode();record=json.loads(r.stdout);p=Path(__file__).resolve().parent/'neural-pod-guided-diagnostic08-archive-measurement.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
