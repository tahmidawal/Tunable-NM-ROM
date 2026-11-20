"""Verify the code-only remote blobs before deleting precisely owned job directories."""
from pathlib import Path
import subprocess,hashlib,json,datetime
L=Path(__file__).resolve().parents[1];repo=L.parents[1];prefix=str(L.relative_to(repo));NS='/cluster/tufts/paralab/tawal01/jcp-width-deploy-20261119'
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0',*a],cwd=repo)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def tree(ref):
 d={}
 for line in git('ls-tree','-r','-z',ref,'--',prefix).split(b'\0'):
  if line:
   meta,path=line.decode().split('\t',1);d[path]=meta.split()[2]
 return d
branch=git('rev-parse','--abbrev-ref','HEAD').decode().strip();head=git('rev-parse','HEAD').decode().strip();remote_ref='refs/heads/codeonly/'+branch
remote=git('ls-remote','origin',remote_ref).decode().split()[0];localtree=tree(head);remotetree=tree(remote)
ledger=json.loads((L/'runs/LEDGER.json').read_text());records=[]
assert len(ledger['attempts'])==3
queued=subprocess.check_output(['ssh','tufts-login','squeue -u tawal01 -h -o "%i|%j|%T"'],text=True)
ownedids={str(x['job_id']) for x in ledger['attempts']};assert not any(x.split('|')[0] in ownedids for x in queued.splitlines())
for record in ledger['attempts']:
 a=record['attempt'];folder=L/'runs'/a;collect=json.loads((folder/'collection.json').read_text());assert collect['scientifically_qualified']
 files=[]
 for line in (folder/'RESULTS.sha256').read_text().splitlines():
  expected,rel=line.split('  ',1);p=folder/rel;assert sha(p)==expected
  path=str(p.relative_to(repo));working_blob=git('hash-object','--',str(p)).decode().strip()
  assert localtree.get(path)==working_blob and working_blob==remotetree.get(path),path
  files.append({'path':rel,'sha256':expected,'verified_remote_blob':remotetree[path]})
 records.append(dict(attempt=a,job_id=str(record['job_id']),allocated_gpu_seconds=record['allocated_gpu_seconds'],scientific_files_verified=len(files),files=files,remote_job_directory=f'{NS}/{a}'))
# Recheck exact owned names before deleting only these checksum-verified duplicates.
for r in records:
 remote_path=r['remote_job_directory'];a=r['attempt'];assert remote_path in [f'{NS}/f128',f'{NS}/f512',f'{NS}/b01']
 command=f'test -f {remote_path}/RESULTS.sha256 && test -f {remote_path}/code/OWNER.json && rm -rf -- {remote_path} && test ! -e {remote_path}'
 subprocess.check_call(['ssh','tufts-login',command]);r['remote_duplicate_removed']=True
subprocess.check_call(['ssh','tufts-login',f'rmdir -- {NS} && test ! -e {NS}'])
summary=dict(status='complete; all registered work terminal and sparse continuation gate closed',local_source_commit=head,verified_remote_commit=remote,remote_ref=remote_ref,generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_removed=True,total_allocated_gpu_seconds=sum(x['allocated_gpu_seconds'] for x in records),ceiling_allocated_gpu_seconds=7200,all_scientific_files_backed_up=True,worktree_unmerged=True,canonical_lab_log_owner='parent /root',jobs=records)
assert summary['total_allocated_gpu_seconds']==ledger['total_allocated_gpu_seconds']<=7200
(L/'checks/CLOSEOUT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='jobs'}))
