"""Remove only independently audited, Git-backed remote population duplicates.

No numerical work, retry, local cooling, or mutation of immutable retention records.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess
B=Path(__file__).resolve().parents[1]
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login']
REMOTE_PY='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def require(value,message):
 if not value:raise ValueError(message)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--group',required=True);args=parser.parse_args()
 group=next(g for g in read(B/'g2/population-rest01/SCHEDULE.json')['groups']if g['group']==args.group)
 frozen=B/'g2/population01/frozen'/args.group;candidate=frozen/'QUALIFICATION-CANDIDATE.json';review=frozen/'INDEPENDENT-ROW-REVIEW.json';audit=read(review)
 require(audit['passed'] is True and audit['candidate_sha256']==sha(candidate) and audit['Re']==group['Re'] and audit['split']==group['split'],'independent row review differs')
 artifact=B/'artifacts'/group['attempt'];retention=read(artifact/'RETENTION.json');cold_path=artifact/'GIT-COLD-ARCHIVE.json';cold=read(cold_path);mapping=artifact/'DELTA-RETENTION-MAP.json';logical=read(mapping)
 require(audit['archive_sha256']==retention['archive_sha256'],'reviewed archive differs')
 require(cold['source_retention_sha256']==sha(artifact/'RETENTION.json') and cold['archive_sha256']==retention['archive_sha256'] and cold['git_only_stream_verification']['archive_sha256']==retention['archive_sha256'],'cold archive binding differs')
 require(sha(mapping)==retention['retention_map_sha256'] and not logical['symlinks'],'map differs or unsupported symbolic links')
 subprocess.run(['git','-c','gc.auto=0','merge-base','--is-ancestor',cold['source_commit'],'HEAD'],check=True)
 advertised=subprocess.check_output(['git','-c','gc.auto=0','ls-remote','origin',cold['mirror_ref']],text=True).split();require(len(advertised)==2 and advertised[1]==cold['mirror_ref'],'advertised mirror absent');subprocess.run(['git','-c','gc.auto=0','merge-base','--is-ancestor',cold['mirror_commit'],advertised[0]],check=True)
 job=str(retention['job_id']);require(job.isdigit(),'invalid job id');accounting=subprocess.run(SSH+['sacct -j '+job+' --noheader --parsable2 --format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,check=True);require(any(line==job+'|COMPLETED|0:0' for line in accounting.stdout.splitlines()),'job is not completed')
 receipt=artifact/'REMOTE-CLEANUP-LIFECYCLE.json';require(not receipt.exists(),'cleanup already attempted; inspect authoritative state before any retry')
 directory='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/'+group['attempt']
 expected={name:{'bytes':row['bytes'],'sha256':row['sha256']}for name,row in logical['files'].items()}
 payload={'directory':directory,'files':expected,'archive_sha256':retention['archive_sha256'],'archive_bytes':retention['archive_bytes']}
 program='from pathlib import Path\nimport json,hashlib,shutil\npayload=json.loads('+repr(json.dumps(payload))+')\n'+'''root=Path(payload['directory'])
archive=Path(str(root)+'.delta.tar.gz');written=Path(str(root)+'.DELTA-WRITE.json')
def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
if not root.is_dir() or not archive.is_file() or not written.is_file():raise ValueError('expected remote originals absent')
actual={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file()}
if any(p.is_symlink() for p in root.rglob('*')) or set(actual)!=set(payload['files']):raise ValueError('remote file inventory differs')
for name,row in payload['files'].items():
 p=actual[name]
 if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('remote bytes differ:'+name)
if archive.stat().st_size!=payload['archive_bytes'] or digest(archive)!=payload['archive_sha256'] or json.loads(written.read_text())['archive_sha256']!=payload['archive_sha256']:raise ValueError('remote delta differs')
if Path(str(archive)+'.partial').exists():raise ValueError('unexpected partial archive')
removed=sum(p.stat().st_size for p in actual.values())+archive.stat().st_size+written.stat().st_size
shutil.rmtree(root);archive.unlink();written.unlink()
print(json.dumps({'status':'all_original_and_delta_hashes_verified_then_removed','regular_files':len(actual),'removed_bytes':removed,'archive_sha256':payload['archive_sha256']}))
'''
 record={'schema':'ns-obstacle-population-remote-duplicate-cleanup-v1','requested_utc':datetime.datetime.now(datetime.UTC).isoformat(),'group':args.group,'job_id':job,'candidate_sha256':sha(candidate),'row_review_sha256':sha(review),'cold_manifest_sha256':sha(cold_path),'advertised_mirror_tip':advertised[0],'helper_sha256':sha(Path(__file__)),'accounting':accounting.stdout,'status':'submitted_unknown_until_receipt','immutable_retention_unchanged':True}
 receipt.write_text(json.dumps(record,indent=2)+'\n')
 result=subprocess.run(SSH+[REMOTE_PY+' -'],input=program,capture_output=True,text=True);record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,status='verified_removed'if result.returncode==0 else'uncertain_inspect_before_any_retry');receipt.write_text(json.dumps(record,indent=2)+'\n')
 if result.returncode:raise RuntimeError('remote cleanup uncertain; no automatic retry')
 print(result.stdout,end='')
if __name__=='__main__':main()
