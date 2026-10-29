from pathlib import Path
import json,hashlib,subprocess,shutil,time
B=Path(__file__).resolve().parents[1];repo=B.parents[1];a=B/'artifacts/development-materialize02'
root=repo.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/development-materialize02/WORKING-CHUNK-COOLING-RELEASE.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*v):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),*v],text=True).strip()
assert sha(root)=='9d3b983984a6e2ede1e9103192769fae481cd50aac29c809acf258478b7974c3'
r=json.loads(root.read_text());p=json.loads((a/'INDEPENDENT-COLD-PROPOSAL.json').read_text());c=json.loads((a/'GIT-COLD-ARCHIVE.json').read_text())
assert sha(a/'INDEPENDENT-COLD-PROPOSAL.json')==r['proposal_sha256'] and p['proof_passed'] and len(p['exact_proposed_paths'])==r['authorized_files']==11
assert sha(a/'RETENTION.json')==p['retention_sha256'] and c['archive_sha256']==r['archive_sha256']==p['archive_sha256']
tip=git('ls-remote','--exit-code','origin',p['advertised_mirror_ref']).split()[0]
git('merge-base','--is-ancestor',p['source_commit'],'HEAD');git('merge-base','--is-ancestor',p['mirror_commit'],tip);git('merge-base','--is-ancestor',c['mirror_commit'],tip)
paths=[]
for row in p['exact_proposed_paths']:
 f=a/row['path'];assert str(f)==row['absolute_path'] and not f.is_symlink() and f.is_file();s=f.stat();assert s.st_nlink==row['link_count']==1 and s.st_size==row['bytes'] and sha(f)==row['sha256']
 for commit,blob in row['git_blobs'].items():assert git('rev-parse',commit+':'+str(f.relative_to(repo)))==blob
 paths.append(f)
assert set((a/'archive-chunks').iterdir())==set(paths)
life=a/'CHUNK-COOLING-LIFECYCLE.json';assert not life.exists();free=shutil.disk_usage(a).free;allocated=sum(f.stat().st_blocks*512 for f in paths);assert allocated==r['maximum_allocated_credit_bytes']
record={'schema':'ns-obstacle-exact-chunk-cooling-lifecycle-v1','status':'verified_before_retirement','release_sha256':sha(root),'proposal_sha256':sha(a/'INDEPENDENT-COLD-PROPOSAL.json'),'cold_proof_sha256':sha(a/'GIT-COLD-ARCHIVE.json'),'original_retention_sha256':sha(a/'RETENTION.json'),'advertised_tip':tip,'paths':[str(f.relative_to(a)) for f in paths],'local_free_before_bytes':free,'allocated_duplicate_bytes':allocated,'unique_data_removed':False}
life.write_text(json.dumps(record,indent=2)+'\n')
for f in paths:f.unlink()
assert all(not f.exists() for f in paths) and sha(a/'RETENTION.json')==p['retention_sha256'];after=shutil.disk_usage(a).free
record.update(status='complete_exact_duplicates_absent',local_free_after_bytes=after,observed_free_space_change_bytes=after-free,chunk_storage='git_history_only',completed_epoch=time.time());life.write_text(json.dumps(record,indent=2)+'\n');shutil.copyfile(root,a/'CHUNK-COOLING-RELEASE.json');print(json.dumps(record,indent=2))
