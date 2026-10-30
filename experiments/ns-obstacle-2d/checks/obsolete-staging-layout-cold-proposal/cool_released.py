from pathlib import Path
import hashlib,json,subprocess,os,stat,datetime
O=Path(__file__).resolve().parent;W=O.parents[3]
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(W),*a])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rp=W.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/OBSOLETE-REFERENCE-STAGING-LAYOUT-COOLING-RELEASE.json';rh='378930d3052d51332629a4f459bc78242bbae65fff7b384b10c98640d6129bcb';assert sha(rp)==rh
r=json.loads(rp.read_text());p=json.loads((O/'PROPOSAL.json').read_text());assert r['authorized'] and r['proposal_sha256']==sha(O/'PROPOSAL.json');assert r['exact_paths']==p['exact_paths'] and len(r['exact_paths'])==2
out=O/'LIFECYCLE.json';assert not out.exists();tip=git('ls-remote','origin','refs/heads/codeonly/exp/2026-10-27-ns-obstacle-2d').decode().split()[0]
c=Path(p['canonical_preserved']['path']);cs=c.stat();assert sha(c)==p['canonical_preserved']['sha256'] and cs.st_ino==p['canonical_preserved']['inode'] and cs.st_dev==p['canonical_preserved']['device']
for row in r['exact_paths']:
 f=Path(row['path']);s=f.lstat();assert stat.S_ISREG(s.st_mode) and not f.is_symlink();assert (s.st_ino,s.st_dev,s.st_size,s.st_nlink)==(row['inode'],row['device'],row['bytes'],1);assert sha(f)==row['sha256']==sha(c)
 for commit in [p['source_commit'],p['advertised_mirror_commit'],tip]:
  data=git('show',commit+':'+row['repository_path']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:assert fd.resolve()!=f,('open_reader',str(fd))
    except (FileNotFoundError,PermissionError):pass
  except (FileNotFoundError,PermissionError):pass
v=os.statvfs(W);before=v.f_bavail*v.f_frsize;removed=[]
for row in r['exact_paths']:
 f=Path(row['path']);s=f.lstat();assert (s.st_ino,s.st_dev,s.st_size)==(row['inode'],row['device'],row['bytes']) and sha(f)==row['sha256'];f.unlink();removed.append(row['repository_path'])
v=os.statvfs(W);after=v.f_bavail*v.f_frsize;assert sha(c)==p['canonical_preserved']['sha256']
j=dict(schema='ns-obstacle-exact-staging-layout-cooling-lifecycle-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),release_sha256=rh,proposal_sha256=r['proposal_sha256'],source_commit=p['source_commit'],mirror_commit=p['advertised_mirror_commit'],fresh_advertised_mirror=tip,removed_paths=removed,allocated_bytes=p['allocated_bytes'],free_before=before,free_after=after,observed_gain=after-before,canonical_preserved=p['canonical_preserved'],no_reader_guard='Root/ROM/numerical exact-path clearances retained; fresh visible /proc handles checked.',recovery='Exact original pinned source/mirror blobs and SHA/size/path remain in PROPOSAL.json. Canonical warm copy remains byte-identical. No other removal.')
out.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2))
