from pathlib import Path
import hashlib,json,subprocess,os,stat,datetime,numpy as np
O=Path(__file__).resolve().parent;W=O.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(W),*args])
rp=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/SUPERSEDED08-PLOT-INPUT-COOLING-RELEASE.json');rh='b08693790ee1d94629d5071fdd3bf67afd3638bbe862ee0703587d5e78a4e19e';assert sha(rp)==rh
r=json.loads(rp.read_text());p=json.loads((O/'PROPOSAL.json').read_text());assert r['authorized'] and sha(O/'PROPOSAL.json')==r['proposal_sha256'];assert r['exact_paths']==p['exact_paths'] and len(r['exact_paths'])==1
out=O/'LIFECYCLE.json';assert not out.exists()
ref=p['advertised_mirror_ref'];tip=git('ls-remote','origin',ref).decode().split()[0]
base=Path(p['latest_preserved_dataset']['path']);assert sha(base)==p['latest_preserved_dataset']['sha256']
with np.load(base,allow_pickle=False) as z:latest={k:z[k] for k in z.files if k!='metadata_json'}
for row in r['exact_paths']:
 f=Path(row['path']);s=f.lstat();assert stat.S_ISREG(s.st_mode) and not f.is_symlink();assert (s.st_ino,s.st_dev,s.st_size,s.st_nlink)==(row['inode'],row['device'],row['bytes'],row['link_count']);assert sha(f)==row['sha256']
 for commit in [p['source_commit'],p['mirror_commit'],tip]:
  blob=git('rev-parse',commit+':'+row['repository_path']).decode().strip();assert blob==row['git_blob'];raw=git('cat-file','blob',blob);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 metadata=Path(row['retained_metadata']['path']);assert sha(metadata)==row['retained_metadata']['sha256']
 with np.load(f,allow_pickle=False) as z:
  for k in z.files:
   if k!='metadata_json':assert z[k].dtype==latest[k].dtype and z[k].shape==latest[k].shape and z[k].tobytes()==latest[k].tobytes()
  md=z['metadata_json'];assert json.loads(md.tobytes() if md.dtype==np.uint8 else str(md))==json.loads(metadata.read_text())
 # Reject open file handles visible to this user before unlink.
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:assert fd.resolve()!=f,('open_reader',str(fd))
    except (FileNotFoundError,PermissionError):pass
  except (FileNotFoundError,PermissionError):pass
before=os.statvfs(W);before=before.f_bavail*before.f_frsize
removed=[]
for row in r['exact_paths']:
 f=Path(row['path']);s=f.lstat();assert (s.st_ino,s.st_dev,s.st_size)==(row['inode'],row['device'],row['bytes']) and sha(f)==row['sha256'];f.unlink();removed.append(row['repository_path'])
after=os.statvfs(W);after=after.f_bavail*after.f_frsize;assert sha(base)==p['latest_preserved_dataset']['sha256']
j=dict(schema='ns-obstacle-exact-superseded-plot-cooling-lifecycle-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),release_sha256=rh,proposal_sha256=r['proposal_sha256'],fresh_advertised_mirror=tip,source_commit=p['source_commit'],historical_mirror=p['mirror_commit'],removed_paths=removed,allocated_bytes=r['allocated_bytes'],free_before=before,free_after=after,observed_gain=after-before,latest_preserved_sha256=sha(base),all_original_metadata_preserved=True,reader_clearance='Root/ROM/numerical clearance unchanged; fresh visible /proc file-handle check passed. Complete original containers retained in pinned source and mirror.',recovery='PROPOSAL.json retains exact Git blob, source/mirror commits, original size/hash and path for each container; restore from Git only after identity verification.')
out.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2))
