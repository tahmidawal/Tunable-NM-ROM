from pathlib import Path
import hashlib,json,subprocess,stat
O=Path(__file__).resolve().parent;W=O.parents[3]
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(W),*a])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=git('rev-parse','HEAD').decode().strip();mirror=git('ls-remote','origin','refs/heads/codeonly/exp/2026-10-27-ns-obstacle-2d').decode().split()[0]
canonical=W/'experiments/ns-obstacle-2d/artifacts/g2-p4trend/results/dfg_h3/fe_layout.json';ch=sha(canonical);assert ch=='ea8c51f2e29fac8bb1b98e8d6fd2e8d4b975e605d3e6c356e8861074e4d1b249';rows=[]
for rp in ['experiments/ns-obstacle-2d/staging/g2-p4trend/preliminary/fe_layout.json','experiments/ns-obstacle-2d/staging/g2-solverquad/payload/inputs/p4_g4_h3/fe_layout.json']:
 p=W/rp;s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_nlink==1;assert sha(p)==ch;blobs={}
 for c in [source,mirror]:
  blob=git('rev-parse',c+':'+rp).decode().strip();raw=git('cat-file','blob',blob);assert len(raw)==s.st_size and hashlib.sha256(raw).hexdigest()==ch;blobs[c]=blob
  assert git('show',c+':'+str(canonical.relative_to(W)))==canonical.read_bytes()
 assert len(set(blobs.values()))==1
 rows.append(dict(path=str(p),repository_path=rp,sha256=ch,bytes=s.st_size,allocated_bytes=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,link_count=s.st_nlink,git_blobs=blobs))
s=canonical.stat();j=dict(schema='ns-obstacle-obsolete-staging-layout-cold-proposal-v1',proof_passed=True,deletion_authorized=False,source_commit=source,advertised_mirror_commit=mirror,exact_paths=rows,allocated_bytes=sum(r['allocated_bytes'] for r in rows),canonical_preserved=dict(path=str(canonical),sha256=ch,bytes=s.st_size,inode=s.st_ino,device=s.st_dev),scope='Only two explicitly authorized obsolete staging metadata copies, not native mesh/solution/rule/scientific arrays. Canonical fine layout and all other files preserved.',recovery='Original exact source and advertised mirror Git blobs retained for each path; restore by pinned commit/path or blob and require size/SHA. Canonical warm copy is independently identical.',reader_clearance='Final no-reader confirmations required before root exact cleanup release.',deletions_performed=False)
(O/'PROPOSAL.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({'sha256':sha(O/'PROPOSAL.json'),'allocated_bytes':j['allocated_bytes'],'source':source,'mirror':mirror}))
