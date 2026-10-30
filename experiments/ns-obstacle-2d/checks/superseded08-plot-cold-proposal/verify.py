"""Read-only exact direct-Git recovery and scientific-buffer containment proof."""
from pathlib import Path
import hashlib,json,subprocess,stat,time,numpy as np
O=Path(__file__).resolve().parent;W=O.parents[3];C=W/'experiments/ns-obstacle-2d/checks';start=time.monotonic()
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(W),*a])
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=git('rev-parse','HEAD').decode().strip();ref='refs/heads/codeonly/exp/2026-10-27-ns-obstacle-2d';mirror=git('ls-remote','origin',ref).decode().split()[0];git('cat-file','-e',mirror+'^{commit}')
base=C/'neural-scanned-coordinate-diagnostic-plot-data/plots-with-scanned_coordinate-neural.npz';assert sha(base)=='c23b561ff624b5188c55b387608948657f9697914ef76a1b7a8abce34723fe1c'
with np.load(base,allow_pickle=False) as z:latest={k:z[k] for k in z.files if k!='metadata_json'}
names=[('neural-output-polish-diagnostic-plot-data','plots-with-output_polish-neural.npz')]
rows=[]
for directory,name in names:
 p=C/directory/name;st=p.lstat();assert stat.S_ISREG(st.st_mode) and not p.is_symlink();rp=str(p.relative_to(W));h=sha(p);blobs={c:git('rev-parse',c+':'+rp).decode().strip() for c in (source,mirror)};assert len(set(blobs.values()))==1;blob=next(iter(blobs.values()));raw=git('cat-file','blob',blob);assert len(raw)==st.st_size and hashlib.sha256(raw).hexdigest()==h
 with np.load(p,allow_pickle=False) as z:
  arrays=[]
  for k in z.files:
   if k=='metadata_json':continue
   a=z[k];assert k in latest and a.dtype==latest[k].dtype and a.shape==latest[k].shape and a.tobytes()==latest[k].tobytes();arrays.append({'key':k,'shape':list(a.shape),'dtype':str(a.dtype),'buffer_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
  a=z['metadata_json'];md=json.loads(a.tobytes() if a.dtype==np.uint8 else str(a))
 metadata=p.parent/'METADATA.json';assert json.loads(metadata.read_text())==md
 for c in (source,mirror):assert git('show',c+':'+str(metadata.relative_to(W)))==metadata.read_bytes()
 rows.append({'path':str(p),'repository_path':rp,'sha256':h,'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'inode':st.st_ino,'device':st.st_dev,'link_count':st.st_nlink,'git_blob':blob,'source_mirror_blobs':blobs,'retained_metadata':{'path':str(metadata),'sha256':sha(metadata),'decoded_exact_match':True},'scientific_arrays_byte_identical_in_latest':arrays})
out={'schema':'ns-obstacle-superseded-plot-input-direct-history-cold-proposal-v1','proof_passed':True,'deletion_authorized':False,'deletions_performed':False,'source_commit':source,'mirror_commit':mirror,'advertised_mirror_ref':ref,'advertised_mirror_tip':mirror,'latest_preserved_dataset':{'path':str(base),'sha256':sha(base),'bytes':base.stat().st_size,'inode':base.stat().st_ino,'device':base.stat().st_dev},'exact_paths':rows,'logical_bytes':sum(r['bytes'] for r in rows),'conservative_allocated_reclaim_bytes':sum(r['allocated_bytes'] for r in rows if r['link_count']==1),'recovery':'Read exact pinned source or mirror Git blob, require original length and SHA before restoring exact original path. Every original NPZ container is retained in both histories, all scientific buffers also in latest dataset, decoded original metadata separately retained. No current-tip-only recovery or Git pruning.','dependencies':'Root explicitly authorized proof for superseded plot inputs; final no-reader/root exact release required. Latest09c23b561f preserves all prior actual predictions and supports future scanned-coordinate comparison. Old containers retain exact original metadata_json via pinned complete blobs, plus decoded matching separate METADATA.json.','preserve':'Latest NPZ and every JSON/PNG/PDF/model/figure/history/source; no other deletion.','elapsed_seconds':time.monotonic()-start};(O/'PROPOSAL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('proof_passed','logical_bytes','conservative_allocated_reclaim_bytes','elapsed_seconds')}))
