from pathlib import Path
import hashlib,json,subprocess,tarfile,io
O=Path(__file__).resolve().parent;W=O.parents[3];R=W.parent/'2026-10-27-ns-obstacle-rom';rel='experiments/ns-obstacle-rom/artifacts/neural-pod-guided-output-polish01';F=R/rel
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(R),*a])
source=git('rev-parse','53ef2504c').decode().strip();mirror=git('rev-parse','0d080927e').decode().strip();tip=git('ls-remote','origin','refs/heads/codeonly/exp/2026-10-27-ns-obstacle-rom').decode().split()[0]
for old,new in [(source,'HEAD'),(mirror,tip)]:subprocess.run(['git','-c','gc.auto=0','-C',str(R),'merge-base','--is-ancestor',old,new],check=True)
ret=json.loads((F/'RETENTION.json').read_text());inv=json.loads((F/'ORIGINAL-FILE-INVENTORY.json').read_text());ledger=(F/'COLLECTION-SHA256SUMS').read_bytes();assert sha(ledger)==ret['collection_ledger_sha256'];assert len(inv)==205
for name in ['RETENTION.json','ORIGINAL-FILE-INVENTORY.json','COLLECTION-SHA256SUMS']:
 for c in [source,mirror]:assert git('show',c+':'+rel+'/'+name)==(F/name).read_bytes()
chunk=ret['chunks'][0];assert len(ret['chunks'])==1 and ret['archive_sha256']=='e57ce5b2fb132da05fc74275d19b0b56b0f6ae12c36385a415db31a1d8b3b993'
raw=git('show',source+':'+rel+'/'+chunk['path']);assert sha(raw)==ret['archive_sha256'] and len(raw)==ret['archive_bytes']
assert raw==git('show',mirror+':'+rel+'/'+chunk['path'])
expected={k.removeprefix('./'):v for v,k in (line.split('  ',1) for line in ledger.decode().splitlines())};seen={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as ar:
 for m in ar:
  if m.isdir():continue
  assert m.isfile();key=str(Path(m.name).relative_to('neural-pod-guided-output-polish01'));assert key not in seen;data=ar.extractfile(m).read();h=sha(data);assert h==inv[key]['sha256'] and len(data)==inv[key]['bytes']
  assert h==(sha(ledger) if key=='COLLECTION-SHA256SUMS' else expected[key]);seen[key]=dict(sha256=h,bytes=len(data))
assert set(seen)==set(inv)==set(expected)|{'COLLECTION-SHA256SUMS'}
paths=[]
for name in [chunk['path'],'metadata/results/fit/POLISH.jsonl','metadata/results/fit/EVALUATIONS.jsonl','metadata/results/fit/QR-BLOCKS.jsonl']:
 f=F/name;s=f.lstat();assert f.is_file() and not f.is_symlink();data=f.read_bytes();h=sha(data);rp=rel+'/'+name;blobs={c:git('rev-parse',c+':'+rp).decode().strip() for c in [source,mirror,tip]};assert len(set(blobs.values()))==1
 for c in [source,mirror,tip]:assert git('show',c+':'+rp)==data
 member=name.removeprefix('metadata/') if name.startswith('metadata/') else None
 if member:assert seen[member]==dict(sha256=h,bytes=len(data))
 else:assert h==ret['archive_sha256']
 paths.append(dict(path=str(f),repository_path=rp,sha256=h,bytes=s.st_size,allocated_bytes=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,link_count=s.st_nlink,source_mirror_advertised_blobs=blobs,archive_member=member))
j=dict(schema='ns-obstacle-output-polish-working-duplicates-cold-proposal-v1',proof_passed=True,deletion_authorized=False,source_commit=source,mirror_commit=mirror,advertised_mirror=tip,archive_sha256=ret['archive_sha256'],archive_bytes=len(raw),originals_verified=len(seen),original_ledger_sha256=sha(ledger),historical_retention_sha256=sha((F/'RETENTION.json').read_bytes()),historical_retention_status_preserved=ret['status'],exact_paths=paths,allocated_bytes=sum(r['allocated_bytes'] for r in paths if r['link_count']==1),recovery='Restore exact original working file from pinned source or mirror blob; verify SHA and byte length. Original JSONL also verified as full archive members; all205 originals and ledger streamed and checked. Archive itself is exact retained one-chunk blob.',reader_clearance='ROM confirms no active/immediate chunk or threeJSONL reader. Current scan uses remote709model/receipt; all JSON/model/fixture/remoteinputs preserved.',preserve='All other paths, especially709model/allJSON/fixture/current scan histories/remoteinputs/Git.',deletions_performed=False)
(O/'POLISH-PROPOSAL.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({'sha256':sha((O/'POLISH-PROPOSAL.json').read_bytes()),'allocated_bytes':j['allocated_bytes'],'originals_verified':len(seen)}))
