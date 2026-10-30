from pathlib import Path,PurePosixPath
import json,hashlib,shutil,subprocess,tarfile,argparse
a=argparse.ArgumentParser();a.add_argument('--release-sha256',required=True);a=a.parse_args()
B=Path(__file__).resolve().parents[1];out=B/'artifacts/neural-pod-guided-residual01';root=B.parents[1].parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/accuracy-improvement01/RESIDUAL-COLLECTION-RELEASE.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(root)==a.release_sha256
m=json.loads((B/'checks/neural-pod-guided-residual01-archive-measurement.json').read_text());release=json.loads(root.read_text());limit=release['metadata_limit_bytes'];assert type(limit) is int and 0<limit<=m['expanded_bytes'] and m['metadata_extraction_bytes']<=limit
required=10*2**30+256*2**20+2*limit+2*m['archive_bytes'];free=shutil.disk_usage(B).free;assert free>=required;assert release['archive_sha256']==m['archive_sha256'] and release['archive_bytes']==m['archive_bytes'] and release['required_local_free_bytes']==required
assert m['job_id']==6655193 and release['job_id']==6655193
terminal=[x.split('|') for x in m['terminal_status'].splitlines() if x.split('|')[0]==str(6655193)];assert len(terminal)==1 and terminal[0][1]=='COMPLETED' and terminal[0][3]=='0:0'
out.mkdir(exist_ok=False);shutil.copyfile(root,out/'COLLECTION-RELEASE.json');shutil.copyfile(B/'checks/neural-pod-guided-residual01-archive-measurement.json',out/'ARCHIVE-MEASUREMENT.json');archive=out/'transport.tar.gz'
subprocess.run(['scp','-q','tufts-login:'+m['archive_path'],str(archive)],check=True);assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
rows={};ledger=None;metadata=0
with tarfile.open(archive,'r|gz') as tf:
 for member in tf:
  p=PurePosixPath(member.name);assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='neural-pod-guided-residual01'
  if member.isdir():continue
  assert member.isfile();name=str(PurePosixPath(*p.parts[1:]));assert name and name not in rows
  keep=p.suffix in {'.json','.jsonl','.py','.sh','.txt','.out','.err','.stdout','.stderr'} or p.name in {'COLLECTION-SHA256SUMS','SOURCE-SHA256SUMS'}
  if keep:assert metadata+member.size<=limit
  stream=tf.extractfile(member);h=hashlib.sha256();n=0;pieces=[]
  for block in iter(lambda:stream.read(2**20),b''):
   h.update(block);n+=len(block)
   if keep:pieces.append(block)
  assert n==member.size;rows[name]={'sha256':h.hexdigest(),'bytes':n}
  if keep:
   target=out/'metadata'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b''.join(pieces));assert sha(target)==h.hexdigest();metadata+=n
  if name=='COLLECTION-SHA256SUMS':ledger=b''.join(pieces)
assert metadata==m['metadata_extraction_bytes']
assert ledger is not None and len(rows)==m['regular_files'] and sum(r['bytes'] for r in rows.values())==m['expanded_bytes']
assert hashlib.sha256(ledger).hexdigest()==m['ledger_sha256'];ledger_rows=[line.split('  ',1) for line in ledger.decode().splitlines()];assert len({name for digest,name in ledger_rows})==len(ledger_rows);expected={name:digest for digest,name in ledger_rows};assert set(expected)==set(rows)-{'COLLECTION-SHA256SUMS'} and all(rows[n]['sha256']==d for n,d in expected.items())
(out/'COLLECTION-SHA256SUMS').write_bytes(ledger);(out/'ORIGINAL-FILE-INVENTORY.json').write_text(json.dumps(rows,indent=2)+'\n')
chunks=out/'archive-chunks';chunks.mkdir();parts=[]
with archive.open('rb') as f:
 for i in range(100):
  blob=f.read(48*2**20)
  if not blob:break
  p=chunks/f'collection.tar.gz.part-{i:03d}';p.write_bytes(blob);parts.append({'path':str(p.relative_to(out)),'bytes':len(blob),'sha256':sha(p)})
h=hashlib.sha256()
for row in parts:
 with (out/row['path']).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
assert h.hexdigest()==m['archive_sha256'];archive.unlink()
final_metadata=json.loads((out/'metadata/results/fit/FINAL.json').read_text());assert final_metadata['warm_restart'] is False
record={'schema':'ns-obstacle-supervised-neural-retention-v2','status':'checksum_verified_pending_independent_review_and_mirror','job_id':6655193,'state':'COMPLETED','allocated_seconds':int(next(x.split('|')[2] for x in m['terminal_status'].splitlines() if x.split('|')[0]==str(m['job_id']))),'archive_sha256':m['archive_sha256'],'archive_bytes':m['archive_bytes'],'chunks':parts,'verified_original_files':len(rows),'original_file_inventory_sha256':sha(out/'ORIGINAL-FILE-INVENTORY.json'),'collection_ledger_sha256':sha(out/'COLLECTION-SHA256SUMS'),'collection_release_sha256':sha(root),'local_free_before_bytes':free,'required_local_free_before_bytes':required,'metadata_extracted_bytes':metadata,'bulk_scientific_extraction':False,'remote_originals_preserved':True,'transport_retired_after_exact_chunk_verification':True,'optimizer_status':json.loads((out/'metadata/results/fit/FINAL.json').read_text())['status'],'warm_restart':final_metadata['warm_restart'],'selected_theta_scope':'child parameters only; frozen parent separately embedded','optimizer_converged':json.loads((out/'metadata/results/fit/FINAL.json').read_text())['status']=='gradient_stationary','production_eligible':False,'automatic_retry':False}
(out/'RETENTION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
