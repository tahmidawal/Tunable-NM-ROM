from pathlib import Path,PurePosixPath
import json,hashlib,shutil,subprocess,tarfile
B=Path(__file__).resolve().parents[1];out=B/'artifacts/development-materialize02';root=B.parents[1].parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/development-materialize02/COLLECTION-RELEASE.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(root)=='4ce4e0c11c7db3e7c9be9153adc1cfa2b076460cdf917213cf011cd46a5fc25e'
r=json.loads(root.read_text());measurement=json.loads((out/'ARCHIVE-MEASUREMENT.json').read_text());required=12082428950;free=shutil.disk_usage(B).free;assert free>=required
shutil.copyfile(root,out/'COLLECTION-RELEASE.json');archive=out/'transport.tar.gz';assert not archive.exists();subprocess.run(['scp','-q','tufts-login:'+measurement['archive_path'],str(archive)],check=True);assert archive.stat().st_size==measurement['archive_bytes'] and sha(archive)==measurement['archive_sha256']
rows={};ledger=None
with tarfile.open(archive,'r|gz') as tf:
 for m in tf:
  p=PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and p.parts[0]=='development-materialize02';name=str(PurePosixPath(*p.parts[1:]));assert name not in rows;stream=tf.extractfile(m);h=hashlib.sha256();n=0;pieces=[]
  for block in iter(lambda:stream.read(2**20),b''):
   h.update(block);n+=len(block)
   if name=='COLLECTION-SHA256SUMS':pieces.append(block)
  assert n==m.size;rows[name]={'sha256':h.hexdigest(),'bytes':n}
  if name=='COLLECTION-SHA256SUMS':ledger=b''.join(pieces)
assert ledger is not None and len(rows)==measurement['files'];expected={name:digest for digest,name in (line.split('  ',1) for line in ledger.decode().splitlines())};assert set(expected)==set(rows)-{'COLLECTION-SHA256SUMS'} and all(rows[n]['sha256']==d for n,d in expected.items())
for name,d in measurement['metadata'].items():assert rows[name]==d and sha(out/'metadata'/name)==d['sha256']
(out/'COLLECTION-SHA256SUMS').write_bytes(ledger);(out/'ORIGINAL-FILE-INVENTORY.json').write_text(json.dumps(rows,indent=2)+'\n')
chunks=out/'archive-chunks';chunks.mkdir(exist_ok=False);parts=[]
with archive.open('rb') as f:
 for i in range(100):
  blob=f.read(48*2**20)
  if not blob:break
  p=chunks/f'collection.tar.gz.part-{i:03d}';p.write_bytes(blob);parts.append({'path':str(p.relative_to(out)),'bytes':len(blob),'sha256':sha(p)})
h=hashlib.sha256()
for row in parts:
 with (out/row['path']).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
assert h.hexdigest()==measurement['archive_sha256'];archive.unlink()
record={'schema':'ns-obstacle-development-handoff-retention-v1','status':'checksum_verified_pending_independent_review_and_mirror','job_id':6635386,'state':'COMPLETED','allocated_seconds':51,'archive_sha256':measurement['archive_sha256'],'archive_bytes':measurement['archive_bytes'],'chunks':parts,'verified_original_files':len(rows),'original_file_inventory_sha256':sha(out/'ORIGINAL-FILE-INVENTORY.json'),'collection_ledger_sha256':sha(out/'COLLECTION-SHA256SUMS'),'handoff_sha256':rows['results/population/HANDOFF.json']['sha256'],'collection_release_sha256':sha(root),'local_free_before_bytes':free,'required_local_free_before_bytes':required,'bulk_scientific_extraction':False,'remote_originals_preserved':True,'transport_retired_after_exact_chunk_verification':True,'automatic_retry':False}
(out/'RETENTION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
