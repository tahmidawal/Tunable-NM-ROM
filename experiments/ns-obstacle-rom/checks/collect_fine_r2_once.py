"""One exact released archive-only collection; never expand scientific payloads."""
from pathlib import Path, PurePosixPath
import hashlib, json, shutil, subprocess, tarfile, time
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];A=B/'artifacts/fine-preparation01'
RELEASE=REPO.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/fine-preparation01/RELEASE-COMPACT-R2-COLLECTION.json'
EXPECTED_RELEASE='15b75a7e9781697e4eb6c55bfd1f24a50ced35e4c8f999fbb33d12661799f44f'
FLOOR=10*2**30;OTHER=256*2**20;SMALL=64*2**20

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def safe(name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe archive member')
 return name
assert sha(RELEASE)==EXPECTED_RELEASE
m=json.loads((A/'ARCHIVE-MEASUREMENT-R2.json').read_text());index_path=B/'staging/fine-retention01/REUSE-INDEX.json';index=json.loads(index_path.read_text());composition=json.loads((A/'COMPOSITION-R2.json').read_text())
archive=B/'collections/fine-preparation01.delta-r2.tar.gz';chunks=A/'archive-chunks';assert not archive.exists() and not chunks.exists()
free=shutil.disk_usage(B).free;required=2*m['archive_bytes']+FLOOR+OTHER+SMALL;assert free>=required
start=time.time();dump(B/'checks/fine-r2-collection-preflight.json',{'release_sha256':EXPECTED_RELEASE,'free_bytes':free,'required_bytes':required,'other_bytes':OTHER,'scientific_expansion_bytes':0})
archive.parent.mkdir(exist_ok=True)
subprocess.run(['scp','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login:/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/fine-retention01/fine-preparation01.delta-r2.tar.gz',str(archive)],check=True)
assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
seen={};raw_map=None
with tarfile.open(archive,'r|gz') as tar:
 for member in tar:
  name=safe(member.name);assert member.isfile() and name not in seen
  stream=tar.extractfile(member)
  if name=='RETENTION-MAP.json':
   assert member.size<SMALL;raw_map=stream.read();digest=hashlib.sha256(raw_map).hexdigest()
  else:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  seen[name]={'sha256':digest,'bytes':member.size}
assert raw_map is not None and hashlib.sha256(raw_map).hexdigest()==m['retention_map_sha256']
r=json.loads(raw_map);assert r['reuse_index_sha256']==sha(index_path)
assert len(r['files'])==m['original_regular_files'] and sum(x['bytes'] for x in r['files'].values())==m['logical_regular_bytes']
new_names={'RETENTION-MAP.json'};verified_reuse=set()
for name,row in r['files'].items():
 safe(name)
 if row['storage']=='new':
  key='data/'+safe(row['member']);new_names.add(key);assert seen[key]=={k:row[k] for k in ('sha256','bytes')}
 elif row['storage']=='reuse':
  e=index['entries'][row['reuse_key']];assert all(e[k]==row[k] for k in ('sha256','bytes'))
  if row['reuse_key'] not in verified_reuse:
   p=REPO/e['warm_source'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];verified_reuse.add(row['reuse_key'])
 elif row['storage']=='alias':
  target=r['files'][row['member']];assert target['storage']=='new' and all(target[k]==row[k] for k in ('sha256','bytes'))
 else:raise ValueError('unknown delta storage')
assert set(seen)==new_names
added={'COMPOSITION.json','PRIMARY-REVIEW.json','NUMERICAL-REVIEW.json','INDEPENDENT-RESULT.json','INDEPENDENT-SPECTRAL-RESULT.json'}
assert set(r['files'])==set(composition['retained_original_files'])|added
for name,row in composition['retained_original_files'].items():assert all(r['files'][name][k]==row[k] for k in ('sha256','bytes'))
assert r['files']['COMPOSITION.json']['sha256']==m['composition_sha256']==sha(A/'COMPOSITION-R2.json')
for name in added-{'COMPOSITION.json'}:
 source=B/'staging/fine-retention01'/('PRIMARY-REVIEW.json' if name=='INDEPENDENT-RESULT.json' else name)
 assert sha(source)==r['files'][name]['sha256']
 target=A/'retained-reviews'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
for descriptor in index['archives'].values():assert sha(REPO/descriptor['cold_manifest_path'])==descriptor['cold_manifest_sha256']
(A/'DELTA-RETENTION-MAP.json').write_bytes(raw_map)
verification={'schema':'ns-obstacle-archive-only-stream-verification-v1','archive_sha256':m['archive_sha256'],'archive_bytes':m['archive_bytes'],'retention_map_sha256':m['retention_map_sha256'],'payload_members_streamed':len(seen)-1,'logical_regular_files_verified':len(r['files']),'reuse_identities_reverified':len(verified_reuse),'all_new_payload_and_reused_and_alias_bytes_verified':True,'scientific_arrays_expanded':False,'native_reconstruction_executed':False,'symlinks_retained_in_exact_map':r['symlinks'],'original_remote_trees_preserved':True,'release_sha256':EXPECTED_RELEASE,'collection_script_sha256':sha(__file__)}
dump(A/'ARCHIVE-VERIFICATION.json',verification)
assert shutil.disk_usage(B).free>=m['archive_bytes']+FLOOR+OTHER+SMALL
chunks.mkdir();rows=[]
with archive.open('rb') as stream:
 while data:=stream.read(48*2**20):
  assert shutil.disk_usage(B).free>=len(data)+FLOOR+OTHER
  p=chunks/('collection.delta.tar.gz.part-'+str(len(rows)).zfill(3));p.write_bytes(data);rows.append({'path':str(p.relative_to(A)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
h=hashlib.sha256()
for row in rows:
 p=A/row['path'];assert sha(p)==row['sha256']
 with p.open('rb') as stream:
  while data:=stream.read(8*2**20):h.update(data)
assert h.hexdigest()==m['archive_sha256'];archive.unlink();assert not archive.exists()
retention={'schema':'ns-obstacle-fine-primary-delta-retention-v1','status':'archive_only_all_payload_and_reuse_bytes_verified_pending_durable_mirror','job_id':6633574,'archive_kind':'lossless_delta_of_compact_coefficient_primary_view','archive_sha256':m['archive_sha256'],'archive_bytes':m['archive_bytes'],'chunks':rows,'composition_sha256':m['composition_sha256'],'composition_path':'COMPOSITION-R2.json','retention_map_sha256':m['retention_map_sha256'],'retention_map_path':'DELTA-RETENTION-MAP.json','verification_sha256':sha(A/'ARCHIVE-VERIFICATION.json'),'reuse_index_path':str(index_path.relative_to(REPO)),'reuse_index_sha256':sha(index_path),'recovery_reader_path':str((B/'retention_delta/delta_retention.py').relative_to(REPO)),'recovery_reader_sha256':sha(B/'retention_delta/delta_retention.py'),'base_archives':index['archives'],'original_volume_recovery':'Exact existing primary_volume.materialize under approved_capacity, recorded source/runtime and full archived release/input closure; compare every recorded original archive and array hash. Charge regeneration offline.','original_remote_volumes_preserved':True,'scientific_arrays_expanded_locally':False,'omission_or_deletion_authorized':False,'duplicate_transport_retired_after_complete_chunk_rehash':True,'collected_unix':time.time(),'collection_wall_seconds':time.time()-start}
dump(A/'RETENTION.json',retention);shutil.copyfile(RELEASE,A/'COLLECTION-RELEASE.json')
print(json.dumps({'archive_sha256':m['archive_sha256'],'chunks':len(rows),'retention_sha256':sha(A/'RETENTION.json'),'free_after_bytes':shutil.disk_usage(B).free,'next_git_requirement_bytes':m['archive_bytes']+FLOOR+OTHER+SMALL},indent=2))
