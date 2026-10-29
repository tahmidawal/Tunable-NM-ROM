"""Retire only byte-verified duplicate copies; canonical arrays/chunks stay present."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];record_path=B/'staging/COLD-DUPLICATE-CLEANUP-20261028.json'
mirrors={'g2-ladder01':'11be81669','g2-ladder02':'887a06bb1','g2-comparison01':'9957c5a67','g2-h3':'a193581c4','g2-p4trend':'fa940ed39','g2-p4controls':'fa940ed39','g2-crossp01':'609215032','g2-crossp02':'454e0a3ea'}
parser=argparse.ArgumentParser();parser.add_argument('--group',choices=('initial','postp4','batch'),default='initial');args=parser.parse_args()
if args.group=='postp4':
 record_path=B/'staging/COLD-DUPLICATE-CLEANUP-POSTP4-20261029.json'
 mirrors={'g2-solverquad01':'20b812ae3','g2-domain-upstream':'aef32dde0','g2-domain-downstream':'aef32dde0','g2-domain-symmetric':'aef32dde0'}
if args.group=='batch':
 record_path=B/'staging/COLD-DUPLICATE-CLEANUP-BATCH-20261029.json'
 mirrors={'g2-export-batch01':'caee29610'}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(REPO),*args],text=True).strip()
def treebytes(p):return sum(x.stat().st_size for x in p.rglob('*') if x.is_file()) if p.is_dir() else p.stat().st_size
if record_path.exists():raise FileExistsError('cleanup receipt already exists; inspect authoritative state before resuming')
record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_head':git('rev-parse','HEAD'),'free_bytes_before':shutil.disk_usage(B).free,'canonical_arrays_and_archive_chunks_preserved':True,'history_rewritten_or_pruned':False,'cases':[],'completed':False}
def save():record_path.write_text(json.dumps(record,indent=2)+'\n')
save()
for name,mirror in mirrors.items():
 folder=B/'artifacts'/name;retention_path=folder/'RETENTION.json';retention=json.loads(retention_path.read_text());digest=hashlib.sha256();chunk_evidence=[]
 for part in retention['chunks']:
  path=folder/part['path']
  if sha(path)!=part['sha256'] or path.stat().st_size!=part['bytes']:raise ValueError('canonical chunk integrity failed')
  with path.open('rb') as f:
   while block:=f.read(2**20):digest.update(block)
  relative=str(path.relative_to(REPO));blob=git('hash-object',str(path));mirror_blob=git('rev-parse',mirror+':'+relative)
  if blob!=mirror_blob:raise ValueError('durable mirror chunk differs')
  chunk_evidence.append({**part,'git_blob':blob,'mirror_commit':git('rev-parse',mirror)})
 if digest.hexdigest()!=retention['archive_sha256']:raise ValueError('ordered chunks do not reconstruct exact archive')
 collection={}
 for line in (folder/'COLLECTION-SHA256SUMS').read_text().splitlines():
  h,p=line.split('  ',1);collection[p.removeprefix('./')]=h
 candidates=[B/'runs'/name,B/'runs'/(name+'-collection.tar.gz'),B/'staging'/name/'payload'];retired=[]
 case={'name':name,'archive_sha256':retention['archive_sha256'],'chunks':chunk_evidence,'retired':retired,'recovery':'Concatenate canonical chunks in recorded order; verify archive SHA; extract; verify COLLECTION-SHA256SUMS. Staged payload members are in the original job root. Local and mirrored Git also retain exact historical blobs.'};record['cases'].append(case);save()
 for path in candidates:
  if not path.exists():continue
  if path.is_file():
   if sha(path)!=retention['archive_sha256']:raise ValueError('duplicate tar differs')
  else:
   for member in path.rglob('*'):
    if member.is_symlink():raise ValueError('unexpected duplicate symlink')
    if not member.is_file():continue
    key=str(member.relative_to(path))
    if key=='COLLECTION-SHA256SUMS':
     if sha(member)!=sha(folder/'COLLECTION-SHA256SUMS'):raise ValueError('duplicate collection manifest differs')
    elif key in collection:
     if sha(member)!=collection[key]:raise ValueError('duplicate member differs: '+key)
    elif '__pycache__' in member.parts and member.suffix=='.pyc':pass
    elif (folder/key).is_file() and sha(member)==sha(folder/key):
     canonical=folder/key;blob=git('hash-object',str(canonical))
     if blob!=git('rev-parse',mirror+':'+str(canonical.relative_to(REPO))):raise ValueError('extra metadata not durably mirrored')
     case.setdefault('additional_mirrored_metadata',[]).append({'path':key,'sha256':sha(member),'mirror_blob':blob});save()
    else:raise ValueError('unaccounted duplicate member: '+str(member))
  row={'path':str(path.relative_to(B)),'bytes':treebytes(path),'removed':False};retired.append(row);save()
  if path.is_dir():shutil.rmtree(path)
  else:path.unlink()
  row['removed']=True;save()
 retention['full_local_archive']=None;retention['full_local_extraction']=None;retention['canonical_expanded_artifacts']=str(folder);retention['duplicate_extraction_retired']=True;retention['full_tar_duplicate_retired']=True;retention['cold_duplicate_receipt']=str(record_path.relative_to(B));retention_path.write_text(json.dumps(retention,indent=2)+'\n')
record['completed']=True;record['free_bytes_after']=shutil.disk_usage(B).free;record['reclaimed_logical_bytes']=sum(x['bytes'] for c in record['cases'] for x in c['retired'] if x['removed']);record['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps({'reclaimed_bytes':record['reclaimed_logical_bytes'],'free_bytes_after':record['free_bytes_after']}))
