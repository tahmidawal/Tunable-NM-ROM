"""Remove only staged-input duplicates of live canonical cases, with archive recovery."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
B=Path(__file__).resolve().parents[1];REPO=B.parents[1]
mirrors={'g2-crossp01':'609215032','g2-crossp02':'454e0a3ea','g2-comparison01':'9957c5a67','g2-h3':'a193581c4','g2-solverquad01':'20b812ae3'}
receipt=B/'staging/RETIRE-COPIED-CANONICAL-INPUTS-20261029.json'
if receipt.exists():raise FileExistsError('inspect existing retirement receipt before resuming')
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(REPO),*args],text=True).strip()
record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_results_preserved':True,'archive_chunks_preserved':True,'completed':False,'cases':[],'free_bytes_before':shutil.disk_usage(B).free}
def save():receipt.write_text(json.dumps(record,indent=2)+'\n')
save()
for name,mirror in mirrors.items():
 folder=B/'artifacts'/name;inputs=folder/'inputs';manifest=json.loads((folder/'INPUT-MANIFEST.json').read_text());lookup={x.get('destination',x.get('path')):x for x in manifest['files']};retention=json.loads((folder/'RETENTION.json').read_text())
 archive=hashlib.sha256();chunk_records=[]
 for item in retention['chunks']:
  part=folder/item['path']
  if sha(part)!=item['sha256']:raise ValueError('archive chunk changed')
  blob=git('hash-object',str(part));relative=str(part.relative_to(REPO))
  if blob!=git('rev-parse',mirror+':'+relative):raise ValueError('archive chunk not durably mirrored')
  with part.open('rb') as stream:
   while block:=stream.read(2**20):archive.update(block)
  chunk_records.append({**item,'git_blob':blob,'mirror_commit':git('rev-parse',mirror)})
 if archive.hexdigest()!=retention['archive_sha256']:raise ValueError('full archive differs')
 members=[]
 for path in sorted(inputs.rglob('*')):
  if path.is_symlink():raise ValueError('unexpected duplicate symlink')
  if not path.is_file():continue
  key=str(path.relative_to(folder));item=lookup[key];original=B/item.get('source',item.get('source_path'))
  if not original.is_file() or '/results/' not in str(original) or sha(path)!=item['sha256'] or sha(original)!=item['sha256']:raise ValueError('live original input differs or missing: '+key)
  members.append({'removed':str(path.relative_to(B)),'original_preserved':str(original.relative_to(B)),'bytes':path.stat().st_size,'sha256':item['sha256']})
 case={'name':name,'archive_sha256':retention['archive_sha256'],'mirror_chunks':chunk_records,'members':members,'removed':False};record['cases'].append(case);save()
 shutil.rmtree(inputs);case['removed']=True;save()
 retention['copied_input_subtree_retired']=True;retention['copied_input_recovery_receipt']=str(receipt.relative_to(B));(folder/'RETENTION.json').write_text(json.dumps(retention,indent=2)+'\n')
record.update(completed=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reclaimed_bytes=sum(m['bytes'] for c in record['cases'] for m in c['members']),free_bytes_after=shutil.disk_usage(B).free);save();print(record['reclaimed_bytes'],record['free_bytes_after'])
