"""Lossless regular-file archive reuse with complete inventories and verified recovery.

Existing archives remain immutable. Reuse is credited only for matching SHA256 and
size. A delta archive retains every new byte once and a map for every logical file.
"""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import argparse,hashlib,importlib.util,io,json,os,shutil,stat,tarfile,tempfile
RESERVE=10*1024**3

def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def safe(name):
 p=PurePosixPath(name)
 if p.is_absolute()or'..'in p.parts or not p.parts:raise ValueError('unsafe retained path')
 return str(p)
def key(digest,size):return digest+':'+str(size)
def read(p):return json.loads(Path(p).read_text())
def dump(p,r):Path(p).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')

def archive_catalog(archive,archive_id,expected,descriptor,warm_root=None,job_name=None):
 archive=Path(archive)
 if sha(archive)!=expected:raise ValueError('base archive changed')
 out={'schema':'ns-obstacle-content-reuse-index-v1','archives':{archive_id:{**descriptor,'sha256':expected,'bytes':archive.stat().st_size}},'entries':{}}
 with tarfile.open(archive,'r|gz')as tar:
  for m in tar:
   if not m.isfile():continue
   f=tar.extractfile(m);digest=hashlib.file_digest(f,'sha256').hexdigest();entry={'sha256':digest,'bytes':m.size,'archive_id':archive_id,'member':safe(m.name)}
   if warm_root and job_name:
    relative=Path(m.name).relative_to(job_name);p=Path(warm_root)/relative
    if p.is_file()and p.stat().st_size==m.size and sha(p)==digest:entry['warm_source']=str(p.resolve())
   out['entries'].setdefault(key(digest,m.size),entry)
 return out

def add_warm_catalog(index,root,prefix,archive_id,descriptor):
 root=Path(root);lines=(root/'COLLECTION-SHA256SUMS').read_text().splitlines();inventory={safe(line.split('  ',1)[1]):line.split('  ',1)[0]for line in lines}
 index['archives'][archive_id]=descriptor
 for p in(root/prefix).rglob('*'):
  if not p.is_file():continue
  relative=str(p.relative_to(root));digest=sha(p)
  if inventory.get(relative)!=digest:raise ValueError('warm member differs from original complete inventory')
  n=p.stat().st_size;entry=index['entries'].setdefault(key(digest,n),{'sha256':digest,'bytes':n,'archive_id':archive_id,'member':safe(descriptor['job_name']+'/'+relative)})
  entry['warm_source']=str(p.resolve())
 return index

def write_delta(root,index_path,output,inventory_path=None,extra_root=None):
 root=Path(root).resolve();output=Path(output).resolve();index=read(index_path)
 if output.exists()or output.is_relative_to(root):raise ValueError('fresh archive outside source directory required')
 if inventory_path:
  expected={safe(line.split('  ',1)[1]):line.split('  ',1)[0]for line in Path(inventory_path).read_text().splitlines()};names=sorted(expected)
 else:expected=None;names=sorted(str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and not p.is_symlink())
 rows={};unique={};symlinks=[]
 for name in names:
  name=safe(name);p=root/name
  if not p.is_file()and extra_root:p=Path(extra_root)/name
  if not p.is_file()or p.is_symlink():raise ValueError('regular source absent:'+name)
  st=p.stat();digest=sha(p)
  if expected and expected[name]!=digest:raise ValueError('original source inventory changed')
  identity=key(digest,st.st_size);row={'sha256':digest,'bytes':st.st_size,'mode':stat.S_IMODE(st.st_mode),'mtime_ns':st.st_mtime_ns}
  if identity in index['entries']:row.update(storage='reuse',reuse_key=identity)
  elif identity in unique:row.update(storage='alias',member=unique[identity]['name'])
  else:row.update(storage='new',member=name);unique[identity]={'name':name,'path':p}
  rows[name]=row
 if not inventory_path:
  for p in root.rglob('*'):
   if p.is_symlink():
    target=os.readlink(p);resolved=(p.parent/target).resolve()
    if not resolved.is_relative_to(root):raise ValueError('external symlink cannot be silently omitted')
    symlinks.append({'path':safe(str(p.relative_to(root))),'target':target})
 manifest={'schema':'ns-obstacle-lossless-delta-archive-v1','reuse_index_sha256':sha(index_path),'files':rows,'symlinks':symlinks,'original_regular_files':len(rows),'logical_regular_bytes':sum(r['bytes']for r in rows.values()),'unique_payload_bytes':sum(v['path'].stat().st_size for v in unique.values()),'reused_logical_bytes':sum(r['bytes']for r in rows.values()if r['storage']=='reuse'),'alias_logical_bytes':sum(r['bytes']for r in rows.values()if r['storage']=='alias'),'no_numerical_omission':True}
 raw=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
 worst_archive=int(manifest['unique_payload_bytes']*1.01)+len(raw)+4096*(len(unique)+1)+16*1024**2
 if shutil.disk_usage(output.parent).free<worst_archive+RESERVE:raise MemoryError('uncompressed delta output reserve')
 manifest['conservative_archive_bound_bytes']=worst_archive
 temporary=output.with_suffix(output.suffix+'.partial')
 if temporary.exists():raise FileExistsError('uncertain prior delta archive')
 with tarfile.open(temporary,'w:gz')as tar:
  remaining_payload=manifest['unique_payload_bytes']
  for entry in unique.values():
   if shutil.disk_usage(output.parent).free<int(remaining_payload*1.01)+len(raw)+4096*(len(unique)+1)+16*1024**2+RESERVE:raise MemoryError('delta write reserve consumed')
   tar.add(entry['path'],arcname='data/'+entry['name'],recursive=False);remaining_payload-=entry['path'].stat().st_size
  raw=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode();info=tarfile.TarInfo('RETENTION-MAP.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 os.replace(temporary,output);receipt={k:manifest[k]for k in ('schema','original_regular_files','logical_regular_bytes','unique_payload_bytes','reused_logical_bytes','alias_logical_bytes')};receipt.update(archive_sha256=sha(output),archive_bytes=output.stat().st_size,retention_map_sha256=hashlib.sha256(raw).hexdigest());return receipt

def recover_delta(archive,expected,index_path,target,archive_paths=None,repo=None,reserve=RESERVE):
 archive=Path(archive);target=Path(target).resolve();index=read(index_path);archive_paths=archive_paths or{}
 if sha(archive)!=expected:raise ValueError('delta archive hash differs')
 if target.exists():raise FileExistsError('recovery never overwrites')
 with tarfile.open(archive,'r:gz')as tar:
  manifest=json.load(tar.extractfile('RETENTION-MAP.json'))
  if manifest['reuse_index_sha256']!=sha(index_path):raise ValueError('reuse index differs')
  rows=manifest['files'];logical=sum(r['bytes']for r in rows.values())
  largest_base=max((index['archives'][index['entries'][r['reuse_key']]['archive_id']]['bytes']for r in rows.values()if r['storage']=='reuse'),default=0)
  if shutil.disk_usage(target.parent).free<logical+largest_base+reserve:raise MemoryError('logical output plus largest cold archive recovery reserve')
  remaining_logical=logical
  target.mkdir();pending={}
  def write(name,stream,row):
   nonlocal remaining_logical
   if shutil.disk_usage(target.parent).free<remaining_logical+reserve:raise MemoryError('remaining logical output reserve consumed')
   p=target/safe(name);p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():raise FileExistsError('recovery collision')
   h=hashlib.sha256();size=0
   with p.open('xb')as f:
    while data:=stream.read(8*1024**2):h.update(data);size+=len(data);f.write(data)
   if h.hexdigest()!=row['sha256']or size!=row['bytes']:raise ValueError('recovered member differs:'+name)
   remaining_logical-=row['bytes']
  for name,row in rows.items():
   safe(name)
   if row['storage']=='new':
    m=tar.getmember('data/'+safe(row['member']))
    if not m.isfile():raise ValueError('payload must be regular')
    write(name,tar.extractfile(m),row)
   elif row['storage']=='reuse':
    e=index['entries'][row['reuse_key']]
    if e['sha256']!=row['sha256']or e['bytes']!=row['bytes']:raise ValueError('reuse identity differs')
    warm=Path(e.get('warm_source','/nonexistent'))
    if not warm.is_absolute()and repo:warm=Path(repo)/safe(str(warm))
    if warm.is_file()and warm.stat().st_size==row['bytes']and sha(warm)==row['sha256']:
     with warm.open('rb')as f:write(name,f,row)
    else:pending.setdefault(e['archive_id'],{}).setdefault(safe(e['member']),[]).append(name)
   elif row['storage']!='alias':raise ValueError('unknown storage representation')
  with tempfile.TemporaryDirectory(prefix='delta-recovery-',dir=target.parent)as scratch:
   for archive_id,needed in pending.items():
    descriptor=index['archives'][archive_id];base=Path(archive_paths[archive_id])if archive_id in archive_paths else None
    if base is None:
     if repo is None:raise ValueError('historical Git recovery repository required')
     helper_path=Path(__file__).with_name('git_archive_recovery.py')
     if sha(helper_path)!=index['recovery_helper_sha256']:raise ValueError('historical recovery helper source differs')
     spec=importlib.util.spec_from_file_location('verified_git_archive_recovery',helper_path);git_archive_recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(git_archive_recovery)
     cold=Path(descriptor['cold_manifest_path'])
     if not cold.is_absolute():cold=Path(repo)/safe(str(cold))
     if sha(cold)!=descriptor['cold_manifest_sha256']:raise ValueError('base archive recovery certificate differs')
     if shutil.disk_usage(target.parent).free<remaining_logical+descriptor['bytes']+reserve:raise MemoryError('cold archive plus remaining logical output reserve')
     base=Path(scratch)/(archive_id+'.tar.gz');git_archive_recovery.recover(SimpleNamespace(repo=Path(repo),manifest=cold,output=base,reserve_bytes=reserve))
    if sha(base)!=descriptor['sha256']:raise ValueError('base archive differs')
    with tarfile.open(base,'r|gz')as original:
     for m in original:
      if m.name not in needed:continue
      if not m.isfile():raise ValueError('reuse locator must be a regular archived member')
      names=needed.pop(m.name);first=names[0];write(first,original.extractfile(m),rows[first])
      for name in names[1:]:
       with(target/first).open('rb')as f:write(name,f,rows[name])
    if needed:raise ValueError('missing declared archive members')
    if str(base).startswith(scratch+'/'):base.unlink()
  for name,row in rows.items():
   if row['storage']=='alias':
    member=safe(row['member'])
    if rows[member]['storage']!='new':raise ValueError('alias must point directly to a new retained member')
    with(target/member).open('rb')as f:write(name,f,row)
   p=target/name;os.chmod(p,row['mode']);os.utime(p,ns=(row['mtime_ns'],row['mtime_ns']))
  for link in manifest['symlinks']:
   p=target/safe(link['path']);p.parent.mkdir(parents=True,exist_ok=True)
   if not(p.parent/link['target']).resolve().is_relative_to(target):raise ValueError('symlink escapes recovered tree')
   p.symlink_to(link['target'])
 return {'all_regular_files_verified':True,'regular_files':len(rows),'logical_bytes':logical,'archive_sha256':expected,'reuse_index_sha256':sha(index_path)}

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True);w=sub.add_parser('write');w.add_argument('--root',type=Path,required=True);w.add_argument('--index',type=Path,required=True);w.add_argument('--output',type=Path,required=True);w.add_argument('--receipt',type=Path,required=True);r=sub.add_parser('recover');r.add_argument('--archive',type=Path,required=True);r.add_argument('--sha256',required=True);r.add_argument('--index',type=Path,required=True);r.add_argument('--target',type=Path,required=True);r.add_argument('--repo',type=Path,required=True);r.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 result=write_delta(a.root,a.index,a.output)if a.action=='write'else recover_delta(a.archive,a.sha256,a.index,a.target,repo=a.repo);dump(a.receipt,result);print(json.dumps(result))
if __name__=='__main__':main()
