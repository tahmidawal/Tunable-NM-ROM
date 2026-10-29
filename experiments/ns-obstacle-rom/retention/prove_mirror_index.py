"""Exact actual-tree replay of original mirror loop versus batched helper; no push."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,time
import build_mirror_index as batch
BASE='93f399faad67f94ea2b56fe25f13003143e64694';SEED='5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f';MAX=50*1024**2
GIT=['git','-c','gc.auto=0']
def git(args,env=None,data=None):return subprocess.check_output(GIT+args,env=env,input=data)
def main():
 source=git(['rev-parse','HEAD']).decode().strip();mirror=git(['rev-parse','origin/codeonly/exp/2026-10-27-ns-obstacle-rom']).decode().strip()
 results={}
 for strategy in ('original','batched'):
  with tempfile.TemporaryDirectory() as folder:
   env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'));git(['read-tree',SEED+'^{tree}'],env);started=time.perf_counter()
   if strategy=='original':
    changes=git(['diff','--no-renames','--name-status','-z',BASE,source]).split(b'\0')[:-1]
    for status,path in zip(changes[::2],changes[1::2]):
     name=os.fsdecode(path)
     if status==b'D':git(['update-index','--force-remove','--',name],env);continue
     entry=git(['ls-tree','-l',source,'--',name]).split(b'\t',1)[0].split();mode,kind,digest,size=entry
     if kind==b'blob' and int(size)>MAX:git(['update-index','--force-remove','--',name],env);continue
     git(['update-index','--add','--cacheinfo',','.join([mode.decode(),digest.decode(),name])],env)
   else:
    payload,count,skipped=batch.updates(source,BASE,MAX);git(['update-index','-z','--index-info'],env,payload)
   results[strategy]={'tree':git(['write-tree'],env).decode().strip(),'elapsed_seconds':time.perf_counter()-started}
 advertised=git(['rev-parse',mirror+'^{tree}']).decode().strip();assert results['original']['tree']==results['batched']['tree']==advertised
 result={'passed':True,'source_commit':source,'mirror_commit':mirror,'seed_commit':SEED,'local_base':BASE,'maximum_blob_bytes':MAX,
  'helper_sha256':hashlib.sha256(Path(batch.__file__).read_bytes()).hexdigest(),'reference_helper_commit':'d844f77b4','reference_helper_mirror':'4d4951f84',
  'results':results,'current_mirror_tree':advertised,'source_history_or_push_performed':False,'scientific_source_changed':False}
 out=Path('experiments/ns-obstacle-rom/checks/mirror-index-equivalence.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
