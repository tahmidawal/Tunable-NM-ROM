"""Exact unsubmitted69-file staging provenance; archive-only, no scientific outputs."""
from pathlib import Path,PurePosixPath
import argparse,json,hashlib,subprocess,tarfile,shutil
B=Path(__file__).resolve().parents[1];manifest=B/'checks/diagnostic06-failed-staging-inventory.json';measure=B/'checks/diagnostic06-failed-staging-archive-measurement.json'
REMOTE='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-diagnostic06-preflight-failed01';PY='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
a=argparse.ArgumentParser();a.add_argument('mode',choices=['pack','collect']);a.add_argument('--release',type=Path);a.add_argument('--release-sha256');args=a.parse_args();assert sha(manifest)=='fdac2361a2de802b03db02611719685e2155d9485ed6c1bc74d0f73a8cece8d7';expected=json.loads(manifest.read_text())['files'];assert len(expected)==69
if args.mode=='pack':
 assert not measure.exists()
 code=f'''from pathlib import Path
import json,hashlib,subprocess,shutil
p=Path({REMOTE!r});archive=Path(str(p)+'.tar.gz');expected={expected!r}
assert p.is_dir() and not archive.exists() and not any(x.is_symlink() for x in p.rglob('*'))
assert not any((p/n).exists() for n in ('SUBMISSION-CLAIM.json','SUBMISSION.json','TRAINING-CLAIM.json','DIAGNOSTIC-CLAIM.json'))
actual={{str(x.relative_to(p)):x for x in p.rglob('*') if x.is_file()}};assert set(actual)==set(expected)
for n,x in actual.items():assert x.stat().st_size==expected[n]['bytes'] and hashlib.sha256(x.read_bytes()).hexdigest()==expected[n]['sha256']
roots=[p.parent,Path('/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027'),Path('/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027')]
assert sum(int(subprocess.check_output(['du','-sb',str(x)]).split()[0]) for x in roots)+15*2**30+256*2**20+48*2**20<50*2**30
assert shutil.disk_usage(p).free>16*2**30
subprocess.run(['tar','-czf',str(archive),'--null','-T','-'],cwd=p.parent,input=b'\\0'.join((p.name+'/'+n).encode() for n in sorted(expected))+b'\\0',check=True)
print(json.dumps({{'archive_path':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'original_files':len(expected),'expanded_bytes':sum(x['bytes'] for x in expected.values()),'failed_before_submission':True,'GPU_allocations':0,'remote_originals_preserved':True}}))
'''
 result=json.loads(subprocess.check_output(['ssh','tufts-login',PY,'-'],input=code.encode()));assert result['archive_bytes']<48*2**20;measure.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
else:
 assert args.release and sha(args.release)==args.release_sha256;r=json.loads(args.release.read_text());m=json.loads(measure.read_text());required=10*2**30+256*2**20+2*m['archive_bytes'];assert r['schema']=='ns-obstacle-failed-staging-collection-release-v1' and r['collector_sha256']==sha(__file__) and r['authorized'] is True and r['archive_sha256']==m['archive_sha256'] and r['archive_bytes']==m['archive_bytes'] and r['manifest_sha256']==sha(manifest) and r['required_local_free_bytes']==required
 free=shutil.disk_usage(B).free;assert free>=required;out=B/'artifacts/diagnostic06-preflight-failed01';out.mkdir(exist_ok=False);dest=out/'original-stage.tar.gz';subprocess.run(['scp','-q','tufts-login:'+m['archive_path'],str(dest)],check=True);assert dest.stat().st_size==m['archive_bytes'] and sha(dest)==m['archive_sha256'];seen=set()
 with tarfile.open(dest,'r|gz') as tf:
  for member in tf:
   p=PurePosixPath(member.name);assert member.isfile() and not p.is_absolute() and '..' not in p.parts and p.parts[0]==Path(REMOTE).name;n=str(PurePosixPath(*p.parts[1:]));assert n in expected and n not in seen;seen.add(n);h=hashlib.sha256();size=0
   stream=tf.extractfile(member)
   for block in iter(lambda:stream.read(2**20),b''):h.update(block);size+=len(block)
   assert size==member.size==expected[n]['bytes'] and h.hexdigest()==expected[n]['sha256']
 assert seen==set(expected)
 for src,name in [(manifest,'ORIGINAL-FILE-INVENTORY.json'),(measure,'ARCHIVE-MEASUREMENT.json'),(args.release,'COLLECTION-RELEASE.json')]:shutil.copyfile(src,out/name)
 (out/'RETENTION.json').write_text(json.dumps({'status':'verified_pending_mirror_and_independent_review','archive_sha256':sha(dest),'archive_bytes':dest.stat().st_size,'verified_original_files':len(seen),'GPU_allocations':0,'local_free_before':free,'required_local_free':required,'bulk_extraction':False,'remote_originals_preserved':True},indent=2)+'\n')
 print('verified all69 original staging files; no expansion')
