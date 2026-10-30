"""Read-only independent replay of the archived diagnostic inventory and Git bindings."""
import argparse,hashlib,json,pathlib,subprocess,tarfile
a=argparse.ArgumentParser()
for k in ("source", "mirror", "retention-sha256"):a.add_argument("--"+k, required=True)
a.add_argument('--attempt', choices=['neural-pod-guided-trace01', 'neural-pod-guided-energy01',
                                    'neural-pod-guided-diagnostic03', 'neural-pod-guided-diagnostic04',
                                    'neural-pod-guided-traction01', 'neural-pod-guided-diagnostic05'], required=True)
args=a.parse_args()
B=pathlib.Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle-rom')
A=B/'experiments/ns-obstacle-rom/artifacts'/args.attempt
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(B),*args],text=True).strip()
r=json.loads((A/'RETENTION.json').read_text());assert digest(A/'RETENTION.json')==args.retention_sha256
source=git('rev-parse',args.source);mirror=git('rev-parse',args.mirror)
advertised=git('ls-remote','origin','refs/heads/codeonly/exp/2026-10-27-ns-obstacle-rom').split()[0]
subprocess.run(['git','-C',str(B),'merge-base','--is-ancestor',mirror,advertised],check=True)
for name in ['RETENTION.json','ORIGINAL-FILE-INVENTORY.json','COLLECTION-SHA256SUMS']:
 p=A/name;rel=str(p.relative_to(B));blob=git('hash-object',str(p));assert git('rev-parse',source+':'+rel)==blob==git('rev-parse',mirror+':'+rel)
whole=hashlib.sha256();paths=[]
for c in r['chunks']:
 p=A/c['path'];assert p.stat().st_size==c['bytes'] and digest(p)==c['sha256'];paths.append(str(p))
 blob=git('hash-object',str(p));rel=str(p.relative_to(B))
 assert git('rev-parse',source+':'+rel)==blob==git('rev-parse',mirror+':'+rel)
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):whole.update(block)
assert whole.hexdigest()==r['archive_sha256']
assert digest(A/'ORIGINAL-FILE-INVENTORY.json')==r['original_file_inventory_sha256']
inventory=json.loads((A/'ORIGINAL-FILE-INVENTORY.json').read_text());seen={}
proc=subprocess.Popen(['cat',*paths],stdout=subprocess.PIPE)
with tarfile.open(fileobj=proc.stdout,mode='r|gz') as tf:
 for m in tf:
  if m.isdir():continue
  assert m.isfile();p=pathlib.PurePosixPath(m.name);assert not p.is_absolute() and '..' not in p.parts and p.parts[0]==args.attempt
  name=str(pathlib.PurePosixPath(*p.parts[1:]));assert name not in seen
  stream=tf.extractfile(m);h=hashlib.sha256();n=0
  for block in iter(lambda:stream.read(1048576),b''):h.update(block);n+=len(block)
  seen[name]={'bytes':n,'sha256':h.hexdigest()}
assert proc.wait()==0 and seen==inventory and len(seen)==r['verified_original_files']
ledger=A/'COLLECTION-SHA256SUMS';assert digest(ledger)==r['collection_ledger_sha256']
entries=[line.split('  ',1) for line in ledger.read_text().splitlines()]
assert len({name for _,name in entries})==len(entries)
expected={name:h for h,name in entries}
assert set(expected)==set(seen)-{'COLLECTION-SHA256SUMS'} and all(seen[n]['sha256']==h for n,h in expected.items())
print(json.dumps({'passed':True,'source_commit':source,'mirror_commit':mirror,'archive_sha256':whole.hexdigest(),'verified_regular_files':len(seen),'chunk_count':len(paths),'retention_sha256':digest(A/'RETENTION.json'),'bulk_scientific_extraction':False,'remote_deletion_performed':False},indent=2))
