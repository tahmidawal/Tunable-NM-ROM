"""Execute only the coordinator's exact released direct-warm staging policy."""
from pathlib import Path
import subprocess,hashlib,json,shutil,sys
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];REF=REPO.parent/'2026-10-27-ns-obstacle-2d/experiments/ns-obstacle-2d';PACK=B/'staging/development-materialize02';ROOTREVIEW=REPO.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews/development-materialize02';PY='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
release=ROOTREVIEW/'HANDOFF-RELEASE.json';assert sha(release)=='e64ebe7b17b194d1e00df6185d19c359510e41fc2b7a60bf838dd033b423aebe';r=json.loads(release.read_text());inventory=PACK/'INPUT-INVENTORY.json';assert sha(inventory)==r['input_inventory_sha256'];m=json.loads(inventory.read_text());assert sha(PACK/'HANDOFF-PLAN.json')==r['plan_sha256'];job=r['job_directory']
sys.path.insert(0,str(B));from development_handoff_launch2d import validate_release_contract
validate_release_contract(r);assert 'jax' not in sys.modules and 'ngsolve' not in sys.modules
(PACK/'ACTUAL-RELEASE-CONTRACT-CHECK.json').write_text(json.dumps({'release_sha256':sha(release),'launcher_sha256':sha(B/'development_handoff_launch2d.py'),'passed':True,'native_or_JAX_imported':False},indent=2)+'\n')
assert shutil.disk_usage(B).free>=r['local_floor_bytes']+r['other_local_reserved_bytes']+64*2**20
files=[];seen=set()
for row in m['files']:
 base={'rom':B,'reference':REF}[row['owner']];source=base/row['source'];dest=Path(row['destination']);assert not dest.is_absolute() and '..' not in dest.parts and str(dest) not in seen;seen.add(str(dest));assert source.resolve().is_relative_to(base) and not source.is_symlink() and source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
 if 'archive_source' in row:assert sha(base/row['archive_source']['manifest_source'])==row['archive_source']['manifest_sha256']
 files.append((source,str(dest)))
for name in ('HANDOFF-RELEASE.json','PACKET-REVIEW.json'):shutil.copyfile(ROOTREVIEW/name,PACK/name)
assert sha(PACK/'PACKET-REVIEW.json')==r['independent_packet_review_sha256']
files += [(inventory,'INPUT-INVENTORY.json'),(PACK/'HANDOFF-RELEASE.json','HANDOFF-RELEASE.json'),(PACK/'PACKET-REVIEW.json','PACKET-REVIEW.json')]
code=f'''from pathlib import Path
import subprocess,json,shutil
root=Path({job!r});assert not root.exists();base=root.parent.parent
names=['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027'];usage={{n:int(subprocess.check_output(['du','-sb',str(base/n)]).split()[0]) for n in names}};projected=sum(usage.values())+{r['remote_workspace_archive_envelope_bytes']}+{r['other_active_disk_reservation_bytes']};assert projected<{r['global_remote_limit_bytes']} and shutil.disk_usage(base).free>16*2**30
root.mkdir();print(json.dumps({{'namespace_bytes':usage,'maximum_global_bytes':projected}}))
'''
guard=json.loads(subprocess.check_output(['ssh','tufts-login',PY,'-'],input=code.encode()))
dirs=sorted({str(Path(name).parent) for _,name in files if str(Path(name).parent)!='.'})
subprocess.run(['ssh','tufts-login','mkdir','-p',*[job+'/'+name for name in dirs]],check=True)
for source,name in files:subprocess.run(['scp','-q',str(source),'tufts-login:'+job+'/'+name],check=True)
source_commit=subprocess.check_output(['git','-c','gc.auto=0','rev-parse','06309871d'],cwd=REPO,text=True).strip()
code=f'''from pathlib import Path
import os,sys,json,hashlib
root=Path({job!r});os.chdir(root);sys.path.insert(0,str(root));from development_handoff_launch2d import verify
verify('HANDOFF-PLAN.json','HANDOFF-RELEASE.json','INPUT-INVENTORY.json');assert 'jax' not in sys.modules and 'ngsolve' not in sys.modules
Path('SOURCE_COMMIT.txt').write_text({source_commit!r}+'\\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts];Path('SOURCE-SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\\n' for p in sorted(files)));print(json.dumps({{'verified_inputs':92,'native_or_reconstruction_executed':False,'source_manifest_sha256':sha('SOURCE-SHA256SUMS')}}))
'''
check=json.loads(subprocess.check_output(['ssh','tufts-login',PY,'-'],input=code.encode()))
receipt={'schema':'ns-obstacle-released-direct-warm-staging-v1','release_sha256':sha(release),'inventory_sha256':sha(inventory),'guard':guard,'local_bulk_copy_bytes':0,'cold_recovery_executed':False,'source_commit':source_commit,'automatic_submission':False,'staging_script_sha256':sha(__file__),**check};(PACK/'STAGING.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
