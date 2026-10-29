from pathlib import Path
import subprocess,json,hashlib
B=Path(__file__).resolve().parents[1];R=B.parents[1].parent/'2026-10-27-ns-obstacle';rel='experiments/ns-obstacle/reviews/accuracy-improvement01/DIAGNOSTIC-REMOTE-CLEANUP-RELEASE.json';release=R/rel
raw=release.read_bytes();assert raw==subprocess.check_output(['git','-c','gc.auto=0','-C',str(R),'show','99f672891:'+rel]);assert raw==subprocess.check_output(['git','-c','gc.auto=0','-C',str(R),'show','87030370c:'+rel]);assert hashlib.sha256(raw).hexdigest()=='b8d496e351c92ef700f0763f136f62d22fdcb63888d6705cc81ea1486ce1b317';r=json.loads(raw)
a=B/'artifacts/neural-pod-guided-diagnostic01';inventory=json.loads((a/'ORIGINAL-FILE-INVENTORY.json').read_text());ret=json.loads((a/'RETENTION.json').read_text());h=hashlib.sha256()
for row in ret['chunks']:
 data=(a/row['path']).read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];h.update(data)
assert h.hexdigest()==r['archive_sha256'];subprocess.run(['git','-c','gc.auto=0','merge-base','--is-ancestor',r['local_archive_source_commit'],'HEAD'],check=True);tip=subprocess.check_output(['git','-c','gc.auto=0','ls-remote','origin','refs/heads/codeonly/exp/2026-10-27-ns-obstacle-rom'],text=True).split()[0];subprocess.run(['git','-c','gc.auto=0','merge-base','--is-ancestor',r['local_archive_mirror_commit'],tip],check=True)
remote='r=json.loads('+repr(json.dumps(r))+');expected=json.loads('+repr(json.dumps(inventory))+')\n'+r'''
base=Path('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-diagnostic01');archive=base.with_suffix('.tar.gz')
assert r['authorized'] is True and r['exact_job_directory']==str(base) and r['exact_archive_path']==str(archive) and r['job_id']==6649088
status=subprocess.check_output(['sacct','-j','6649088','-n','-P','--format=JobID,State,ElapsedRaw,ExitCode'],text=True)
assert '6649088|COMPLETED|534|0:0' in status
queue=subprocess.check_output(['squeue','-u','tawal01','-h','-o','%A|%Z'],text=True);assert '6649088|' not in queue and str(base) not in queue
assert base.is_dir() and not base.is_symlink() and archive.is_file() and not archive.is_symlink()
paths=list(base.rglob('*'));assert not any(p.is_symlink() for p in paths)
files={str(p.relative_to(base)):p for p in paths if p.is_file()};assert set(files)==set(expected)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(archive)==r['archive_sha256'] and archive.stat().st_size==r['archive_bytes'] and sha(base/'COLLECTION-SHA256SUMS')==r['ledger_sha256']
for name,p in files.items():assert p.stat().st_size==expected[name]['bytes'] and sha(p)==expected[name]['sha256']
print(json.dumps({'preflight_verified':True,'files':len(files),'terminal':status,'queue':queue}),flush=True)
removed_bytes=sum(p.stat().st_size for p in files.values())+archive.stat().st_size
shutil.rmtree(base);archive.unlink();assert not base.exists() and not archive.exists()
print(json.dumps({'schema':'ns-obstacle-neural-diagnostic-remote-cleanup-v1','removed_exact_paths':[str(base),str(archive)],'removed_logical_bytes':removed_bytes,'absence_verified':True,'shared_inputs_untouched':True,'other_jobs_untouched':True}),flush=True)
'''
remote='from pathlib import Path\nimport json,subprocess,hashlib,shutil\n'+remote
result=subprocess.run(['ssh','tufts-login','/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python','-'],input=remote,text=True,capture_output=True)
Path(__file__).with_suffix('.stdout').write_text(result.stdout);Path(__file__).with_suffix('.stderr').write_text(result.stderr);result.check_returncode();receipt=json.loads(result.stdout.splitlines()[-1]);receipt.update(release_sha256=hashlib.sha256(raw).hexdigest(),retention_sha256=hashlib.sha256((a/'RETENTION.json').read_bytes()).hexdigest(),advertised_mirror_tip=tip,source_commit=r['local_archive_source_commit'],mirror_commit=r['local_archive_mirror_commit'],reviewers_remote_read_dependency=False)
p=a/'REMOTE-CLEANUP-LIFECYCLE.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n');(a/'REMOTE-CLEANUP-RELEASE.json').write_bytes(raw);print(json.dumps(receipt,indent=2))
