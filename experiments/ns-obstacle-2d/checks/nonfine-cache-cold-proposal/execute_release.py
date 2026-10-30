"""Execute only the exact backed root release after all preflight checks."""
from pathlib import Path
import hashlib,json,subprocess,shutil,time,os
W=Path(__file__).resolve().parents[4];B=W/'experiments/ns-obstacle-2d';P=Path(__file__).parent;C=W.parent/'2026-10-27-ns-obstacle'/'experiments/ns-obstacle';release_path=C/'reviews/accuracy-improvement01/NONFINE-CACHE-WORKING-COOLING-RELEASE.json'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(W),*args])
assert sha(release_path)=='e07e2cb13a5283c8b87723d72662851976e4a28c8a370f9cb6a9edbce0ff639e';r=json.loads(release_path.read_text());assert r['authorized'] is True
assert sha(P/'PROPOSAL.json')==r['proposal_sha256'];proposal=json.loads((P/'PROPOSAL.json').read_text());paths=[x for g in proposal['groups'] for x in g['proposed_paths']];assert paths==r['exact_paths'] and len(paths)==22
assert all('/caches/p4_g4_h3/' not in row['path'] for row in paths)
lifecycle=P/'LIFECYCLE.json';assert not lifecycle.exists();head=git('rev-parse','HEAD').decode().strip();tip=git('ls-remote','origin',proposal['advertised_mirror_ref']).decode().split()[0]
for group in proposal['groups']:
 folder=B/'artifacts'/group['group'];assert sha(folder/'RETENTION.json')==group['retention_sha256'];assert sha(folder/'COLLECTION-SHA256SUMS')==group['original_ledger_sha256']
 for old,new in ((group['source_commit'],head),(group['mirror_commit'],tip)):subprocess.run(['git','-c','gc.auto=0','-C',str(W),'merge-base','--is-ancestor',old,new],check=True)
 assert sha(folder/'GIT-COLD-ARCHIVE.json')==group['cold_manifest_sha256']
 for chunk in group['chunks']:
  for commit in (group['source_commit'],group['mirror_commit']):assert git('rev-parse',commit+':'+chunk['repository_path']).decode().strip()==chunk['git_blob']
for row in paths:
 p=Path(row['path']);assert p.is_relative_to(B/'artifacts') and p.is_file() and not p.is_symlink();st=p.stat();assert (st.st_size,st.st_dev,st.st_ino,st.st_nlink,st.st_blocks*512)==(row['bytes'],row['device'],row['inode'],row['link_count'],row['allocated_bytes']);assert sha(p)==row['sha256']
# Reverify unchanged archive reader without materializing any bytes.
import importlib.util
reader=B/'g2/git_archive_recovery.py';assert sha(reader)=='da6d150859f688ad7fba8767d249d9c7fd8f4f237af8c20e80e93351777365f2'
spec=importlib.util.spec_from_file_location('recovery',reader);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
recovery_checks=[module.verify(W,json.loads((B/'artifacts'/g['group']/'GIT-COLD-ARCHIVE.json').read_text()),output=None) for g in proposal['groups']]
# Install each exact backed recovery descriptor before retiring any payload.
descriptors=[{'path':str(B/'artifacts'/g['group']/'GIT-COLD-ARCHIVE.json'),'sha256':g['cold_manifest_sha256'],'already_present':True} for g in proposal['groups']]
for group in proposal['groups']:
 if 'proposed_recovery_manifest' not in group:continue
 binding=group['proposed_recovery_manifest'];manifest=Path(binding['path']);assert sha(manifest)==binding['sha256'];target=B/'artifacts'/group['group']/'GIT-COLD-ARCHIVE.json';assert not target.exists();target.write_bytes(manifest.read_bytes());descriptors.append({'path':str(target),'sha256':sha(target)})
record={'schema':'ns-obstacle-exact-reference-duplicate-cooling-lifecycle-v1','release_sha256':sha(release_path),'proposal_sha256':r['proposal_sha256'],'source_head_before':head,'fresh_advertised_mirror_tip':tip,'original_metadata_preserved':True,'no_active_consumers_confirmed':'ROM owner confirmed no active local consumers/staging; all actual jobs use independent immutable remote inputs','recovery_descriptors':descriptors,'fresh_recovery_checks':recovery_checks,'free_before_bytes':shutil.disk_usage(W).free,'removed':[],'status':'preflight_complete','started_unix':time.time()}
def save():lifecycle.write_text(json.dumps(record,indent=2)+'\n')
save()
for row in paths:
 p=Path(row['path']);st=p.stat();assert (st.st_dev,st.st_ino,st.st_size)==(row['device'],row['inode'],row['bytes']);assert sha(p)==row['sha256'];p.unlink();record['removed'].append(row);save()
for g in proposal['groups']:assert sha(B/'artifacts'/g['group']/'RETENTION.json')==g['retention_sha256']
record.update(status='complete_exact22_duplicates_retired',free_after_bytes=shutil.disk_usage(W).free,verified_unlinked_allocated_bytes=sum(x['allocated_bytes'] for x in record['removed'] if x['link_count']==1));record['observed_filesystem_free_gain_bytes']=record['free_after_bytes']-record['free_before_bytes'];save();print(json.dumps({k:record[k] for k in ('status','free_before_bytes','free_after_bytes','verified_unlinked_allocated_bytes','observed_filesystem_free_gain_bytes')}))
