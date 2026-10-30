"""Execute only three exact backed releases after all shared preflights pass."""
from pathlib import Path
import hashlib,json,subprocess,shutil,time,importlib.util,numpy as np
O=Path(__file__).resolve().parent;W=O.parents[3];B=W/'experiments/ns-obstacle-2d';C=W.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle';checks=B/'checks'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0','-C',str(W),*a])
settings=[('coarse-full-rule-cold-proposal','INACTIVE-COARSE-FULL-RULE-COOLING-RELEASE.json','83e80e2e70fe7550fbaa473587cba73e34670ff512aa4228a134f5f2d7b3219a',13),('export-batch-fullrule-cold-proposal','REDUNDANT-EXPORT-FULL-RULE-COOLING-RELEASE.json','5210bebd2191c7c301c0355312ab54aeba11bc347e851da8c4d460e7aee83495',1),('superseded-plot-input-cold-proposal','SUPERSEDED-PLOT-INPUT-COOLING-RELEASE.json','f9e514608916c178f25189ffa0bd209e35a5f70fee5a86ec3a1d94884b7ce1da',9)]
life=O/'LIFECYCLE.json';assert not life.exists();head=git('rev-parse','HEAD').decode().strip();tip=git('ls-remote','origin','refs/heads/codeonly/exp/2026-10-27-ns-obstacle-2d').decode().split()[0]
reader=B/'g2/git_archive_recovery.py';assert sha(reader)=='da6d150859f688ad7fba8767d249d9c7fd8f4f237af8c20e80e93351777365f2';spec=importlib.util.spec_from_file_location('recovery',reader);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
allpaths=[];groups=[];protected=[]
for folder,name,h,count in settings:
 rp=C/'reviews/accuracy-improvement01'/name;assert sha(rp)==h;r=json.loads(rp.read_text());assert r['authorized'] is True;p=checks/folder/'PROPOSAL.json';assert sha(p)==r['proposal_sha256'];j=json.loads(p.read_text());paths=[x for g in j['groups'] for x in g['proposed_paths']] if 'groups' in j else j['exact_paths'];assert paths==r['exact_paths'] and len(paths)==count
 recovery=[]
 if 'groups' in j:
  for g in j['groups']:
   d=B/'artifacts'/g['group'];assert sha(d/'RETENTION.json')==g['retention_sha256'] and sha(d/'COLLECTION-SHA256SUMS')==g['original_ledger_sha256'];cp=d/'GIT-COLD-ARCHIVE.json';assert sha(cp)==g['cold_manifest_sha256'];protected.extend([(d/'RETENTION.json',g['retention_sha256']),(d/'COLLECTION-SHA256SUMS',g['original_ledger_sha256']),(cp,g['cold_manifest_sha256'])])
   for old,new in [(g['source_commit'],head),(g['mirror_commit'],tip)]:git('merge-base','--is-ancestor',old,new)
   for chunk in g['chunks']:
    for commit in (g['source_commit'],g['mirror_commit']):assert git('rev-parse',commit+':'+chunk['repository_path']).decode().strip()==chunk['git_blob']
   recovery.append(module.verify(W,json.loads(cp.read_text()),output=None))
 else:
  for old,new in [(j['source_commit'],head),(j['mirror_commit'],tip)]:git('merge-base','--is-ancestor',old,new)
  latest=j['latest_preserved_dataset'];lp=Path(latest['path']);assert sha(lp)==latest['sha256']=='312fd3da9c9cf4c91756319bc1f3253fa463db6d71239c580c331e576b3fee5a';protected.append((lp,latest['sha256']))
  with np.load(lp,allow_pickle=False) as z:current={k:z[k] for k in z.files if k!='metadata_json'}
  for row in paths:
   for commit in (j['source_commit'],j['mirror_commit']):assert git('rev-parse',commit+':'+row['repository_path']).decode().strip()==row['git_blob']
   raw=git('cat-file','blob',row['git_blob']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
   md=Path(row['retained_metadata']['path']);assert sha(md)==row['retained_metadata']['sha256'];protected.append((md,row['retained_metadata']['sha256']))
   with np.load(row['path'],allow_pickle=False) as z:
    for k in z.files:
     if k!='metadata_json':assert k in current and z[k].dtype==current[k].dtype and z[k].shape==current[k].shape and z[k].tobytes()==current[k].tobytes()
    a=z['metadata_json'];assert json.loads(a.tobytes() if a.dtype==np.uint8 else str(a))==json.loads(md.read_text())
  recovery.append({'direct_git_blobs_verified':len(paths),'latest_scientific_containment_replayed':True,'all_original_metadata_separately_retained':True})
 if 'canonical_preserved_copy' in j:
  canonical=j['canonical_preserved_copy'];cp=Path(canonical['path']);st=cp.stat();assert cp.is_file() and not cp.is_symlink() and (st.st_size,st.st_dev,st.st_ino)==(canonical['bytes'],canonical['device'],canonical['inode']) and sha(cp)==canonical['sha256']==paths[0]['sha256'];protected.append((cp,canonical['sha256']))
 for row in paths:
  p=Path(row['path']);assert p.is_relative_to(B) and p.is_file() and not p.is_symlink();st=p.stat();assert (st.st_size,st.st_dev,st.st_ino,st.st_nlink,st.st_blocks*512)==(row['bytes'],row['device'],row['inode'],row['link_count'],row['allocated_bytes']) and sha(p)==row['sha256']
 groups.append({'proposal_path':str(checks/folder/'PROPOSAL.json'),'proposal_sha256':r['proposal_sha256'],'release_path':str(rp),'release_sha256':h,'fresh_recovery_checks':recovery,'exact_paths':paths});allpaths.extend(paths)
assert len(allpaths)==23 and len({r['path'] for r in allpaths})==23 and not {r['path'] for r in allpaths}&{str(p) for p,h in protected}
record={'schema':'ns-obstacle-exact-reference-working-duplicate-cooling-v1','status':'all_preflights_passed','source_head_before':head,'advertised_mirror_tip':tip,'groups':groups,'protected_hashes':[{'path':str(p),'sha256':h} for p,h in protected],'no_reader_clearance':'Root/ROM/numerical explicit no active or immediately planned consumers; latest312fd3da/canonicalfine protected. All source-only future Galerkin exports use latest dataset.','free_before_bytes':shutil.disk_usage(W).free,'started_unix':time.time(),'removed':[]}
def save():life.write_text(json.dumps(record,indent=2)+'\n')
save()
for row in allpaths:
 p=Path(row['path']);st=p.stat();assert (st.st_dev,st.st_ino,st.st_size)==(row['device'],row['inode'],row['bytes']) and sha(p)==row['sha256'];p.unlink();record['removed'].append(row);save()
for p,h in protected:assert sha(p)==h
record.update(status='complete_exact23_working_duplicates_retired',free_after_bytes=shutil.disk_usage(W).free,verified_unlinked_allocated_bytes=sum(r['allocated_bytes'] for r in allpaths if r['link_count']==1));record['observed_free_gain_bytes']=record['free_after_bytes']-record['free_before_bytes'];save();print(json.dumps({k:record[k] for k in ['status','free_before_bytes','free_after_bytes','verified_unlinked_allocated_bytes','observed_free_gain_bytes']}))
