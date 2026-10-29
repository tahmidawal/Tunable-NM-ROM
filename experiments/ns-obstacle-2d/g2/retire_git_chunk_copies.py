"""Retire only Git-proven archive working copies; both histories remain intact."""
from pathlib import Path
import datetime,json,shutil,types
import git_archive_recovery as recovery
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];source='fa1d87196';mirror='caee29610';mirror_ref='refs/heads/codeonly/exp/2026-10-27-ns-obstacle-2d'
receipt=B/'staging/GIT-COLD-CHUNK-RETIREMENT-20261029.json'
if receipt.exists():raise FileExistsError('inspect existing retirement before continuing')
record={'source_commit':source,'mirror_commit':mirror,'free_bytes_before':shutil.disk_usage(B).free,'canonical_results_unchanged':True,'history_pruned_or_rewritten':False,'cases':[],'skipped':[],'completed':False}
def save():receipt.write_text(json.dumps(record,indent=2)+'\n')
save()
for retention_path in sorted((B/'artifacts').glob('*/RETENTION.json')):
 folder=retention_path.parent;retention=json.loads(retention_path.read_text());relative=str(retention_path.relative_to(REPO))
 if retention['status']!='complete_all_regular_files_verified':record['skipped'].append({'name':folder.name,'reason':'status outside reviewed helper'});save();continue
 try:recovery.git(REPO,'cat-file','-e',source+':'+relative)
 except Exception:record['skipped'].append({'name':folder.name,'reason':'not yet in pinned source/mirror'});save();continue
 proof_path=folder/'GIT-COLD-ARCHIVE.json'
 recovery.prepare(types.SimpleNamespace(repo=REPO,retention_path=relative,source_commit=source,mirror_commit=mirror,mirror_ref=mirror_ref,output=proof_path))
 proof=json.loads(proof_path.read_text());case={'name':folder.name,'proof':str(proof_path.relative_to(B)),'proof_sha256':recovery.file_hash(proof_path),'archive_bytes':proof['archive_bytes'],'removed':[]};record['cases'].append(case);save()
 for item in proof['chunks']:
  path=REPO/item['repository_path']
  if recovery.file_hash(path)!=item['sha256']:raise ValueError('working copy changed after proof')
  path.unlink();case['removed'].append(item['repository_path']);save()
 retention.update(chunk_storage='git_history_only',chunk_recovery_manifest='GIT-COLD-ARCHIVE.json',chunk_recovery_tool='experiments/ns-obstacle-2d/g2/git_archive_recovery.py',chunk_recovery_requires='Reachable historical mirror objects, not a shallow tip-only checkout. Use recover with exact proof; do not prune or rewrite either history.',chunk_retirement_receipt=str(receipt.relative_to(B)))
 retention_path.write_text(json.dumps(retention,indent=2)+'\n');print(folder.name,'working chunks retired',flush=True)
record.update(completed=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes_after=shutil.disk_usage(B).free,reclaimed_bytes=sum(c['archive_bytes'] for c in record['cases']));save();print(record['reclaimed_bytes'],record['free_bytes_after'])
