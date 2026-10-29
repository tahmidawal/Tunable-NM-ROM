"""Retire only verified recoverable working copies after the independent audit."""
from pathlib import Path
import datetime,json,shutil
import git_archive_recovery as recovery
B=Path(__file__).resolve().parents[1];REPO=B.parents[1]
receipt=B/'staging/CACHE-PROOF-COLD-RETIREMENT.json'
if receipt.exists():raise FileExistsError('inspect existing retirement receipt before resuming')
record={'completed':False,'free_bytes_before':shutil.disk_usage(B).free,'history_rewritten_or_pruned':False,'removed':[],'proofs':{}}
def save():receipt.write_text(json.dumps(record,indent=2)+'\n')
save()
for name in ('g2-cache-primary01','g2-domain-comparison01'):
    folder=B/'artifacts'/name;proof_path=folder/'GIT-COLD-ARCHIVE.json';proof=json.loads(proof_path.read_text())
    recovery.verify(REPO,proof)
    record['proofs'][name]={'path':str(proof_path.relative_to(B)),'sha256':recovery.file_hash(proof_path),'mirror_commit':proof['mirror_commit'],'archive_sha256':proof['archive_sha256']};save()
    targets=[{'path':str((REPO/x['repository_path']).relative_to(B)),'bytes':x['bytes'],'sha256':x['sha256'],'kind':'archive_working_chunk'}for x in proof['chunks']]
    if name=='g2-cache-primary01':
        storage=json.loads((folder/'STORAGE-INVENTORY.json').read_text())
        targets += [{'path':str((folder/x['path']).relative_to(B)),'bytes':x['bytes'],'sha256':x['sha256'],'kind':'archive_backed_audit_only'}for x in storage['archive_backed_only']]
    for item in targets:
        path=B/item['path']
        if path.is_symlink()or not path.is_file()or path.stat().st_size!=item['bytes']or recovery.file_hash(path)!=item['sha256']:raise ValueError('working duplicate changed: '+str(path))
    for item in targets:
        path=B/item['path'];path.unlink();record['removed'].append(item);save()
    retention_path=folder/'RETENTION.json';retention=json.loads(retention_path.read_text());retention.update(chunk_storage='git_history_only',chunk_recovery_manifest='GIT-COLD-ARCHIVE.json',chunk_recovery_tool='experiments/ns-obstacle-2d/g2/git_archive_recovery.py',chunk_recovery_requires='Reachable historical mirror objects, not a shallow tip-only checkout; no history pruning or rewriting.',cold_working_copy_receipt=str(receipt.relative_to(B)))
    if name=='g2-cache-primary01':retention.update(audit_only_working_files_retired=True,warm_file_inventory='STORAGE-INVENTORY.json',warm_inventory_interpretation='directly_tracked entries except historical archive chunks remain warm; archive_backed_only entries recover from exact full archive')
    retention_path.write_text(json.dumps(retention,indent=2)+'\n')
record.update(completed=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),logical_bytes_retired=sum(x['bytes']for x in record['removed']),free_bytes_after=shutil.disk_usage(B).free);save();print(json.dumps({k:record[k]for k in ('completed','logical_bytes_retired','free_bytes_after')}))
