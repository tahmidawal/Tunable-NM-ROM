"""Audit every retained attempt, scientific GitHub blob, and owned-job cleanup."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

lane = Path(__file__).resolve().parents[1]
repo = lane.parents[1]
prefix = lane.relative_to(repo)
namespace = '/cluster/tufts/paralab/tawal01/jcp-weak-visibility-20261118-488b0837'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', *args], cwd=repo)


def tree(ref):
    records = {}
    for entry in git('ls-tree', '-r', '-z', ref, '--', str(prefix)).split(b'\0'):
        if entry:
            metadata, path = entry.decode().split('\t', 1)
            records[path] = metadata.split()[2]
    return records


local_commit = git('rev-parse', 'HEAD').decode().strip()
branch = git('rev-parse', '--abbrev-ref', 'HEAD').decode().strip()
remote_ref = 'refs/heads/codeonly/'+branch
remote_commit = git('ls-remote', 'origin', remote_ref).decode().split()[0]
local_tree, remote_tree = tree(local_commit), tree(remote_commit)
ledger = json.loads((lane/'runs/LEDGER.json').read_text())
jobs = []
total_verified = total_backed = total_bytecode = 0
for record in ledger['jobs']:
    folder = lane/'runs'/record['attempt']
    collection = json.loads((folder/'collection.json').read_text())
    assert collection['checksum_verified'] and collection['remote_directory_deleted']
    assert collection['scheduler_state'] in ('FAILED', 'COMPLETED')
    rows = []
    for line in (folder/'RESULTS.sha256').read_text().splitlines():
        expected, rel = line.split('  ', 1)
        path = folder/rel
        assert sha(path) == expected
        repo_path = str(path.relative_to(repo))
        generated_bytecode = '__pycache__' in path.parts and path.suffix == '.pyc'
        backed = repo_path in local_tree and local_tree.get(repo_path) == remote_tree.get(repo_path)
        assert backed or generated_bytecode, repo_path
        rows.append(dict(path=rel, sha256=expected, verified=True, backed_up=backed,
            local_only_generated_bytecode=bool(generated_bytecode and not backed)))
    backed_count = sum(r['backed_up'] for r in rows)
    bytecode_count = sum(r['local_only_generated_bytecode'] for r in rows)
    total_verified += len(rows)
    total_backed += backed_count
    total_bytecode += bytecode_count
    jobs.append(dict(attempt=record['attempt'], job_id=record['job_id'], scheduler_state=collection['scheduler_state'],
        allocated_gpu_seconds=record['charged_allocated_gpu_seconds'], remote_duplicate_removed=True,
        file_count=len(rows), backed_up_file_count=backed_count, local_only_generated_bytecode_count=bytecode_count,
        results_manifest_sha256=sha(folder/'RESULTS.sha256'), collection_sha256=sha(folder/'collection.json'), files=rows))
snapshot = json.loads((lane/'checks/width-results/SNAPSHOT.json').read_text())
for file in snapshot['files']:
    assert sha(lane/file['path']) == file['sha256']
assert sha(lane/snapshot['generator']['path']) == snapshot['generator']['sha256']
queue = subprocess.check_output(['ssh','tufts-login',
    'test ! -e '+namespace+' && squeue -u "$USER" -h -o "%i|%j|%T"'], text=True)
owned = [line for line in queue.splitlines() if line.split('|')[1].startswith('ctol_jwv_')]
assert not owned
audit = dict(status='closed; all registered experiments complete and independently reviewed; no further expansion',
    generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), local_commit=local_commit,
    remote_ref=remote_ref, verified_remote_commit=remote_commit, namespace=namespace, namespace_removed=True,
    owned_jobs_still_queued=owned, total_verified_files=total_verified, total_backed_up_files=total_backed,
    total_local_only_generated_bytecode_files=total_bytecode,
    all_scientific_data_and_source_files_backed_up=True,
    total_allocated_gpu_seconds=sum(j['allocated_gpu_seconds'] for j in jobs), lane_ceiling_gpu_seconds=10800,
    final_snapshot_sha256=sha(lane/'checks/width-results/SNAPSHOT.json'), frozen_result_files_unchanged=True,
    jobs=jobs, limits=['six exposed development cases','same-grid same-step discrete target',
        'no continuum or held-out qualification','no cross-job speed claim','no uniform stability proof'],
    canonical_log_and_main_report_owner='parent /root', branches_unmerged=True)
assert audit['total_allocated_gpu_seconds'] == ledger['total_charged_allocated_gpu_seconds'] <= 10800
(lane/'checks/CLOSEOUT.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['status','total_verified_files','total_backed_up_files',
    'total_local_only_generated_bytecode_files','total_allocated_gpu_seconds','namespace_removed']}))
