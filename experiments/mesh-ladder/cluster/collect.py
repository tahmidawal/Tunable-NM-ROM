"""Checksum-collect one finished mesh-ladder attempt, archive it, then remove it.

The remote attempt directory is deleted only after every byte has been verified
locally, and only that exact directory is ever removed -- never the namespace.
Any failure leaves the remote directory in place and records why.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL = HERE.parent
ROOT = CELL.parents[1]
NAMESPACE = '/cluster/tufts/paralab/tawal01/mrladder_20260914'
FORBIDDEN = re.compile(
    r'Traceback|RESOURCE_EXHAUSTED|CUDA_ERROR|out of memory|No space left'
    r'|jax_backend=cpu|captured.{0,30}constant|Segmentation fault|core dumped')
CHUNK = 64 * 1024 * 1024


def ssh(command, check=True):
    return subprocess.run(('ssh', 'tufts-login', command), check=check,
                          capture_output=True, text=True).stdout


def job_state(job_id):
    """Slurm returns exit 1 for an expired explicit selector, so filter the account."""
    queued = ssh(f'squeue -u tawal01 -h -o "%i %j %T"', check=False)
    matches = [line for line in queued.splitlines() if line.split()[:1] == [str(job_id)]]
    assert len(matches) <= 1, matches
    if matches:
        return dict(finished=False, queue_line=matches[0])
    record = ssh(f'sacct -X -j {job_id} -n -P -o JobID,JobName%40,State,ElapsedRaw,NodeList',
                 check=False).strip()
    return dict(finished=True, accounting=record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--wait-hours', type=float, default=6.0)
    parser.add_argument('--keep-remote', action='store_true',
                        help='verify and archive but do not remove the remote directory')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_]+', args.attempt), args.attempt
    remote = f'{NAMESPACE}/{args.attempt}'
    local = CELL / 'runs' / args.attempt
    artifacts = CELL / 'artifacts' / args.attempt
    status_path = CELL / 'runs' / f'{args.attempt}-collection.json'
    status = dict(attempt=args.attempt, job=args.job, remote=remote, phase='waiting')

    def save():
        temporary = status_path.with_suffix('.partial')
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)

    save()
    deadline = time.time() + args.wait_hours * 3600
    while True:
        state = job_state(args.job)
        if state['finished']:
            break
        if time.time() > deadline:
            status.update(phase='still_running_at_deadline', **state)
            save()
            raise SystemExit(f'job {args.job} still queued or running: {state["queue_line"]}')
        print(f'waiting: {state["queue_line"]}', flush=True)
        time.sleep(60)
    status.update(accounting=state['accounting'])
    fields = state['accounting'].split('|') if state['accounting'] else []
    status['terminal_state'] = fields[2] if len(fields) > 2 else 'unknown'
    status['node'] = fields[4] if len(fields) > 4 else 'unknown'
    save()

    # ---- logs must prove the GPU ran and nothing silently degraded ----------
    logs = ssh(f'cat {shlex.quote(remote)}/job.out {shlex.quote(remote)}/job.err 2>/dev/null',
               check=False)
    status['jax_backend_gpu'] = 'jax_backend=gpu' in logs
    hits = sorted(set(FORBIDDEN.findall(logs)))
    status['forbidden_log_patterns'] = hits
    status['log_tail'] = logs[-4000:]
    status['driver_complete'] = 'LADDER COMPLETE' in logs
    save()
    if not status['jax_backend_gpu'] or hits:
        status['phase'] = 'failed_collection_requires_review'
        save()
        raise SystemExit(f'log check failed: gpu={status["jax_backend_gpu"]} patterns={hits}')

    # ---- copy every byte and verify it against the remote digests ----------
    status['phase'] = 'copying'
    save()
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(('rsync', '-a', '--checksum',
                    f'tufts-login:{remote}/', str(local) + '/'), check=True)
    manifest = (local / 'OUTPUTS.sha256').read_text().splitlines()
    bad = []
    for line in manifest:
        expected, name = line.split('  ', 1)
        path = local / 'out' / name[2:] if name.startswith('./') else local / 'out' / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            bad.append(name)
    status.update(output_files=len(manifest), checksum_mismatches=bad)
    save()
    if bad:
        status['phase'] = 'failed_collection_requires_review'
        save()
        raise SystemExit(f'{len(bad)} output checksums do not match')

    result = json.loads((local / 'out' / 'result.json').read_text())
    status.update(driver_reported_complete=result.get('complete'),
                  gpu=result.get('gpu'), jax_backend=result.get('jax_backend'),
                  matmul_precision=result.get('matmul_precision'),
                  elapsed_seconds=result.get('elapsed_seconds'),
                  invocations=len(result.get('invocations', [])),
                  cached_invocations=len(result.get('cached_invocations', [])))
    save()
    assert result.get('complete'), 'driver did not record a complete run'

    # ---- one streamed tarball, chunked so git can carry it ------------------
    status['phase'] = 'archiving'
    save()
    artifacts.mkdir(parents=True, exist_ok=False)
    tarball = local.parent / f'{args.attempt}.tar.gz'
    with tarfile.open(tarball, 'w:gz') as archive:
        archive.add(local, arcname=args.attempt)
    whole = hashlib.sha256()
    records = []
    with tarball.open('rb') as stream:
        index = 0
        while True:
            block = stream.read(CHUNK)
            if not block:
                break
            name = f'collection.tar.gz.part{index:04d}'
            (artifacts / name).write_bytes(block)
            records.append(dict(path=name, bytes=len(block),
                                sha256=hashlib.sha256(block).hexdigest()))
            whole.update(block)
            index += 1
    (artifacts / 'archive.json').write_text(json.dumps(dict(
        archive='collection.tar.gz', sha256=whole.hexdigest(),
        bytes=tarball.stat().st_size, chunks=records), indent=2) + '\n')
    (artifacts / 'SHA256SUMS').write_text(
        ''.join(f"{r['sha256']}  {r['path']}\n" for r in records))
    for name in ('PROVENANCE.json', 'MANIFEST.sha256'):
        source = local / 'code' / name
        if source.exists():
            (artifacts / name).write_bytes(source.read_bytes())
    (artifacts / 'result.json').write_bytes((local / 'out' / 'result.json').read_bytes())
    (artifacts / 'README.md').write_text(f"""# {args.attempt}

Checksum-verified collection of cluster job `{args.job}` in namespace
`{NAMESPACE}`. The remote attempt directory was removed after this archive was
verified. `result.json` beside these chunks is the driver record, copied out so
the report generator does not need to unpack the archive.

Restore:

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored/{args.attempt}/out && sha256sum -c ../OUTPUTS.sha256
```
""")
    status.update(phase='archived', archive_sha256=whole.hexdigest(), chunks=len(records),
                  archive_bytes=tarball.stat().st_size)
    save()
    tarball.unlink()

    if args.keep_remote:
        status['phase'] = 'archived_remote_kept'
        save()
        print('archived; remote directory kept by request')
        return
    ssh(f'rm -rf -- {shlex.quote(remote)} && test ! -e {shlex.quote(remote)}')
    status.update(phase='complete', remote_removed=True)
    save()
    print(f'collected {args.attempt}: {len(records)} chunks, remote removed')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
