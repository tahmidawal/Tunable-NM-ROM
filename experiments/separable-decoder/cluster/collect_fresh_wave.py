"""Verify, archive, and remove one completed approved fresh-wave job directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

NAMESPACE = '/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906'


def run(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs)


def archive(record, never_started=False):
    cluster = record/'cluster'
    run(['sha256sum', '-c', 'PULL.sha256', '--quiet'], cwd=cluster)
    run(['sha256sum', '-c', 'MANIFEST.sha256', '--quiet'], cwd=cluster)
    outputs = list((cluster/'out').rglob('*'))
    has_outputs = any(path.is_file() for path in outputs)
    if (cluster/'RESULTS.sha256').exists() and (cluster/'RESULTS.sha256').read_text().strip():
        run(['sha256sum', '-c', 'RESULTS.sha256', '--quiet'], cwd=cluster)
    elif has_outputs or (not never_started and not (cluster/'RESULTS.sha256').exists()):
        raise RuntimeError('Missing output checksum coverage')
    path = record/'verified-cluster.tar.gz'
    if path.exists():
        raise RuntimeError('Archive already exists; refusing to replace it')
    with tarfile.open(path, 'w:gz') as tar:
        tar.add(cluster, arcname='cluster')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    size = path.stat().st_size
    parts = []
    if size > 90*1024**2:
        joined = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(90*1024**2):
                part = record/f'{path.name}.part-{len(parts):03d}'
                part.write_bytes(chunk)
                joined.update(chunk)
                parts.append((part.name, hashlib.sha256(chunk).hexdigest()))
        if joined.hexdigest() != digest:
            raise RuntimeError('Split archive does not reconstruct original bytes')
        path.unlink()
    else:
        parts = [(path.name, digest)]
    (record/'ARCHIVE.sha256').write_text(''.join(f'{sha}  {name}\n' for name,sha in parts))
    (record/'ARCHIVE.json').write_text(json.dumps(dict(
        original_name=path.name, original_sha256=digest, original_bytes=size,
        ordered_parts=[name for name,_ in parts],
        restore='Concatenate ordered parts byte-for-byte, then extract gzip tar; full PULL.sha256 is inside.'
    ), indent=2)+'\n')
    return digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('label')
    parser.add_argument('--archive-existing', action='store_true', help='Archive a previously verified and removed job; no network mutation')
    parser.add_argument('--never-started', action='store_true', help='Require canceled, zero elapsed, no start/allocation/output; archive staged inputs only')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,30}', args.label):
        parser.error('invalid immutable attempt label')
    record = Path(__file__).resolve().parents[1]/'runs/fresh_wave_campaign'/args.label
    cfg = json.loads((record/'submission.json').read_text())
    remote = f'{NAMESPACE}/{args.label}'
    if cfg['remote'] != remote or not re.fullmatch(r'[0-9]+', str(cfg['job_id'])):
        raise RuntimeError('Unexpected remote namespace or nonnumeric job ID')
    if args.archive_existing:
        cleanup = json.loads((record/'cleanup.json').read_text())
        if not cleanup['remote_deleted_and_absence_checked']:
            raise RuntimeError('Existing archive mode requires recorded cleanup')
        digest = archive(record)
        print(json.dumps({'archive_sha256': digest, 'record': str(record)}))
        return
    jid = str(cfg['job_id'])
    queue = run(['ssh', 'tufts-login', 'squeue -h -u tawal01 -o "%i %T"'])
    if any(line.split()[0] == jid for line in queue.splitlines() if line.strip()):
        raise RuntimeError('Job remains queued; do not collect a mutable directory')
    accounting = run(['ssh', 'tufts-login', f'sacct -j {jid} --format=JobID,JobName,State%30,ExitCode,Elapsed,ElapsedRaw,Start,NodeList,AllocTRES -P'])
    (record/'accounting.txt').write_text(accounting)
    parsed = [line.split('|') for line in accounting.splitlines()]
    job = next((dict(zip(parsed[0], row)) for row in parsed[1:] if row[0] == jid), None)
    if not job or job['State'].split()[0] not in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL'):
        raise RuntimeError('No terminal accounting row')
    if args.never_started and not (job['State'].startswith('CANCELLED') and job['ElapsedRaw'] == '0' and job['Start'] == 'None' and job['NodeList'] == 'None assigned'):
        raise RuntimeError('Never-started collection requested for a job that may have run')
    quoted = shlex.quote(remote)
    if args.never_started:
        if run(['ssh', 'tufts-login', f'find {quoted}/out {quoted}/logs -type f']).strip():
            raise RuntimeError('Unexpected runtime files in never-started job')
        guard = 'test ! -f EXIT_CODE'
    else:
        guard = 'test -f EXIT_CODE'
    run(['ssh', 'tufts-login', f'cd {quoted} && {guard} && find . -type f ! -name PULL.sha256 -print0 | sort -z | xargs -0 sha256sum > PULL.sha256'])
    cluster = record/'cluster'
    if cluster.exists():
        raise RuntimeError('Local pull exists; inspect it rather than mixing pulls')
    cluster.mkdir()
    subprocess.run(['scp', '-q', '-r', f'tufts-login:{remote}/.', str(cluster)+'/'], check=True)
    digest = archive(record, never_started=args.never_started)
    for name, expected in cfg['source_hashes'].items():
        actual = hashlib.sha256((cluster/'code'/name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Source provenance mismatch: {name}')
    # The exact approved directory is removed only after every local check passed.
    run(['ssh', 'tufts-login', f'test -f {quoted}/PULL.sha256 && rm -rf -- {quoted} && test ! -e {quoted}'])
    queue_after = run(['ssh', 'tufts-login', 'squeue -u tawal01 -o "%.18i %.30j %.8T"'])
    cleanup = dict(job_id=jid, remote=remote, pull_manifest_verified=True,
                   output_manifest_verified=True, input_manifest_verified=True,
                   source_commit=cfg['source_commit'], archive_sha256=digest,
                   never_started=args.never_started,
                   remote_deleted_and_absence_checked=True,
                   checked_at_utc=datetime.now(timezone.utc).isoformat(),
                   queue_after_cleanup=queue_after)
    (record/'cleanup.json').write_text(json.dumps(cleanup, indent=2)+'\n')
    print(json.dumps(cleanup, indent=2))


if __name__ == '__main__':
    main()
