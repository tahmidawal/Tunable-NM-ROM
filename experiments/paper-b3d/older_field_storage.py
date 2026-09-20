"""Verify older duplicate fields and restore selected paths; never delete them.

The plan is restricted to inactive B001--B005 output NPZs containing fields.
Tracked files, checkpoints, offline assets, metadata and archives are excluded.
Any later removal requires separate authorization for the exact saved plan.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from scientific_archive import EXP, ROOT, GitReader

PINS = {
    'b3d001': 'bb0fe9ed715b81933a15f44abba6d6c8a6283392',
    'b3d002': 'bb0fe9ed715b81933a15f44abba6d6c8a6283392',
    'b3d003': '7528fba6519ae56fccec8d83fb0e9d5a13ba9514',
    'b3d004': '7528fba6519ae56fccec8d83fb0e9d5a13ba9514',
    'b3d005': '7528fba6519ae56fccec8d83fb0e9d5a13ba9514',
}
PLAN = EXP / 'checks/older-field-storage-20260920/PLAN.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return value.hexdigest()


def manifest(attempt):
    directory = EXP / 'runs' / attempt / 'scientific-archive'
    path = directory / 'manifest.json'
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        PINS[attempt] + ':' + str(path.relative_to(ROOT))])
    assert raw == path.read_bytes(), 'local manifest differs from pinned Git'
    return directory, json.loads(raw), hashlib.sha256(raw).hexdigest()


def verify_archive(attempt):
    started = now()
    directory, data, sha = manifest(attempt)
    expected = {row['path']: row for row in data['files']}
    seen = {}
    reader = GitReader(PINS[attempt], directory, data['parts'])
    with tarfile.open(fileobj=reader, mode='r|') as archive:
        for member in archive:
            assert member.name in expected and member.name not in seen
            row = expected[member.name]
            if member.islnk():
                assert member.linkname in seen and member.linkname == row['hardlink_to']
                actual, size = seen[member.linkname]
            else:
                assert member.isfile()
                value = hashlib.sha256()
                size = 0
                stream = archive.extractfile(member)
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    value.update(block)
                    size += len(block)
                actual = value.hexdigest()
            assert (actual, size) == (row['sha256'], row['bytes']), member.name
            seen[member.name] = actual, size
    while reader.read(8 * 1024 * 1024):
        pass
    assert set(seen) == set(expected)
    assert reader.total.hexdigest() == data['archive_sha256']
    assert reader.bytes == data['archive_bytes']
    proof = dict(passed=True, started_utc=started, completed_utc=now(), attempt=attempt,
        archive_commit=PINS[attempt], manifest_sha256=sha, files=len(seen),
        parts=len(reader.verified), archive_bytes=reader.bytes,
        restored_file_bytes=sum(size for _, size in seen.values()),
        scope='Every file and exact-content hard-link alias reconstructed and SHA256 checked from actual committed Git blobs; every part and concatenated archive checked; no local array used as proof.')
    dump(PLAN.parent / (attempt + '-GIT-RECHECK.json'), proof)
    print('GIT VERIFIED', attempt, len(seen), reader.bytes, flush=True)
    return proof


def tracked():
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z'])
    return {value.decode() for value in raw.split(b'\0') if value}


def is_output_field_path(name):
    path = PurePosixPath(name)
    return str(path.parent) in ('out', 'out/head64', 'out/reference_screen',
        'out/reference-profile', 'out/seed0', 'out/seed1') and path.suffix == '.npz'


def plan():
    assert not PLAN.exists(), 'preserve the previous plan'
    started = now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        proofs = dict(zip(PINS, pool.map(verify_archive, PINS)))
    kept = tracked()
    attempts = {}
    for attempt in PINS:
        directory, data, sha = manifest(attempt)
        collected = directory.parent / 'collected'
        candidates = []
        preserved = []
        for row in data['files']:
            path = collected / row['path']
            reason = None
            if not is_output_field_path(row['path']):
                reason = 'outside inactive output NPZ scope; preserve all models, checkpoints, offline assets and metadata'
            elif str(path.relative_to(ROOT)) in kept:
                reason = 'individually tracked source or retained replay/offline asset'
            else:
                assert path.is_file() and not path.is_symlink()
                with zipfile.ZipFile(path) as archive:
                    if 'fields.npy' not in archive.namelist():
                        reason = 'not a dense field bundle; preserve parameters, bases, directions and diagnostics'
            if reason:
                preserved.append(dict(path=row['path'], bytes=row['bytes'], reason=reason))
                continue
            stat = path.stat()
            assert stat.st_nlink == 1 and stat.st_size == row['bytes']
            assert digest(path) == row['sha256'], row['path']
            assert path.stat().st_mtime_ns == stat.st_mtime_ns
            candidates.append(dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256'],
                allocated_bytes=stat.st_blocks * 512, inode=stat.st_ino,
                mtime_ns=stat.st_mtime_ns, hardlink_count=stat.st_nlink, removed=False))
        for row in data['parts']:
            path = directory / row['path']
            assert path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
            assert str(path.relative_to(ROOT)) in kept
        attempts[attempt] = dict(archive_commit=PINS[attempt], archive_manifest_sha256=sha,
            git_proof=proofs[attempt], collection=str(collected), candidates=candidates,
            candidate_files=len(candidates), candidate_bytes=sum(row['bytes'] for row in candidates),
            candidate_allocated_bytes=sum(row['allocated_bytes'] for row in candidates),
            preserved_scientific_files=preserved, preserved_archive_exclusions=data['excluded'],
            local_archive_parts_preserved=len(data['parts']), local_archive_part_check='presence, size and individually tracked status; actual Git archive bytes fully hashed above')
        print('LOCAL VERIFIED', attempt, len(candidates), attempts[attempt]['candidate_allocated_bytes'], flush=True)
    result = dict(state='verified plan only; no older fields removed; broader removal authorization pending',
        started_utc=started, completed_utc=now(), approval_required='Root must authorize this exact older-attempt candidate list; the earlier B007-only approval does not apply.',
        attempts=attempts, candidate_files=sum(row['candidate_files'] for row in attempts.values()),
        candidate_bytes=sum(row['candidate_bytes'] for row in attempts.values()),
        candidate_allocated_bytes=sum(row['candidate_allocated_bytes'] for row in attempts.values()),
        scientific_results_changed=False,
        preserve='All checkpoints/offline assets, all tracked files, metadata, archived source, archive parts and Git blobs; all B006/B007 and all local-only cache/optimizer exclusions untouched.',
        restore_command='/home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/older_field_storage.py restore ATTEMPT --path PATH_RELATIVE_TO_COLLECTED --destination NEW_DIRECTORY',
        removal_helper='No deletion mode is implemented. Authorization and a fresh unchanged-file check are required before any later removal.')
    dump(PLAN, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'attempts'}, indent=2))


def restore(args):
    record = json.loads(PLAN.read_text())['attempts'][args.attempt]
    directory, data, sha = manifest(args.attempt)
    assert sha == record['archive_manifest_sha256']
    available = {row['path']: row for row in record['candidates']}
    names = sorted(available) if args.all_candidates else args.path
    assert names and len(set(names)) == len(names) and set(names) <= set(available)
    destination = Path(args.destination).resolve()
    assert destination.is_relative_to(ROOT), 'this owner writes only inside its worktree'
    destination.mkdir(parents=True, exist_ok=True)
    entries = {row['path']: row for row in data['files']}
    sources = {}
    already = []
    temporary = []
    for name in names:
        path = destination / name
        path.resolve().relative_to(destination)
        row = entries[name]
        source = row.get('hardlink_to', name)
        assert 'hardlink_to' not in entries[source]
        assert entries[source]['sha256'] == row['sha256'] == available[name]['sha256']
        if path.exists():
            assert path.is_file() and not path.is_symlink() and digest(path) == row['sha256']
            already.append(name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.restoring')
        assert not temp.exists()
        sources.setdefault(source, []).append((name, path, temp))
    reader = GitReader(PINS[args.attempt], directory, data['parts'])
    restored = []
    try:
        with tarfile.open(fileobj=reader, mode='r|') as archive:
            for member in archive:
                if member.name not in sources:
                    continue
                assert member.isfile()
                targets = sources.pop(member.name)
                handles = [temp.open('xb') for _, _, temp in targets]
                temporary.extend(targets)
                value = hashlib.sha256()
                count = 0
                try:
                    stream = archive.extractfile(member)
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        value.update(block)
                        count += len(block)
                        for handle in handles:
                            handle.write(block)
                    for handle in handles:
                        handle.flush()
                        os.fsync(handle.fileno())
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    for handle in handles:
                        handle.close()
                for name, path, temp in targets:
                    row = entries[name]
                    assert value.hexdigest() == row['sha256'] and count == row['bytes']
                    restored.append(dict(path=name, sha256=row['sha256'], bytes=count,
                        canonical_archive_entry=member.name))
        while reader.read(8 * 1024 * 1024):
            pass
        assert not sources and reader.total.hexdigest() == data['archive_sha256']
        assert reader.bytes == data['archive_bytes']
        for name, path, temp in temporary:
            os.link(temp, path)
            temp.unlink()
    finally:
        for name, path, temp in temporary:
            if temp.exists():
                temp.unlink()
    answer = dict(passed=True, completed_utc=now(), attempt=args.attempt,
        archive_commit=PINS[args.attempt], manifest_sha256=sha,
        archive_bytes_checked=reader.bytes, destination=str(destination),
        restored=restored, already_matching=already,
        scope='Requested fields SHA256 verified from actual Git bytes; all parts and concatenated archive verified.')
    dump(destination / 'restore-audit.json', answer)
    print(json.dumps(answer, indent=2))


def describe():
    record = json.loads(PLAN.read_text())
    lines = ['# Verified older Burgers field storage plan', '',
        'This is a storage plan only. Older fields have not been deleted; broader authorization is still pending. Scientific results and the accepted B007 final are unchanged.', '',
        '| Attempt | Candidate files | Logical bytes | Allocated bytes reclaimable | Actual Git files verified | Archive commit |',
        '|---|---:|---:|---:|---:|---|']
    for attempt, row in record['attempts'].items():
        lines.append(f"| {attempt} | {row['candidate_files']} | {row['candidate_bytes']} | {row['candidate_allocated_bytes']} | {row['git_proof']['files']} | `{row['archive_commit']}` |")
    lines.extend(['',
        f"Total: **{record['candidate_files']} files**, **{record['candidate_allocated_bytes']} allocated bytes** ({record['candidate_allocated_bytes'] / 2**30:.6f} GiB). These are potential savings, not space already reclaimed.", '',
        'The exact per-path SHA256, size, allocated bytes, inode and modification time are in `PLAN.json`. Fresh, timestamped actual-Git proofs are in the adjacent `b3d00*-GIT-RECHECK.json` files. Every archived file, archive hard-link alias, part and concatenated archive checksum passed. Every proposed local file separately matched its archived SHA256. The helper has no deletion mode.', '',
        'Candidates are untracked NPZ bundles containing a `fields` array, only under the recorded inactive output folders. All checkpoints, model/offline assets, tracked replay fields, metadata, archived source, local archive parts and Git blobs are excluded. All B006/B007 files and local-only training/optimizer exclusions are untouched. In particular, `.npy` training caches omitted from the archives are not eligible under this plan.', '',
        'After explicit authorization for this exact list, any remover must recheck local files against the saved hashes and unchanged metadata before removing only those paths. Preserve this plan, the fresh Git proofs and every archive copy. Update the materialization records and handoff only after an authorized removal has actually completed.', '',
        'To restore one listed file from committed Git bytes into a new directory within this worktree:', '',
        '```bash',
        '/home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/older_field_storage.py restore b3d001 --path out/fom_nt1e-02_case0_rep1.npz --destination experiments/paper-b3d/checks/restored-older-b001',
        '```', '',
        'Use repeated `--path` arguments for selected fields or `--all-candidates` for the full listed subset of one attempt. The destination contains paths relative to that attempt’s original `collected` directory. The helper verifies selected file hashes and the entire archive before publishing restored files, and writes `restore-audit.json`. It refuses any existing mismatched destination file.', '',
        'Regenerate this note with `older_field_storage.py describe`. Do not rerun `plan` over the existing manifest; it intentionally refuses to overwrite the audit history.', '',
        '## Glossary', '',
        '- **Candidate:** a verified duplicate proposed for removal, still present locally.',
        '- **Allocated bytes:** actual filesystem blocks occupied by a file; the potential disk-space gain.',
        '- **Logical bytes:** the file length, which can differ slightly from allocated storage.',
        '- **SHA256:** a content checksum used to prove byte-for-byte identity.',
        '- **Actual Git proof:** reconstruction directly from committed archive blobs, independent of local field copies.',
        '- **Archive hard-link alias:** a separately named archived file represented by an earlier entry with exactly identical content.',
        '- **Materialization:** an ordinary local file copy of an archived byte stream.', ''])
    (PLAN.parent / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('plan')
    sub.add_parser('describe')
    rest = sub.add_parser('restore')
    rest.add_argument('attempt', choices=PINS)
    choice = rest.add_mutually_exclusive_group(required=True)
    choice.add_argument('--all-candidates', action='store_true')
    choice.add_argument('--path', action='append')
    rest.add_argument('--destination', required=True)
    args = parser.parse_args()
    if args.mode == 'plan':
        plan()
        describe()
    elif args.mode == 'describe':
        describe()
    else:
        restore(args)


if __name__ == '__main__':
    main()
