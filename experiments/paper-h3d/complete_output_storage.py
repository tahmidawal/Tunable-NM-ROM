"""Release only verified field NPZ duplicates from a complete-output archive."""
import argparse
import datetime
import json
import subprocess
import tarfile
from pathlib import Path
import retain_fields as R


def field_member(name):
    parts = Path(name).parts
    return (len(parts) == 2 and parts[0] == 'fields' or
            len(parts) == 3 and parts[:2] == ('seedB', 'fields')) and name.endswith('.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('attempt')
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prune', action='store_true')
    action.add_argument('--restore', action='store_true')
    p.add_argument('--member')
    args = p.parse_args()
    assert args.attempt.isalnum() and (not args.member or args.restore)
    lane = Path(__file__).resolve().parent
    root = lane.parents[1]
    run = lane/'runs'/args.attempt
    archive = lane/'retained-fields'/args.attempt
    source = run/'archive/out'
    manifest = json.loads((archive/'manifest.json').read_text())
    assert lane/manifest['source_relative'] == source
    proof = json.loads((archive/'git-retention-audit.json').read_text())
    assert proof['passed'] and proof['actual_git_blob_bytes_verified']
    collected = json.loads((run/'COLLECTED.json').read_text())
    assert collected['removed'] and collected['actual_git_blob_bytes_verified']
    files = {r['path']: r for r in manifest['files']}
    selected = {n: r for n, r in files.items() if field_member(n)}
    assert selected and all((source/n).resolve().is_relative_to(source.resolve()) for n in selected)
    marker = run/'RAW-FIELD-STORAGE.json'
    if args.restore:
        record = json.loads(marker.read_text())
        assert record['archive_manifest_sha256'] == R.sha(archive/'manifest.json')
        assert record['actual_git_archive_commit'] == proof['commit']
        assert {r['path']: r for r in record['files']} == selected
        if args.member:
            assert args.member in selected
            selected = {args.member: selected[args.member]}
    else:
        assert not marker.exists(), 'Preserve the existing storage journal; do not overwrite it.'
    tracked = set(subprocess.check_output(['git', 'ls-files', '--', str(source.relative_to(root))],
                                        cwd=root, text=True).splitlines())
    assert all((source/n).relative_to(root).as_posix() not in tracked for n in selected)
    # Both complete archive representations are checked before any raw mutation.
    R.verify_git(archive, proof['commit'])
    R.verify(archive)
    if args.prune:
        for name, row in selected.items():
            f = source/name
            assert f.is_file() and f.stat().st_size == row['bytes'] and R.sha(f) == row['sha256']
        allocated = sum((source/n).stat().st_blocks*512 for n in selected)
        record = dict(action='prune_verified_complete_output_field_duplicates', complete=False,
            date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            archive_manifest_sha256=R.sha(archive/'manifest.json'), actual_git_archive_commit=proof['commit'],
            source_relative=manifest['source_relative'], files=list(selected.values()),
            bytes=sum(r['bytes'] for r in selected.values()), allocated_bytes=allocated,
            selection='Only direct fields/*.npz and seedB/fields/*.npz; all checkpoints, metadata, archives and Git bytes remain.',
            restore_command=f'/home/tahmid/Dev/.venv/bin/python experiments/paper-h3d/complete_output_storage.py {args.attempt} --restore')
        marker.write_text(json.dumps(record, indent=2)+'\n')
        for name in selected:
            (source/name).unlink()
        record['complete'] = True
        marker.write_text(json.dumps(record, indent=2)+'\n')
        print('VERIFIED_FIELD_DUPLICATES_RELEASED', len(selected), allocated)
    else:
        restored = []
        with tarfile.open(fileobj=R.JoinReader([archive/r['path'] for r in manifest['chunks']]), mode='r|') as stream:
            for member in stream:
                if member.name not in selected:
                    continue
                assert member.isfile()
                f = source/member.name
                if not f.exists():
                    f.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as incoming, f.open('xb') as target:
                        for block in iter(lambda: incoming.read(8*1024*1024), b''):
                            target.write(block)
                assert f.stat().st_size == selected[member.name]['bytes'] and R.sha(f) == selected[member.name]['sha256']
                restored.append(member.name)
        assert set(restored) == set(selected)
        print('VERIFIED_FIELD_MEMBERS_RESTORED', len(restored))


if __name__ == '__main__':
    main()
