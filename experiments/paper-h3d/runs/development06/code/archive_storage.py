"""Reclaim only byte-verified raw duplicates; restore from committed split archives.

The working-tree split archive and its actual Git blobs remain untouched.
"""
import argparse,datetime,json,subprocess,tarfile
from pathlib import Path
import retain_fields as R


def require_fields(out):
    """Fail clearly instead of silently skipping intentionally pruned raw fields."""
    out=Path(out);marker=out.parent.parent/'RAW-FIELD-STORAGE.json'
    if not marker.exists():return
    storage=json.loads(marker.read_text())
    source=Path(__file__).resolve().parent/storage['source_relative']
    missing=[row['path'] for row in storage['files'] if not (source/row['path']).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} raw field duplicates are archived in verified Git storage. Restore before this audit: {storage['restore_command']}")


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--prune',action='store_true');parser.add_argument('--restore',action='store_true');parser.add_argument('--member')
    args=parser.parse_args();assert args.attempt.isalnum() and args.prune!=args.restore
    lane=Path(__file__).resolve().parent;root=lane.parents[1];archive=lane/'retained-fields'/args.attempt
    manifest=json.loads((archive/'manifest.json').read_text());source=lane/manifest['source_relative']
    assert source.resolve().is_relative_to((lane/'runs'/args.attempt/'archive/out').resolve())
    proof=json.loads((archive/'git-retention-audit.json').read_text());assert proof['passed']
    files={row['path']:row for row in manifest['files']}
    for name in files:assert (source/name).resolve().is_relative_to(source.resolve())
    if args.member:assert args.restore and args.member in files
    R.verify_git(archive,proof['commit']);R.verify(archive)
    tracked=subprocess.check_output(['git','ls-files','--',str(source.relative_to(root))],cwd=root,text=True).splitlines()
    assert not tracked,'raw pruning/restoration must not alter tracked paths'
    if args.prune:
        for name,row in files.items():
            path=source/name
            assert path.is_file() and path.stat().st_size==row['bytes'] and R.sha(path)==row['sha256'],path
        # Journal the exact recoverable content before removing any duplicate.
        record=dict(action='prune_verified_raw_duplicates',date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            archive_manifest_sha256=R.sha(archive/'manifest.json'),actual_git_archive_commit=proof['commit'],
            source_relative=manifest['source_relative'],files=list(files.values()),
            bytes=sum(row['bytes'] for row in files.values()),complete=False,
            restore_command=f'/home/tahmid/Dev/.venv/bin/python experiments/paper-h3d/archive_storage.py {args.attempt} --restore')
        log=lane/'runs'/args.attempt/'RAW-FIELD-STORAGE.json';log.write_text(json.dumps(record,indent=2)+'\n')
        for name in files:(source/name).unlink()
        record['complete']=True;log.write_text(json.dumps(record,indent=2)+'\n')
        print(args.attempt,'PRUNED_VERIFIED_DUPLICATE_BYTES',record['bytes'],flush=True)
    else:
        wanted={args.member} if args.member else set(files);restored=[]
        reader=R.JoinReader([archive/row['path'] for row in manifest['chunks']])
        with tarfile.open(fileobj=reader,mode='r|') as stream:
            for member in stream:
                if member.name not in wanted:continue
                assert member.isfile();path=source/member.name;row=files[member.name]
                if path.exists():assert path.stat().st_size==row['bytes'] and R.sha(path)==row['sha256']
                else:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with stream.extractfile(member) as incoming,path.open('xb') as target:
                        for block in iter(lambda:incoming.read(8*1024*1024),b''):target.write(block)
                    assert path.stat().st_size==row['bytes'] and R.sha(path)==row['sha256']
                restored.append(member.name)
        assert set(restored)==wanted
        print(args.attempt,'RESTORED_VERIFIED_MEMBERS',len(restored),flush=True)


if __name__=='__main__':main()
