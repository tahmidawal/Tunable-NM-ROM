"""Track verified chunks of oversized collected files without altering archives."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();assert a.attempt.isalnum()
    lane=Path(__file__).resolve().parents[1];root=lane/'runs'/a.attempt
    assert json.loads((root/'COLLECTED.json').read_text())['checksums_verified']
    assert json.loads((root/'audit-local.json').read_text())['passed']
    ignores=[];retained=[]
    for source in sorted((root/'archive').rglob('*')):
        if not source.is_file() or source.stat().st_size<128*1024*1024:continue
        relative=source.relative_to(root);destination=root/'large-artifacts'/str(relative).replace('/','__')
        destination.mkdir(parents=True,exist_ok=False);chunks=[]
        with source.open('rb') as stream:
            for i in range(100000):
                data=stream.read(64*1024*1024)
                if not data:break
                path=destination/f'{source.name}.part{i:03d}';path.write_bytes(data)
                chunks.append(dict(path=path.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        manifest=dict(original_path=str(relative),original_bytes=source.stat().st_size,
                      original_sha256=digest(source),chunks=chunks)
        (destination/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        ignores.append('/'+str(relative));retained.append(manifest)
    if retained:
        shutil.copy2(lane/'runs/tuned02/large-artifacts/restore.py',root/'large-artifacts/restore.py')
        ignore=root/'.gitignore';existing=ignore.read_text().splitlines() if ignore.exists() else []
        ignore.write_text('\n'.join(sorted(set(existing+ignores)))+'\n')
    (root/'large-artifact-retention.json').write_text(json.dumps(dict(original_files_unchanged=True,files=retained),indent=2)+'\n')


if __name__=='__main__':main()
