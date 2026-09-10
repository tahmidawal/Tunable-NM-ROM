"""Preserve and verify collected raw evidence before its remote copy is removed."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def digest(stream):
    h=hashlib.sha256()
    while block:=stream.read(8*1024*1024):h.update(block)
    return h.hexdigest()


root=Path(sys.argv[1]).resolve()
archive=root.parent.parent/'raw-archives'/f'{root.name}.tar'
archive.parent.mkdir(parents=True,exist_ok=True)
assert not archive.exists(),f'archive already exists: {archive}'
expected={}
for line in (root/'COLLECT.sha256').read_text().splitlines():
    checksum,path=line.split('  ',1);expected[f'{root.name}/{path}']=checksum
with tarfile.open(archive,'w') as tar:tar.add(root,arcname=root.name)
verified=0
with tarfile.open(archive,'r') as tar:
    for member in tar:
        if member.name in expected:
            with tar.extractfile(member) as f:actual=digest(f)
            assert actual==expected.pop(member.name),f'archive member mismatch: {member.name}'
            verified+=1
assert not expected,f'archive missing {len(expected)} raw members'
with archive.open('rb') as f:checksum=digest(f)
record=dict(archive_path=str(archive),sha256=checksum,bytes=archive.stat().st_size,
            verified_members=verified,collect_manifest_sha256=hashlib.sha256((root/'COLLECT.sha256').read_bytes()).hexdigest(),
            extraction=f'tar -xf {archive} -C <destination>; cd <destination>/{root.name}; sha256sum -c COLLECT.sha256',
            policy='Full raw fields and references remain in this checksum-verified archive; tracked JSON and checkpoints can be reviewed without extraction.')
(root/'RAW_ARCHIVE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
