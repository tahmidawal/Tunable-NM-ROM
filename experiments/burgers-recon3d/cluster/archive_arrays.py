"""Losslessly archive small raw arrays; large bases remain direct Git blobs.

The code-only mirror omits files above its size limit. Raw local artifacts are
kept in place, and this manifest identifies every locally archived array.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

HERE=Path(__file__).resolve().parents[1]
LIMIT=50*1024**2


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('attempts',nargs='+');a=p.parse_args()
    assert re.fullmatch('[a-zA-Z0-9-]+',a.label)
    assert all(re.fullmatch('[a-zA-Z0-9]+',s) for s in a.attempts)
    root=HERE/'results';archive=root/f'raw-arrays-{a.label}.tar.gz';proof=root/f'raw-arrays-{a.label}.json'
    assert not archive.exists() and not proof.exists()
    small,large={},{}
    for attempt in a.attempts:
        job=root/attempt
        manifest={rel.strip():checksum for checksum,rel in
                  (line.split(None,1) for line in (job/'OUTPUTS.sha256').read_text().splitlines())}
        for rel,checksum in manifest.items():
            path=job/rel
            if path.suffix not in ('.npy','.npz'):continue
            entry=dict(sha256=checksum,bytes=path.stat().st_size)
            (large if entry['bytes']>LIMIT else small)[f'{attempt}/{rel}']=entry
    temporary=archive.with_suffix('.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for name,entry in sorted(small.items()):
                    path=root/name
                    assert sha(path)==entry['sha256'],name
                    info=tar.gettarinfo(str(path),arcname=name)
                    info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                    with path.open('rb') as stream:tar.addfile(info,stream)
    temporary.replace(archive)
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(small)
        for member in tar:
            h=hashlib.sha256();stream=tar.extractfile(member)
            for block in iter(lambda:stream.read(16*1024**2),b''):h.update(block)
            assert h.hexdigest()==small[member.name]['sha256'],member.name
    for name,entry in large.items():assert sha(root/name)==entry['sha256'],name
    result=dict(archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        archived_arrays=small,direct_large_arrays=large,all_arrays_preserved=True,
        restore='Extract this tar.gz inside the lane results directory; large basis files are stored directly on the local experiment branch.',
        mirror='Code-only mirror omits archive/base blobs above50MiB; checksum manifests remain in the mirror.')
    proof.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(archive=str(archive),manifest=str(proof),small_arrays=len(small),large_arrays=len(large),
                         archive_bytes=result['archive_bytes'],direct_large_bytes=sum(v['bytes'] for v in large.values())),indent=2))


if __name__=='__main__':main()
