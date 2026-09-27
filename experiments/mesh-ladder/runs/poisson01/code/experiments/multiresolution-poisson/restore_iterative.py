"""Restore ignored Poisson scientific fields from committed checksum parts."""
import argparse,hashlib,json,tarfile
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):value.update(block)
    return value.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();run=args.run.resolve()
    manifest=json.loads((run/'ARCHIVE.json').read_text());archive=run/manifest['archive_name']
    if not archive.exists():
        temporary=archive.with_suffix(archive.suffix+'.partial')
        with temporary.open('xb') as output:
            for part in manifest['ordered_parts']:
                path=run/part['name'];assert path.stat().st_size==part['bytes'] and digest(path)==part['sha256']
                output.write(path.read_bytes())
        assert temporary.stat().st_size==manifest['bytes'] and digest(temporary)==manifest['archive_sha256']
        temporary.rename(archive)
    assert digest(archive)==manifest['archive_sha256']
    if not (run/'cluster').exists():
        with tarfile.open(archive,'r:gz') as tar:tar.extractall(run,filter='data')
    print(run/'cluster')


if __name__=='__main__':main()
