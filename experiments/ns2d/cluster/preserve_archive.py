"""Split a verified compressed raw archive into Git-trackable bounded chunks (48 MB)."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    attempt = sys.argv[1]
    assert attempt.isalnum()
    source = ROOT / 'experiments/ns2d/runs' / attempt / 'archive'
    out = ROOT / 'experiments/ns2d/artifacts' / attempt
    out.mkdir(parents=True, exist_ok=False)
    records = []
    whole = hashlib.sha256()
    with (source / 'collection.tar.gz').open('rb') as stream:
        index = 0
        while block := stream.read(48 * 1024 * 1024):
            name = f'collection.tar.gz.part{index:04d}'
            (out / name).write_bytes(block)
            records.append(dict(path=name, bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
            whole.update(block)
            index += 1
    expected = (source / 'collection.tar.gz.sha256').read_text().split()[0]
    assert whole.hexdigest() == expected
    (out / 'archive.json').write_text(json.dumps(
        dict(archive='collection.tar.gz', sha256=expected, chunks=records), indent=2) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records))
    for name in ['COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256', 'OUTPUTS.sha256',
                 'run.sbatch', 'EXCLUDED-FROM-ARCHIVE.txt']:
        if (source / name).exists():
            shutil.copy2(source / name, out / name)
    res = source / 'experiments/ns2d/output/result.json'
    if res.exists():
        shutil.copy2(res, out / 'result.json')
    for extra in ('audit.json',):
        if (source.parent / extra).exists():
            shutil.copy2(source.parent / extra, out / extra)
    (out / 'README.md').write_text('''# Full raw ns2d archive

The ordered chunks preserve the checksum-verified original compressed job archive.

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored && sha256sum -c OUTPUTS.sha256 && sha256sum -c MANIFEST.sha256
```

`result.json` beside this file is the uncompressed copy the report generator reads.
''')
    print(out)


if __name__ == '__main__':
    main()
