"""Split a verified compressed raw archive into Git-trackable bounded chunks."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    attempt = sys.argv[1]
    assert attempt.isalnum()
    source = ROOT / 'experiments/head-refine/runs' / attempt / 'archive'
    out = ROOT / 'experiments/head-refine/artifacts' / attempt
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
                 'run.sbatch', 'scheduler.txt']:
        if (source / name).exists():
            shutil.copy2(source / name, out / name)
    if (source / 'output/result.json').exists():
        shutil.copy2(source / 'output/result.json', out / 'result.json')
    for extra in ('audit.json',):
        if (source.parent / extra).exists():
            shutil.copy2(source.parent / extra, out / extra)
    (out / 'README.md').write_text('''# Full raw head-refinement archive

The ordered chunks preserve the checksum-verified original compressed job archive,
including every numerical array, the immutable staged source, logs and manifests.

From this directory, verify and restore into a fresh destination:

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored
tar -xzf collection.tar.gz -C restored
cd restored
sha256sum -c OUTPUTS.sha256
sha256sum -c MANIFEST.sha256
```

`result.json` beside this file is the uncompressed copy the report generator reads.
The whole-archive checksum is recorded in `archive.json`.
''')
    print(out)


if __name__ == '__main__':
    main()
