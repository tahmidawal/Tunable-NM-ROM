"""Split a verified compressed raw archive into Git-trackable bounded chunks (48 MB).

    python experiments/w-ladder/cluster/preserve_archive.py <attempt>

Copied from experiments/q-ridge/cluster/preserve_archive.py. The extracted archive directory
itself is ignored by Git (large fields); the chunks, result.json and manifests are tracked.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    attempt = sys.argv[1]
    assert attempt.isalnum()
    source = ROOT / 'experiments/w-ladder/runs' / attempt / 'archive'
    out = ROOT / 'experiments/w-ladder/artifacts' / attempt
    out.mkdir(parents=True, exist_ok=False)
    records, whole = [], hashlib.sha256()
    with (source / 'collection.tar.gz').open('rb') as stream:
        index = 0
        while block := stream.read(48 * 1024 * 1024):
            name = f'collection.tar.gz.part{index:04d}'
            (out / name).write_bytes(block)
            records.append(dict(path=name, bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
            whole.update(block); index += 1
    expected = (source / 'collection.tar.gz.sha256').read_text().split()[0]
    assert whole.hexdigest() == expected
    (out / 'archive.json').write_text(json.dumps(dict(archive='collection.tar.gz', sha256=expected, chunks=records), indent=2) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records))
    for name in ['COMMIT.txt', 'PROVENANCE.json', 'MANIFEST.sha256', 'OUTPUTS.sha256', 'run.sbatch', 'VERIFIED.txt']:
        if (source / name).exists():
            shutil.copy2(source / name, out / name)
    for name in ['result.json', 'training_ladder_dirichlet.npz']:
        if (source / 'output' / name).exists():
            shutil.copy2(source / 'output' / name, out / name)
    for log in sorted((source / 'logs').glob('*')):
        shutil.copy2(log, out / log.name)
    for extra in ('accounting.txt', 'audit.json', 'cleanup.txt'):
        if (source.parent / extra).exists():
            shutil.copy2(source.parent / extra, out / extra)
    (out / 'README.md').write_text('''# Full raw w-ladder archive

The ordered chunks preserve the checksum-verified original compressed job archive: every
numerical array, the immutable staged source, logs and manifests.

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored && sha256sum -c OUTPUTS.sha256 && sha256sum -c MANIFEST.sha256
```

`result.json` beside this file is the uncompressed copy the report generator reads; the
whole-archive checksum is in `archive.json`.
''')
    print(out, len(records), 'chunks')


if __name__ == '__main__':
    main()
