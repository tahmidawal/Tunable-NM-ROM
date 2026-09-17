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
    source = ROOT / 'experiments/b-seeds/runs' / attempt / 'archive'
    out = ROOT / 'experiments/b-seeds/artifacts' / attempt
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
    for sub, name in (('ladder_seed', 'result-ladder-seed.json'), ('ladder_incumbent', 'result-ladder-incumbent.json'),
                      ('eqcert_seed', 'result-eqcert-seed.json'), ('train', None)):
        p = source / 'output' / sub / 'result.json'
        if name and p.exists():
            shutil.copy2(p, out / name)
    for name in ('sep_burgers_r3_N256_K16_R512.json', 'sep_coeff_N256_K16_R512.json', 'hfit_full.json'):
        p = source / 'output/train' / name
        if p.exists():
            shutil.copy2(p, out / f'train-{name}')
    for p in sorted((source / 'output').glob('sealed_*/result.json')):
        shutil.copy2(p, out / f'result-{p.parent.name}.json')
    for extra in ('audit.json', 'EXCLUDED-FROM-ARCHIVE.txt', 'EXCLUDED-SHA256.txt'):
        if (source.parent / extra).exists():
            shutil.copy2(source.parent / extra, out / extra)
        elif (source / extra).exists():
            shutil.copy2(source / extra, out / extra)
    (out / 'README.md').write_text('''# Full raw b-seeds archive

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

The extraction npz is deliberately NOT in these chunks; see
`EXCLUDED-FROM-ARCHIVE.txt` and `EXCLUDED-SHA256.txt`. `result.json` here is the
SEED LADDER result; the other result files live inside the archive under `output/`.
''')
    print(out)


if __name__ == '__main__':
    main()
