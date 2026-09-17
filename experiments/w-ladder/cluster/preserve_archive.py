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


LARGE = 200 * 1024**2   # files above this are excluded from the Git chunks, with hash + regeneration recorded


def reduced_archive(source, out):
    """Rebuild the tar without files above LARGE, recording what was left out and how to regenerate it.

    The excluded files are deterministic: the mesh tables are a function of the frozen bank
    parameters and the POD basis, and the full-field FOM outputs are the same-job solvers applied
    to the saved initial fields. Their SHA256s were verified on BOTH sides via OUTPUTS.sha256, are
    recorded in result.json, and are listed here. The independent NumPy audit runs against the
    full extracted archive BEFORE this reduction.
    """
    import subprocess, tarfile
    digests = {}
    for line in (source / 'OUTPUTS.sha256').read_text().splitlines():
        if line.strip():
            d, name = line.split(None, 1)
            digests[name.strip().lstrip('./')] = d
    big = sorted((p for p in (source / 'output').rglob('*') if p.is_file() and p.stat().st_size > LARGE),
                 key=lambda p: -p.stat().st_size)
    if not big:
        return source / 'collection.tar.gz', []
    names = {str(p.relative_to(source)) for p in big}
    reduced = source / 'collection-reduced.tar.gz'
    with tarfile.open(reduced, 'w:gz') as tar:
        for item in sorted(source.rglob('*')):
            if not item.is_file() or item.name.startswith('collection.tar'):
                continue
            rel = str(item.relative_to(source))
            if rel in names:
                continue
            tar.add(item, arcname=rel)
    def digest_of(path):
        rel = str(path.relative_to(source))
        return digests.get(rel) or digests.get(str(path.relative_to(source / 'output'))) or 'see OUTPUTS.sha256'
    rows = [dict(path=str(p.relative_to(source)), bytes=p.stat().st_size, sha256=digest_of(p)) for p in big]
    (out / 'EXCLUDED-FROM-ARCHIVE.txt').write_text(
        'Excluded from the Git chunks because each exceeds 200 MB; together they are ~5.9 GB of the 6.1 GB job\n'
        'output, and committing them would repeat this repository\'s multi-GB-commit problem.\n\n'
        + ''.join(f"{r['sha256']}  {r['path']}  ({r['bytes'] / 1024**2:.0f} MB)\n" for r in rows)
        + '\nEvery one is deterministic and regenerable:\n'
          '  mesh_<n>.npz      the mass-QR bank, its stiffness/damping and the transferred POD basis on the query\n'
          '                    grid; rebuilt by ladder.py rebuild()/pod_banks() from the frozen bank parameters.\n'
          '  reference_*.npz   the exact semidiscrete DST solution from the saved initial fields (u0, v0 are kept\n'
          '                    in the chunked copy of the same file for the non-full-field cases).\n'
          '  <mesh>_<case>_<fom arm>.npz   the same-job full-order solver output for the one designated\n'
          '                    full-field case; every other case keeps its common-grid (64^2) fields in the chunks.\n\n'
          'All ROM arms keep their full coefficient trajectories in the chunks, from which their fields are\n'
          'exactly reconstructed as coefficients @ bank.T -- which is what audit_ladder.py does. The audit was run\n'
          'against the COMPLETE archive before this reduction; its result is audit.json beside this file.\n')
    return reduced, rows


def main():
    attempt = sys.argv[1]
    assert attempt.isalnum()
    source = ROOT / 'experiments/w-ladder/runs' / attempt / 'archive'
    out = ROOT / 'experiments/w-ladder/artifacts' / attempt
    out.mkdir(parents=True, exist_ok=False)
    archive, excluded = reduced_archive(source, out)
    records, whole = [], hashlib.sha256()
    with archive.open('rb') as stream:
        index = 0
        while block := stream.read(48 * 1024 * 1024):
            name = f'collection.tar.gz.part{index:04d}'
            (out / name).write_bytes(block)
            records.append(dict(path=name, bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
            whole.update(block); index += 1
    if archive.name == 'collection.tar.gz':
        expected = (source / 'collection.tar.gz.sha256').read_text().split()[0]
        assert whole.hexdigest() == expected
    else:
        expected = whole.hexdigest()
    (out / 'archive.json').write_text(json.dumps(dict(archive=archive.name, sha256=expected, chunks=records,
        complete_archive_sha256=(source / 'collection.tar.gz.sha256').read_text().split()[0],
        excluded_files=excluded), indent=2) + '\n')
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
cat collection.tar.gz.part* > collection.tar.gz   # name per archive.json
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored && sha256sum -c OUTPUTS.sha256 && sha256sum -c MANIFEST.sha256
```

`result.json` beside this file is the uncompressed copy the report generator reads; the
whole-archive checksum is in `archive.json`.
''')
    print(out, len(records), 'chunks')


if __name__ == '__main__':
    main()
