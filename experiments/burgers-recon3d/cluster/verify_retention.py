"""Verify local Git retention before closing this lane; read-only remotely."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parents[1]
TREE = HERE.parents[1]
RESULTS = HERE / 'results'
NS = '/cluster/tufts/paralab/tawal01/b3d-recon-20261027'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(TREE), *args], text=True).strip()


def main():
    retained, blobs = {}, {}
    for name in ('preparation-development', 'validation'):
        proof_path = RESULTS / f'raw-arrays-{name}.json'
        proof = json.loads(proof_path.read_text())
        archive = RESULTS / proof['archive']
        assert sha(archive) == proof['archive_sha256']
        rel = str(archive.relative_to(TREE))
        oid = git('rev-parse', f'HEAD:{rel}')
        assert git('hash-object', '--', rel) == oid
        blobs[rel] = dict(git_object=oid, sha256=proof['archive_sha256'], bytes=archive.stat().st_size)
        for array, entry in proof['archived_arrays'].items():
            assert array not in retained
            retained[array] = dict(**entry, archive=proof['archive'])
        for array, entry in proof['direct_large_arrays'].items():
            path = RESULTS / array
            assert sha(path) == entry['sha256']
            rel = str(path.relative_to(TREE))
            oid = git('rev-parse', f'HEAD:{rel}')
            assert git('hash-object', '--', rel) == oid
            blobs[rel] = dict(git_object=oid, **entry)
            assert array not in retained
            retained[array] = dict(**entry, direct_git_path=rel)
    expected = {}
    for attempt in ('smk01', 'prep01', 'dev02', 'val01'):
        for line in (RESULTS / attempt / 'OUTPUTS.sha256').read_text().splitlines():
            digest, rel = line.split(None, 1)
            if Path(rel).suffix in ('.npy', '.npz'):
                expected[f'{attempt}/{rel.strip()}'] = digest
    assert set(expected) == set(retained)
    assert all(retained[p]['sha256'] == h for p, h in expected.items())
    inventory = subprocess.check_output(['ssh', 'tufts-login',
        f'find {NS} -mindepth 1 -maxdepth 1 -printf "%f\\n"'], text=True)
    assert not inventory.strip(), inventory
    for attempt in ('prep01', 'dev02', 'val01'):
        receipt = json.loads((RESULTS / attempt / 'RETURN.json').read_text())
        assert receipt['verified'] and receipt['remote_cleaned']
    out = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        archived_at_commit=git('rev-parse', 'HEAD'), arrays=len(retained),
        all_returned_arrays_preserved=True, expanded_arrays_retained=True,
        remote_namespace=NS, remote_namespace_empty=True,
        remote_unique_regenerable_training_matrices='Deleted only after compact training provenance, original bases and every required raw array were verified locally; original bank snapshot byte identity remains unavailable.',
        git_blobs=blobs, arrays_manifest=retained,
        mirror_scope='Code-only GitHub branch preserves source, measurements and hashes; large binary arrays/archive blobs remain on the local experiment branch.')
    (RESULTS / 'RETENTION.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('git_blobs', 'arrays_manifest')}, indent=2))


if __name__ == '__main__':
    main()
