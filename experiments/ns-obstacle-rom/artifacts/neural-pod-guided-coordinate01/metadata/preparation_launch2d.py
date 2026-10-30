"""Thin exact-packet guard for the existing native preparation entry point."""
from pathlib import Path
import argparse, hashlib, json, os, time
from prepare2d import preparation_digest, verify_release, prepare


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(release_path, inventory_path):
    release_path = Path(release_path)
    release = json.loads(release_path.read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    if (release.get('launcher_sha256') != sha(__file__) or
        release.get('input_inventory_sha256') != sha(inventory_path) or
        inventory.get('schema') != 'ns-obstacle-fine-preparation-input-inventory-v1' or
        inventory.get('preparation_source_sha256') != preparation_digest() or
        release.get('single_execution') is not True or
        release.get('allocation_seconds') != 1350 or
        release.get('outer_timeout_seconds') != 1335 or
        release.get('outer_kill_grace_seconds') != 10):
        raise ValueError('exact bounded fine-preparation launcher release required')
    base = Path.cwd().resolve()
    if str(base) != release.get('job_directory'):
        raise ValueError('unique released job directory required')
    names = set()
    for row in inventory['files']:
        path = base / row['destination']
        if (Path(row['destination']).is_absolute() or '..' in Path(row['destination']).parts or
            row['destination'] in names or path.is_symlink() or not path.resolve().is_relative_to(base) or
            path.stat().st_size != row['bytes'] or sha(path) != row['sha256']):
            raise ValueError('exact preparation input differs: ' + row['destination'])
        names.add(row['destination'])
    manifest = base / inventory['training_manifest']
    verify_release(manifest, release_path)
    if release.get('point_block_volume') is not True or release.get('verify_primary_volume') is not True:
        raise ValueError('complete original and restored fine replay required')
    return release, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    release, manifest = verify(args.release, args.inventory)
    if args.verify_only:
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('released cluster allocation required')
    claim = Path('PREPARATION-CLAIM.json')
    with claim.open('x') as stream:
        json.dump({'release_sha256': sha(args.release), 'job_id': os.environ['SLURM_JOB_ID'],
                   'started_unix': time.time(), 'automatic_retry': False}, stream, indent=2)
    prepare(manifest, args.release, Path('results/preparation'),
            point_block_volume=True, verify_primary_volume=True)


if __name__ == '__main__':
    main()
