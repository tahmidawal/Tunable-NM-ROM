"""Thin fixed-budget entry around the unchanged, separately released train stage."""
from pathlib import Path
import argparse, json
import stage2d
from fe_adapter import sha_file


def verify(candidate_path, release_path, inventory_path):
    release = json.loads(Path(release_path).read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    if (release.get('launcher_sha256') != sha_file(__file__) or
        release.get('runtime_inventory_sha256') != sha_file(inventory_path) or
        inventory.get('schema') != 'ns-obstacle-bank-fit-runtime-inventory-v1' or
        release.get('authorized_stage') != 'train' or
        release.get('allocation_seconds') != 2100 or
        release.get('outer_timeout_seconds') != 2085 or
        release.get('outer_kill_grace_seconds') != 10 or
        release.get('optimizer_time_cap_seconds') != 1800):
        raise ValueError('exact fixed-budget bank launcher release required')
    base = Path.cwd().resolve()
    if str(base) != release.get('job_directory'):
        raise ValueError('released unique bank job directory required')
    seen = set()
    for row in inventory['files']:
        relative = Path(row['destination'])
        path = base/relative
        if (relative.is_absolute() or '..' in relative.parts or str(relative) in seen or
            path.is_symlink() or not path.resolve().is_relative_to(base) or
            path.stat().st_size != row['bytes'] or sha_file(path) != row['sha256']):
            raise ValueError('bank runtime differs from exact release')
        seen.add(str(relative))
    candidate, _, inputs = stage2d.verify(candidate_path, release_path)
    if candidate['stage'] != 'train' or set(inputs) != {'prepared','preparation_release'}:
        raise ValueError('training-only inputs required')
    # The existing stage verifies every prepared file and the existing consumer
    # revalidates the fine-count amendment/native result review before loading.
    return candidate


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    verify(a.candidate, a.release, a.inventory)
    if not a.verify_only:
        stage2d.run(a.candidate, a.release, Path('results/fit'))
