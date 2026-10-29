"""Released two-case development reconstruction; no fitting or calibration."""
from pathlib import Path
import argparse, json, os
import handoff2d
from fe_adapter import sha_file


def verify(plan_path, release_path, inventory_path):
    release = json.loads(Path(release_path).read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    if (release.get('launcher_sha256') != sha_file(__file__) or
        release.get('input_inventory_sha256') != sha_file(inventory_path) or
        release.get('allocation_seconds') != 120 or
        release.get('outer_timeout_seconds') != 110 or
        release.get('outer_kill_grace_seconds') != 5 or
        release.get('remaining_development_reservation_seconds') != 3480 or
        release.get('split') != 'development'):
        raise ValueError('exact bounded development-only release required')
    root = Path.cwd().resolve()
    if str(root) != release.get('job_directory'):
        raise ValueError('released unique development directory required')
    seen = set()
    for row in inventory['files']:
        relative = Path(row['destination']);path = root/relative
        if (relative.is_absolute() or '..' in relative.parts or str(relative) in seen or
            path.is_symlink() or not path.resolve().is_relative_to(root) or
            path.stat().st_size != row['bytes'] or sha_file(path) != row['sha256']):
            raise ValueError('exact development runtime/input differs')
        seen.add(str(relative))
    plan, _, truth, _ = handoff2d.verify(plan_path, release_path)
    if plan['split'] != 'development' or [r['Re'] for r in plan['cases']] != [16.875, 23.125]:
        raise ValueError('only the two registered development cases')
    for row in plan['cases']:
        adopted = handoff2d.adopted_case(truth, row['Re'], row['directory'])
        if adopted['split'] != 'development' or adopted['storage_mode'] != 'coefficient_primary':
            raise ValueError('actual admitted development primary required')
    return release


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('plan', 'release', 'inventory'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--verify-only', action='store_true');a=p.parse_args()
    release=verify(a.plan,a.release,a.inventory)
    if not a.verify_only:
        with Path('DEVELOPMENT-HANDOFF-CLAIM.json').open('x') as f:
            json.dump({'release_sha256':sha_file(a.release),'job_id':os.environ.get('SLURM_JOB_ID')},f)
        handoff2d.run(a.plan,a.release,Path(release['output_directory']))
