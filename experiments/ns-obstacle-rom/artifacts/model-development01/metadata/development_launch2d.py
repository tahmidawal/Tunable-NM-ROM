"""Thin frozen-bank development entry; no bank retraining, validation, or budget reset."""
from pathlib import Path
import argparse, json
import stage2d
from fe_adapter import sha_file


def verify(candidate_path, release_path, inventory_path):
    release = json.loads(Path(release_path).read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    if (release.get('launcher_sha256') != sha_file(__file__) or
        release.get('runtime_inventory_sha256') != sha_file(inventory_path) or
        inventory.get('schema') != 'ns-obstacle-development-runtime-inventory-v1' or
        release.get('authorized_stage') != 'development' or
        release.get('allocation_seconds') != 3360 or
        release.get('outer_timeout_seconds') != 3345 or
        release.get('outer_kill_grace_seconds') != 10 or
        release.get('original_development_reservation_seconds') != 3600 or
        release.get('prior_materialization_reservation_seconds') != 240):
        raise ValueError('exact remaining-budget development release required')
    base = Path.cwd().resolve()
    if str(base) != release.get('job_directory'):
        raise ValueError('released unique development job directory required')
    seen = set()
    for row in inventory['files']:
        relative = Path(row['destination'])
        path = base/relative
        if (relative.is_absolute() or '..' in relative.parts or str(relative) in seen or
            path.is_symlink() or not path.resolve().is_relative_to(base) or
            path.stat().st_size != row['bytes'] or sha_file(path) != row['sha256']):
            raise ValueError('development runtime differs from exact release')
        seen.add(str(relative))
    candidate, _, inputs = stage2d.verify(candidate_path, release_path)
    if candidate['stage'] != 'development' or set(inputs) != stage2d.INPUTS['development']:
        raise ValueError('development-only inputs required')
    # The existing stage verifies every prepared file and the existing consumer
    # revalidates the fine-count amendment/native result review before loading.
    from pilot2d import read_preparation_metadata
    metadata, inner = read_preparation_metadata(inputs['prepared'], inputs['preparation_release'])
    for key, field in [('bank','bank_sha256'), ('probes','probes_sha256'),
                       ('development','development_manifest_sha256')]:
        if inner.get(field) != sha_file(inputs[key]):
            raise ValueError('development inner-release binding missing or changed: '+field)
    population=json.loads(Path(inputs['development']).read_text())
    if population.get('reference_admission_sha256') != metadata['admission_evidence']['reference_admission_sha256']:
        raise ValueError('development and preparation truth identities differ')
    expected = {'bank_model': ('bank_sha256', 'bank'),
                'development_population': ('manifest_sha256', 'development')}
    if set(release.get('review_bindings', {})) != set(expected):
        raise ValueError('actual model and development-population reviews required')
    for key, (field, input_key) in expected.items():
        descriptor=release['review_bindings'][key];path=base/descriptor['path']
        if (path.is_symlink() or not path.resolve().is_relative_to(base) or
            sha_file(path)!=descriptor['sha256']):
            raise ValueError('review receipt differs')
        review=json.loads(path.read_text())
        if review.get('passed') is not True or review.get(field)!=sha_file(inputs[input_key]):
            raise ValueError('review does not bind this actual development input')
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
        stage2d.run(a.candidate, a.release, Path('results/development-stage'))
