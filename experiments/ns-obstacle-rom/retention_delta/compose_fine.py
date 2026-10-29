"""Build a reversible compact view of the already audited fine preparation.

No source file is removed or changed. Existing lossless delta retention stores
all files in the view; the separate composition map identifies derived volumes.
"""
from pathlib import Path
import argparse, json, os, shutil
from delta_retention import sha, dump, write_delta

AUDIT_SHA = '7643e5ca4235f7d756b3a265c86198f6e9cef45d7dc89d57cc317d0570c0c214'
NUMERICAL_SHA = 'c2c4b6d82138221f330442bbb420dd29873b17328e6c1402183181dfa7a8bfd6'


def compose(source, destination, audit_path, numerical_path, index, archive):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('fresh independent reversible view required')
    if sha(audit_path) != AUDIT_SHA or sha(numerical_path) != NUMERICAL_SHA:
        raise ValueError('exact independent fine reviews required')
    audit = json.loads(Path(audit_path).read_text())
    if not audit['passed'] or audit['omission_authorized'] is not False:
        raise ValueError('candidate composition cannot authorize omission')
    prefixes = ('results/preparation/prepared/volume/', 'results/preparation/restored_volume/')
    expected_volume = dict(audit['volume_files'])
    manifest = source / 'results/preparation/prepared/volume/manifest.json'
    expected_volume['manifest.json'] = {'sha256': audit['volume_manifest_sha256'], 'bytes': manifest.stat().st_size}
    omitted, retained, links = {}, {}, {}
    for path in sorted(source.rglob('*')):
        name = str(path.relative_to(source))
        if path.is_symlink():
            target = os.readlink(path)
            if not (path.parent / target).resolve().is_relative_to(source) or Path(target).is_absolute():
                raise ValueError('unexpected external/absolute symlink: ' + name)
            links[name] = target
        elif path.is_file():
            row = {'sha256': sha(path), 'bytes': path.stat().st_size}
            (omitted if any(name.startswith(p) for p in prefixes) else retained)[name] = row
        elif not path.is_dir():
            raise ValueError('unexpected special file: ' + name)
    expected_omitted = {prefix + name: row for prefix in prefixes for name, row in expected_volume.items()}
    if omitted != expected_omitted:
        raise ValueError('actual omitted file set differs from complete independent replay')
    actual_science = {name.removeprefix('results/preparation/'): row for name, row in retained.items()
                      if name.startswith('results/preparation/')}
    if actual_science != audit['compact_scientific_files']:
        raise ValueError('compact scientific closure differs from independently accepted 45 files')
    destination.mkdir()
    for name in retained:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source / name, target)
    for name, target in links.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    shutil.copyfile(audit_path, destination / 'PRIMARY-REVIEW.json')
    shutil.copyfile(numerical_path, destination / 'NUMERICAL-REVIEW.json')
    numerical = json.loads(Path(numerical_path).read_text())
    for descriptor in numerical['independent_saved_array_reviews'].values():
        name = descriptor['path']
        if Path(name).name != name:
            raise ValueError('unsafe sibling review path')
        path = Path(audit_path) if name == 'INDEPENDENT-RESULT.json' else Path(numerical_path).with_name(name)
        if sha(path) != descriptor['sha256']:
            raise ValueError('sibling independent review differs')
        shutil.copyfile(path, destination / name)
    record = {
        'schema': 'ns-obstacle-fine-primary-composition-v1', 'candidate_only': True,
        'original_root': str(source), 'view_root': str(destination),
        'original_files': {**retained, **omitted}, 'original_symlinks': links,
        'retained_original_files': retained, 'derived_volume_files': omitted,
        'compact_scientific_files': actual_science,
        'compact_scientific_bytes': sum(r['bytes'] for r in actual_science.values()),
        'independent_primary_review_sha256': AUDIT_SHA,
        'independent_numerical_review_sha256': NUMERICAL_SHA,
        'primary_manifest_sha256': audit['primary_manifest_sha256'],
        'primary_source_sha256': audit['primary_source_sha256'],
        'original_volume_manifest_sha256': audit['volume_manifest_sha256'],
        'result_sha256': audit['result_sha256'], 'replay_sha256': audit['replay_sha256'],
        'recovery': {
            'primary_directory': 'results/preparation/primary_volume',
            'training_manifest': 'datasets/fd1-training01-handoff/training.json',
            'preparation_release': 'PREPARATION-RELEASE.json',
            'required_count_context': 'prepare2d.verify_release then capacity_context2d.approved_capacity using exact staged release/capacity/source/FD1 inputs',
            'volume_operation': 'primary_volume.PrimaryVolumeStore then primary_volume.materialize into fresh prepared/volume; byte-identical copy to restored_volume',
            'verification': 'Every regenerated archive and semantic field hash plus complete original manifest; fail closed and retain discrepancies',
            'cost': 'Record regeneration, verification, copying and dependency setup as offline costs; no online timing claim',
            'runtime': 'Exact primary manifest runtime/source identity and retained dependency wheels/prefix; no native recovery executed here',
        },
        'reuse_index_sha256': sha(index),
        'composition_source_sha256': sha(__file__),
        'delta_helper_sha256': sha(Path(__file__).with_name('delta_retention.py')),
        'originals_preserved': True, 'omission_or_deletion_authorized': False,
    }
    dump(destination / 'COMPOSITION.json', record)
    receipt = write_delta(destination, index, archive)
    receipt.update(composition_sha256=sha(destination / 'COMPOSITION.json'), originals_preserved=True,
                   omission_or_deletion_authorized=False)
    dump(Path(archive).with_suffix('.measurement.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'destination', 'audit', 'numerical', 'index', 'archive'):
        parser.add_argument('--' + name, required=True, type=Path)
    a = parser.parse_args()
    print(json.dumps(compose(a.source, a.destination, a.audit, a.numerical, a.index, a.archive), indent=2))
