"""Source-bound production preparation; never creates reference admission.

Use only after independent FD1-scoped training/common-lift admission and an exact-source
preparation release. This program has no training or validation-data path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from fe_adapter import TrainingFEAdapter, adapter_digest, read_training_manifest, CAPS, sha_file
from pilot2d import prepare_probe_arrays, source_digest
from qualification import require_qualification


def preparation_digest():
    from primary_volume import source_digest as primary_source_digest
    return hashlib.sha256(json.dumps({
        'entry_sha256': sha_file(Path(__file__)),
        'capacity_context_sha256': sha_file(Path(__file__).with_name('capacity_context2d.py')),
        'adapter_sha256': adapter_digest(),
        'pilot_sha256': source_digest(),
        'primary_volume_sha256': primary_source_digest()}, sort_keys=True).encode()).hexdigest()


def verify_release(manifest_path, release_path):
    """Verify before native imports and before creating any output directory."""
    release = json.loads(Path(release_path).read_text())
    if (release.get('schema') != 'ns-obstacle-native-preparation-release-v1' or
            release.get('authorized_stage') != 'G3-2d-steady-preparation' or
            release.get('source_sha256') != preparation_digest()):
        raise RuntimeError('reviewed exact-source native preparation release required')
    if sha_file(manifest_path) != release['training_manifest_sha256']:
        raise ValueError('training manifest differs from exact release')
    admission = read_training_manifest(manifest_path)
    if require_qualification(release.get('reference_qualification')) != require_qualification(admission):
        raise ValueError('native preparation qualification scope differs')
    for key in ('reference_admission_sha256', 'adapter_review_sha256'):
        if admission[key] != release[key]:
            raise ValueError('preparation release differs from '+key)
    return release, admission


def prepare(manifest_path, release_path, output, point_block_volume=False, verify_primary_volume=False):
    release, admission = verify_release(manifest_path, release_path)
    from capacity_context2d import approved_capacity
    with approved_capacity(release_path, release, admission, preparation_digest(), CAPS) as capacity:
        return _prepare_released(release, admission, manifest_path, release_path, output, point_block_volume, verify_primary_volume, capacity)


def _prepare_released(release, admission, manifest_path, release_path, output, point_block_volume, verify_primary_volume, capacity):
    started = time.perf_counter()
    if release.get('point_block_volume', False) is not point_block_volume:
        raise ValueError('volume layout differs from exact preparation release')
    if release.get('verify_primary_volume', False) is not verify_primary_volume:
        raise ValueError('primary replay differs from exact preparation release')
    if verify_primary_volume and not point_block_volume:
        raise ValueError('primary replay requires complete point-block volume')
    output = Path(output)
    if output.exists(): raise FileExistsError('fresh preparation directory required')
    output.mkdir(parents=True, exist_ok=False)
    # Preserve a failed partial run with its stage, never reuse it as a successful export.
    stage = 'adapter_open'
    try:
        adapter = TrainingFEAdapter.open(manifest_path, output/'native_evidence')
        stage = 'basis_build'; adapter.build()
        stage = 'array_export'; metadata = adapter.export(output/'prepared', point_block_volume=point_block_volume)
        primary_replay = None
        if verify_primary_volume:
            from primary_volume import write_primary, PrimaryVolumeStore, materialize
            stage = 'primary_volume_capture'
            primary = write_primary(adapter, output/'primary_volume', adapter.cases[0]['folder'],
                                    output/'prepared/volume')
            stage = 'primary_volume_exact_replay'
            primary_sha = sha_file(output/'primary_volume/manifest.json')
            store = PrimaryVolumeStore(output/'primary_volume', primary_sha, adapter.ng)
            restored = materialize(store, output/'restored_volume')
            if restored['manifest_sha256'] != sha_file(output/'prepared/volume/manifest.json'):
                raise ValueError('complete fine preparation replay manifest differs')
            primary_replay = {'primary_manifest_sha256': primary_sha,
                'capture_seconds': primary['creation_seconds'], 'restoration': restored,
                'replay_accounting': store.replay_accounting,
                'original_fields_retained': True, 'restored_fields_retained': True,
                'omission_authorized': False, 'status': 'pending_independent_fine_replay_review'}
            (output/'primary_replay.json').write_text(json.dumps(primary_replay, indent=2, allow_nan=False)+'\n')
            del store
        stage = 'exact_pressure_probes'; probes = prepare_probe_arrays(adapter)
        for name, array in probes.items():
            if array.dtype != np.dtype('float64') or not np.isfinite(array).all():
                raise TypeError('finite original-f64 probe array required: '+name)
            if array.nbytes > CAPS['single_array_bytes']: raise MemoryError('probe array cap')
        probe_file = output/'probes.npz'; np.savez_compressed(probe_file, **probes)
        if probe_file.stat().st_size > CAPS['archive_bytes']: raise MemoryError('probe archive cap')
        record = {'schema': 'ns-obstacle-native-preparation-result-v1',
            'source_sha256': preparation_digest(), 'release_sha256': sha_file(release_path),
            'reference_qualification': require_qualification(admission),
            'training_manifest_sha256': sha_file(manifest_path),
            'reference_admission_sha256': admission['reference_admission_sha256'],
            'adapter_manifest_sha256': sha_file(output/'prepared/manifest.json'),
            'probes_sha256': sha_file(probe_file), 'probe_points_physical': probes['points_physical'].tolist(),
            'shared_stokes_lift': metadata['shared_stokes_lift'],
            'primary_volume_replay': primary_replay, 'capacity_context': capacity,
            'elapsed_seconds': time.perf_counter()-started,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'status': 'execution_completed_pending_independent_review',
            'physical_admission': False, 'training_executed': False}
        (output/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        return record
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'stage': stage, 'error': repr(exc),
            'reference_qualification': require_qualification(admission),
            'source_sha256': preparation_digest(), 'release_sha256': sha_file(release_path),
            'elapsed_seconds': time.perf_counter()-started, 'physical_admission': False}, indent=2)+'\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-manifest', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--point-block-volume', action='store_true')
    parser.add_argument('--verify-primary-volume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.training_manifest, args.release, args.out, args.point_block_volume, args.verify_primary_volume), indent=2))


if __name__ == '__main__': main()
