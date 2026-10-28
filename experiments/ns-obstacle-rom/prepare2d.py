"""Source-bound production preparation; never creates reference admission.

Use only after independent G2 training/common-lift admission and an exact-source
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


def preparation_digest():
    return hashlib.sha256(json.dumps({
        'entry_sha256': sha_file(Path(__file__)),
        'adapter_sha256': adapter_digest(),
        'pilot_sha256': source_digest()}, sort_keys=True).encode()).hexdigest()


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
    for key in ('reference_admission_sha256', 'adapter_review_sha256'):
        if admission[key] != release[key]:
            raise ValueError('preparation release differs from '+key)
    return release, admission


def prepare(manifest_path, release_path, output, point_block_volume=False):
    started = time.perf_counter()
    release, admission = verify_release(manifest_path, release_path)
    if release.get('point_block_volume', False) is not point_block_volume:
        raise ValueError('volume layout differs from exact preparation release')
    output = Path(output)
    if output.exists(): raise FileExistsError('fresh preparation directory required')
    output.mkdir(parents=True, exist_ok=False)
    # Preserve a failed partial run with its stage, never reuse it as a successful export.
    stage = 'adapter_open'
    try:
        adapter = TrainingFEAdapter.open(manifest_path, output/'native_evidence')
        stage = 'basis_build'; adapter.build()
        stage = 'array_export'; metadata = adapter.export(output/'prepared', point_block_volume=point_block_volume)
        stage = 'exact_pressure_probes'; probes = prepare_probe_arrays(adapter)
        for name, array in probes.items():
            if array.dtype != np.dtype('float64') or not np.isfinite(array).all():
                raise TypeError('finite original-f64 probe array required: '+name)
            if array.nbytes > CAPS['single_array_bytes']: raise MemoryError('probe array cap')
        probe_file = output/'probes.npz'; np.savez_compressed(probe_file, **probes)
        if probe_file.stat().st_size > CAPS['archive_bytes']: raise MemoryError('probe archive cap')
        record = {'schema': 'ns-obstacle-native-preparation-result-v1',
            'source_sha256': preparation_digest(), 'release_sha256': sha_file(release_path),
            'training_manifest_sha256': sha_file(manifest_path),
            'reference_admission_sha256': admission['reference_admission_sha256'],
            'adapter_manifest_sha256': sha_file(output/'prepared/manifest.json'),
            'probes_sha256': sha_file(probe_file), 'probe_points_physical': probes['points_physical'].tolist(),
            'shared_stokes_lift': metadata['shared_stokes_lift'],
            'elapsed_seconds': time.perf_counter()-started,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'status': 'execution_completed_pending_independent_review',
            'physical_admission': False, 'training_executed': False}
        (output/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        return record
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'stage': stage, 'error': repr(exc),
            'source_sha256': preparation_digest(), 'release_sha256': sha_file(release_path),
            'elapsed_seconds': time.perf_counter()-started, 'physical_admission': False}, indent=2)+'\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-manifest', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--point-block-volume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.training_manifest, args.release, args.out, args.point_block_volume), indent=2))


if __name__ == '__main__': main()
