"""Pure-array adapter checks. No NGSolve import, mesh creation or FE solve."""
from pathlib import Path
import json
import sys
import tempfile
import time
import numpy as np
from scipy.sparse import diags
from fe_algebra import coefficient_pod, reference_inf_sup
from fe_adapter import (TrainingFEAdapter, bounded_eigenmodes, bounded_factor,
                        read_training_manifest, adapter_digest)
from spaces import weighted_gram


def run():
    start = time.perf_counter()
    rng = np.random.default_rng(27102027)
    n = 40; w = rng.uniform(.3, 1., n)
    vc = rng.normal(size=(18, 9)); vm = rng.normal(size=(n, 2, 18))
    pc = rng.normal(size=(12, 9)); pm = rng.normal(size=(n, 12)); pm[:, 0] = 1.
    one = np.zeros(12); one[0] = 1.
    u = np.einsum('ndk,ks->snd', vm, vc); p = (pm @ pc).T
    pod = coefficient_pod(vc, u, pc, p, one, w)
    ur = np.einsum('ndk,kr->ndr', vm, pod['velocity_coefficients'])
    pr = pm @ pod['pressure_coefficients']
    gaps = {'velocity_arbitrary_FE_map': float(np.max(np.abs(ur-pod['velocity_values']))),
            'pressure_arbitrary_FE_map': float(np.max(np.abs(pr-pod['pressure_values']))),
            'velocity_L2_orthonormal': float(np.max(np.abs(weighted_gram(ur, w)-np.eye(8)))),
            'pressure_L2_orthonormal': float(np.max(np.abs(weighted_gram(pr, w)-np.eye(8))))}
    assert max(gaps.values()) < 1e-12
    # Pressure constant stays solved even when training corrections are all constant.
    pc0 = one[:, None] @ np.arange(1., 10.)[None, :]
    cp0 = coefficient_pod(vc, u, pc0, (pm@pc0).T, one, w)
    assert cp0['pressure_coefficients'].shape == (12, 1)
    assert np.allclose(pm@cp0['pressure_coefficients'], 1/np.sqrt(w.sum()))
    # Schur-complement beta equals an independent full-velocity SVD.
    a = rng.normal(size=(15, 15)); a = a@a.T+np.eye(15)
    b = rng.normal(size=(15, 4)); pg = np.diag([1., 2., 3., 4.])
    s = np.linalg.solve(a, b)
    beta = reference_inf_sup(b, s, pg)['beta']
    direct = np.linalg.svd(np.linalg.solve(np.linalg.cholesky(pg), b.T) @
                          np.linalg.inv(np.linalg.cholesky(a).T), compute_uv=False)[-1]
    assert abs(beta-direct) < 1e-12
    # Generalized eigensolver: complete repeated blocks and an intentional cutoff failure.
    mass = diags(np.linspace(1., 2., 40), format='csc')
    spectrum = 1.+np.arange(40)//2
    h1 = diags(spectrum*mass.diagonal(), format='csc')
    fac, _ = bounded_factor(h1)
    eig, em = bounded_eigenmodes(h1, mass, fac.solve, 6)
    assert np.allclose(em['eigenvalues_H1'], [1, 1, 2, 2, 3, 3])
    assert np.allclose(eig.T@(mass@eig), np.eye(6))
    split_block_rejected = False
    try: bounded_eigenmodes(h1, mass, fac.solve, 5)
    except ValueError as exc: split_block_rejected = 'splits eigenvalue cluster' in str(exc)
    assert split_block_rejected
    # Closed G2 rejects before looking for nonexistent FE files or importing ngsolve.
    with tempfile.TemporaryDirectory(prefix='adapter-synthetic-', dir=Path(__file__).parent) as d:
        path = Path(d)/'manifest.json'
        path.write_text(json.dumps({'schema': 'ns-obstacle-fe-training-v1', 'gate': 'G1',
                                    'passed': True, 'split': 'training', 'cases': []}))
        closed_rejected = False
        try: TrainingFEAdapter.open(path)
        except RuntimeError as exc: closed_rejected = 'G2-steady' in str(exc)
        assert closed_rejected
        path.write_text(json.dumps({'schema': 'ns-obstacle-fe-training-v1', 'gate': 'G2-steady',
            'passed': True, 'split': 'validation', 'adapter_source_sha256': adapter_digest()}))
        validation_rejected = False
        try: read_training_manifest(path)
        except RuntimeError: validation_rejected = True
        assert validation_rejected
    assert 'ngsolve' not in sys.modules
    return {'scope': 'synthetic coefficient/matrix algebra only; no FE execution',
            'passed': True, 'physical_admission': False, 'pinned_NGSolve_integration': 'not_run',
            'coefficient_reconstruction_gaps': gaps, 'constant_pressure_count': 1,
            'full_velocity_reference_beta_gap': abs(beta-direct),
            'generalized_eigen_residuals': em['residuals'],
            'eigenblock_cutoff_rejected': split_block_rejected,
            'G1_manifest_rejected_before_FE_import': closed_rejected,
            'validation_manifest_rejected': validation_rejected,
            'ngsolve_imported': False, 'adapter_source_sha256': adapter_digest(),
            'seconds': time.perf_counter()-start}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
