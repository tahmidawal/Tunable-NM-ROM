"""Pure-array adapter checks. No NGSolve import, mesh creation or FE solve."""
from pathlib import Path
import json
import sys
import tempfile
import time
import numpy as np
from scipy.sparse import diags
from fe_algebra import coefficient_pod, reference_inf_sup, verify_reconstructed_modes
from fe_adapter import (TrainingFEAdapter, bounded_eigenmodes, bounded_factor,
                        read_training_manifest, adapter_digest, load_npz_bounded, strict_scalar)
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
    actual_u_gram, uc = verify_reconstructed_modes(ur, pod['velocity_values'], w, 'velocity')
    actual_p_gram, pcheck = verify_reconstructed_modes(pr, pod['pressure_values'], w, 'pressure')
    assert np.array_equal(actual_u_gram, weighted_gram(ur, w))
    assert np.array_equal(actual_p_gram, weighted_gram(pr, w))
    # A small allowed discrepancy still returns the actual Gram, never ideal identity.
    perturbed = pr*(1.+1e-12)
    perturbed_gram, _ = verify_reconstructed_modes(perturbed, pod['pressure_values'], w, 'pressure')
    assert np.array_equal(perturbed_gram, weighted_gram(perturbed, w))
    assert np.max(np.abs(perturbed_gram-weighted_gram(pod['pressure_values'], w))) > 1e-12
    # A per-mode discrepancy below its tolerance can still fail the distinct Gram gate.
    gram_rejected = False
    try: verify_reconstructed_modes(pr*(1.+.75e-10), pr, w, 'pressure')
    except ArithmeticError as exc:
        report = json.loads(str(exc).split(': ', 1)[1])
        gram_rejected = (report['weighted_relative_discrepancy_max'] <= report['weighted_relative_tolerance']
                         and report['actual_gram_spectral_error'] > report['gram_spectral_tolerance'])
    assert gram_rejected
    # Actual near-cutoff back-transforms: retain the original 1e-12 rank test, but
    # reject cancellation after C @ V / sigma. No fabricated perturbation is added.
    left, _ = np.linalg.qr(rng.normal(size=(18, 2)))
    right, _ = np.linalg.qr(rng.normal(size=(9, 2)))
    tiny_vc = (left*np.array([1., 4e-12]))@right.T
    orth_map, _ = np.linalg.qr(rng.normal(size=(2*n, 18)))
    orth_map = (orth_map/np.repeat(np.sqrt(w), 2)[:, None]).reshape(n, 2, 18)
    tiny_u = np.einsum('ndk,ks->snd', orth_map, tiny_vc)
    tiny_pod = coefficient_pod(tiny_vc, tiny_u, pc, p, one, w)
    tiny_ur = np.einsum('ndk,kr->ndr', orth_map, tiny_pod['velocity_coefficients'])
    conditioned = {}
    try: verify_reconstructed_modes(tiny_ur, tiny_pod['velocity_values'], w, 'velocity')
    except ArithmeticError as exc:
        conditioned['velocity'] = json.loads(str(exc).split(': ', 1)[1])
    assert 'velocity' in conditioned
    assert tiny_pod['metadata']['velocity']['numerical_rank'] == 2
    pressure_map = rng.normal(size=(n, 2))
    pressure_map -= (w@pressure_map/w.sum())[None, :]
    pressure_map, _ = np.linalg.qr(np.sqrt(w[:, None])*pressure_map)
    pressure_map = np.c_[np.ones(n), pressure_map/np.sqrt(w[:, None])]
    tiny_pc = np.r_[np.zeros((1, 9)), np.diag([1., 4e-12])@right.T]
    tiny_pressure = (pressure_map@tiny_pc).T
    one_p = np.array([1., 0., 0.])
    tiny_ppod = coefficient_pod(vc, u, tiny_pc, tiny_pressure, one_p, w)
    tiny_pr = pressure_map@tiny_ppod['pressure_coefficients']
    try: verify_reconstructed_modes(tiny_pr, tiny_ppod['pressure_values'], w, 'pressure')
    except ArithmeticError as exc:
        conditioned['pressure'] = json.loads(str(exc).split(': ', 1)[1])
    assert 'pressure' in conditioned
    assert tiny_ppod['metadata']['pressure']['centered_numerical_rank'] == 2
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
        # Inspect original archive dtypes before np.load can convert/allocate payloads.
        archive = Path(d)/'typed.npz'
        dtype_rejections = []
        for dtype in (np.float32, np.float16, np.int64, np.complex128):
            np.savez_compressed(archive, values=np.array([1., 2.], dtype=dtype))
            original_load = np.load
            def forbidden_payload_load(*args, **kwargs):
                raise AssertionError('non-f64 payload must be rejected from its header')
            np.load = forbidden_payload_load
            try:
                try: load_npz_bounded(archive)
                except TypeError: dtype_rejections.append(np.dtype(dtype).name)
            finally:
                np.load = original_load
        assert len(dtype_rejections) == 4
        for dtype in (np.dtype('<f8'), np.dtype('>f8')):
            expected = np.array([1., 2.], dtype=dtype)
            np.savez_compressed(archive, values=expected)
            loaded = load_npz_bounded(archive)['values']
            assert loaded.dtype == np.float64 and np.array_equal(loaded, expected)
        scalar_accepted = [1.25, (1.25,), np.array([1.25]), np.array([[1.25]])]
        assert all(strict_scalar(value) == 1.25 for value in scalar_accepted)
        scalar_rejected = 0
        for value in ((1., 2.), np.array([1., 2.]), (), np.nan, (np.inf,), 1.+2.j, '1.25'):
            try: strict_scalar(value)
            except ValueError: scalar_rejected += 1
        assert scalar_rejected == 7
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
            'verified_mode_reconstruction': {'velocity': uc, 'pressure': pcheck},
            'actual_pressure_gram_returned': True,
            'orthonormality_gate_rejection_below_discrepancy_tolerance': gram_rejected,
            'near_rank_cutoff_cancellation_rejections': conditioned,
            'stored_dtype_rejected_before_payload_load': dtype_rejections,
            'native_and_opposite_endian_float64_preserved': True,
            'scalar_or_singleton_accepted': len(scalar_accepted),
            'nonscalar_or_nonfinite_rejected': scalar_rejected,
            'full_velocity_reference_beta_gap': abs(beta-direct),
            'generalized_eigen_residuals': em['residuals'],
            'eigenblock_cutoff_rejected': split_block_rejected,
            'G1_manifest_rejected_before_FE_import': closed_rejected,
            'validation_manifest_rejected': validation_rejected,
            'ngsolve_imported': False, 'adapter_source_sha256': adapter_digest(),
            'seconds': time.perf_counter()-start}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
