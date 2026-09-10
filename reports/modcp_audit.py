"""Independent NumPy checks for the modified-CP architecture pilot.

This module imports neither experiment implementations nor JAX. Wave stiffness
norms are assembled from edge differences, independently of the solver stencil.
"""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import pickle
from statistics import median

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def audit_checkpoint(path):
    """Check a trusted, locally generated pilot checkpoint without importing JAX."""
    path = Path(path)
    with path.open('rb') as handle:
        data = pickle.load(handle)
    cfg, codes, extra = data['config'], data['Z'], data['extra']
    def leaves(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from leaves(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from leaves(child)
        else:
            yield np.asarray(value)
    arrays = list(leaves(data['params']))
    if codes.ndim != 2 or codes.shape[1] != cfg['k']:
        raise ValueError('Checkpoint code shape disagrees with architecture')
    if any(a.dtype != np.float64 or not np.isfinite(a).all() for a in arrays+[codes]):
        raise ValueError('Checkpoint contains nonfinite or non-f64 model arrays')
    updates = extra.get('total_training_steps', extra.get('updates'))
    if not updates:
        raise ValueError('Checkpoint is missing completed update metadata')
    return {'path': str(path), 'sha256': digest(path), 'configuration': cfg,
            'code_shape': list(codes.shape), 'parameter_count': sum(a.size for a in arrays),
            'completed_updates': updates, 'metadata': {k: v for k, v in extra.items() if k != 'history'},
            'finite_f64': True}


def wave_weights(n, boundary):
    if boundary not in ('dirichlet', 'absorbing'):
        raise ValueError(f'Unsupported pilot boundary: {boundary}')
    axis = np.ones(n - 1 if boundary == 'dirichlet' else n + 1) / n
    if boundary == 'absorbing':
        axis[[0, -1]] *= .5
    return np.outer(axis, axis), axis


def wave_energy_squared(u, v, n, boundary, speed):
    """Twice the discrete physical energy of the supplied (possibly error) state."""
    mass, axis = wave_weights(n, boundary)
    if u.shape != v.shape or u.shape[-2:] != mass.shape:
        raise ValueError('Wave field shape does not match mesh and boundary')
    if boundary == 'dirichlet':
        pad = [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)]
        z = np.pad(u, pad)
        potential = sum(np.sum(np.diff(z, axis=d)**2, axis=(-2, -1)) for d in (-2, -1))
    else:
        potential = n * np.sum(np.diff(u, axis=-2)**2 * axis, axis=(-2, -1))
        potential += n * np.sum(np.diff(u, axis=-1)**2 * axis[:, None], axis=(-2, -1))
    return np.sum(mass * v * v, axis=(-2, -1)) + speed**2 * potential


def wave_metrics(u, v, truth_u, truth_v, n, boundary, speed):
    arrays = tuple(np.asarray(x, dtype=np.float64) for x in (u, v, truth_u, truth_v))
    u, v, truth_u, truth_v = arrays
    if len({a.shape for a in arrays}) != 1 or u.ndim != 3:
        raise ValueError('Expected matching time,x,y wave trajectories')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite trajectory')
    mass, _ = wave_weights(n, boundary)
    norm = lambda a: np.sqrt(np.sum(mass * a * a, axis=(-2, -1)))
    scale_u = float(norm(truth_u[0]))
    scale_energy = float(np.sqrt(wave_energy_squared(truth_u[0], truth_v[0], n, boundary, speed)))
    if scale_u <= 0 or scale_energy <= 0:
        raise ValueError('Degenerate initial physical normalization')
    error_u = norm(u - truth_u) / scale_u
    error_v = norm(v - truth_v) / scale_energy
    error_e = np.sqrt(wave_energy_squared(u-truth_u, v-truth_v, n, boundary, speed)) / scale_energy
    return {'displacement': float(error_u.max()), 'velocity': float(error_v.max()),
            'energy_state': float(error_e.max())}


def burgers_metrics(u, truth):
    u, truth = np.asarray(u), np.asarray(truth)
    if u.shape != truth.shape or u.ndim != 3 or not np.isfinite(u).all() or not np.isfinite(truth).all():
        raise ValueError('Expected finite matching time,x,y Burgers trajectories')
    scale = np.linalg.norm(truth[0])
    if scale <= 0:
        raise ValueError('Degenerate initial Burgers field')
    return {'displacement': float(np.max(np.linalg.norm((u-truth).reshape(len(u), -1), axis=1)) / scale)}


def row_error(row):
    errors = row.get('errors') or {}
    values = [errors[key] for key in ('displacement', 'velocity', 'energy_state') if key in errors]
    if not values or any(x is None or not math.isfinite(float(x)) or x < 0 for x in values):
        return math.inf
    return max(values)


def summarize_rows(rows, expected_cases, expected_repetitions, target):
    """All-case/all-repetition qualification; failures never disappear from means."""
    if not expected_cases or expected_repetitions <= 0:
        raise ValueError('Explicit case membership and repetition count are required')
    by_case = defaultdict(list)
    identifiers = set()
    for row in rows:
        identity = (row['case'], row['rep'])
        if identity in identifiers:
            raise ValueError(f'Duplicate timed invocation {identity}')
        identifiers.add(identity)
        elapsed = row['seconds']
        if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Nonpositive or nonfinite measured duration')
        by_case[row['case']].append(row)
    expected = {(case, rep) for case in expected_cases for rep in range(expected_repetitions)}
    coverage = identifiers == expected
    failed = [case for case in expected_cases if case not in by_case or any(
        not r.get('finite', False) or not r.get('completed', False) for r in by_case[case])]
    outliers = [case for case in expected_cases if case not in by_case or any(row_error(r) > target for r in by_case[case])]
    def stationary(row):
        if row.get('method') in ('cp', 'modcp', 'film'):
            return row.get('stationary', row.get('solver', {}).get('stationary', False)) is True
        return row.get('stationary', row.get('completed', False)) is True
    nonstationary = [case for case in expected_cases if case not in by_case or any(not stationary(r) for r in by_case[case])]
    per_case_seconds = [median(r['seconds'] for r in by_case[c]) for c in expected_cases if c in by_case]
    worst = max((row_error(r) for r in rows), default=math.inf)
    return {'complete_coverage': coverage, 'cases': len(expected_cases),
            'failed_cases': len(failed), 'outlier_cases': len(outliers),
            'nonstationary_cases': len(nonstationary),
            'median_seconds': median(per_case_seconds) if per_case_seconds else None,
            'worst_error': worst if math.isfinite(worst) else None,
            'qualified': coverage and not failed and not outliers,
            'qualified_and_converged': coverage and not failed and not outliers and not nonstationary,
            'observed_invocations': len(rows), 'expected_invocations': len(expected)}


def audit_field_archive(path, case_name, expected_errors, rtol=1e-8, atol=1e-10):
    """Audit self-contained full-grid NPZ; metadata is scalar, fields include t=0."""
    with np.load(path, allow_pickle=False) as data:
        if case_name == 'burgers2d':
            actual = burgers_metrics(data['u'], data['truth_u'])
        else:
            actual = wave_metrics(data['u'], data['v'], data['truth_u'], data['truth_v'],
                                  int(data['intervals']), str(data['boundary']), float(data['speed']))
    for name, value in actual.items():
        if name not in expected_errors or not np.isclose(value, expected_errors[name], rtol=rtol, atol=atol):
            raise ValueError(f'Independent field error mismatch: {path}: {name}: {value} vs {expected_errors.get(name)}')
    return {'path': str(path), 'sha256': digest(path), 'errors': actual}
