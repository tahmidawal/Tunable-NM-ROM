"""Prospective A6 coefficient-mass QR/SVD. Pure NumPy/sparse-matvec algebra only.

No snapshot combination is divided by a POD singular value. The old quotient path is
preserved under legacy/97847a479 and is not imported by this module.
"""
from __future__ import annotations
import json
import numpy as np
from spaces import positive_weights, weighted_gram

TOL = {'backward': 1e-10, 'gram': 1e-10, 'singular': 1e-10, 'prefix_floor': 1e-10,
       'rank_relative': 1e-12, 'cluster_relative': 1e-10,
       'tie_roundoff_units': 64., 'component_matrix': 1e-12}
EPS = np.finfo(np.float64).eps


def weighted_matrix(snapshots, weights):
    u = np.asarray(snapshots, dtype=np.float64)
    w = positive_weights(weights, u.shape[1])
    if not np.isfinite(u).all(): raise ValueError('nonfinite training snapshots')
    sw = np.repeat(np.sqrt(w), int(np.prod(u.shape[2:])))
    return u.reshape(len(u), -1).T*sw[:, None]


def first_numerical_max(scores, indices=None):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.min(scores) < 0:
        raise ValueError('finite nonnegative pivot scores required')
    maximum = float(np.max(scores))
    if maximum == 0:
        ids = np.arange(len(scores)) if indices is None else np.asarray(indices)
        return int(np.argmin(ids))  # Callers explicitly reject an unusable zero pivot.
    tied = np.flatnonzero(scores >= maximum-TOL['tie_roundoff_units']*EPS*maximum)
    ids = np.arange(len(scores)) if indices is None else np.asarray(indices)
    return int(tied[np.argmin(ids[tied])])


def spectrum_clusters(singular_values):
    sv = np.asarray(singular_values, dtype=np.float64)
    if sv.ndim != 1 or np.any(sv < 0) or np.any(np.diff(sv) > 0):
        raise ValueError('descending nonnegative full singular spectrum required')
    normalized = sv/sv[0] if len(sv) and sv[0] > 0 else np.zeros_like(sv)
    groups = []; start = 0
    while start < len(sv):
        end = start+1
        while end < len(sv):
            tolerance = max(TOL['tie_roundoff_units']*EPS,
                            TOL['cluster_relative']*max(normalized[end-1], normalized[end]))
            if normalized[end-1]-normalized[end] > tolerance: break
            end += 1
        groups.append((start, end)); start = end
    return groups, normalized


def check_cluster_cuts(spectrum, cuts):
    groups, normalized = spectrum_clusters(spectrum)
    for cut in sorted(set(int(c) for c in cuts)):
        if any(a < cut < b for a, b in groups):
            raise ArithmeticError(f'POD cut {cut} splits normalized singular cluster {groups}')
    return {'full_singular_values': np.asarray(spectrum).tolist(),
            'normalized_singular_values': normalized.tolist(),
            'groups': [list(g) for g in groups], 'checked_cuts': sorted(set(cuts)),
            'relative_tolerance': TOL['cluster_relative'],
            'absolute_normalized_roundoff_floor': TOL['tie_roundoff_units']*EPS}


def orient_clusters(coefficients, groups):
    """Rotate only complete clusters; numerical FE-row ties choose the smallest index."""
    c = np.asarray(coefficients, dtype=np.float64)
    out = c.copy(); metadata = []
    for start, end in groups:
        if start >= c.shape[1]: break
        if end > c.shape[1]: raise ValueError('orientation would truncate a cluster')
        block = c[:, start:end]; directions = []; pivots = []
        for _ in range(end-start):
            residual = block.copy()
            if directions:
                q = np.column_stack(directions)
                for _ in range(2): residual -= (residual@q)@q.T
            scores = np.einsum('ij,ij->i', residual, residual)
            pivot = first_numerical_max(scores)
            direction = residual[pivot].copy()
            norm = np.linalg.norm(direction)
            if norm <= 0 or not np.isfinite(norm): raise ArithmeticError('cluster orientation lost rank')
            directions.append(direction/norm); pivots.append(pivot)
        rotation = np.column_stack(directions)
        if np.linalg.norm(rotation.T@rotation-np.eye(end-start), 2) > TOL['gram']:
            raise ArithmeticError('cluster rotation lost orthogonality')
        out[:, start:end] = block@rotation
        metadata.append({'start': start, 'end': end, 'FE_DOF_pivots': pivots,
                         'rotation': rotation.tolist()})
    return out, {'groups': metadata, 'numerical_tie_relative': TOL['tie_roundoff_units']*EPS,
                 'tie_break': 'smallest frozen FE DOF index'}


def normalized_error(numerator, scale, label):
    value = np.asarray(numerator)
    if scale == 0:
        if np.any(value != 0): raise ArithmeticError(label+': nonzero error with zero reference')
        return np.zeros_like(value)
    if scale < 0 or not np.isfinite(scale): raise ValueError('invalid normalization scale')
    return value/scale


def metric_qr(coefficients, mass, rank, reference_scale, protected=None):
    """Pivoted CGS2 with fixed externally determined rank, followed by a backward check."""
    c = np.asarray(coefficients, dtype=np.float64)
    if (c.ndim != 2 or mass.shape != (len(c), len(c)) or not np.isfinite(c).all() or
            not 0 <= rank <= c.shape[1] or reference_scale < 0 or (reference_scale == 0 and rank != 0)):
        raise ValueError('invalid mass-QR inputs')
    protected = np.empty((len(c), 0)) if protected is None else np.asarray(protected)
    if protected.shape[0] != len(c) or (protected.shape[1] and
        np.linalg.norm(protected.T@(mass@protected)-np.eye(protected.shape[1]), 2) > TOL['gram']):
        raise ValueError('protected pressure constant is not metric normalized')
    q = np.empty((len(c), 0)); remaining = list(range(c.shape[1])); pivots = []
    for _ in range(rank):
        residuals = []; scores = []
        for j in remaining:
            v = c[:, j].copy()
            orthogonal_to = np.c_[protected, q]
            for _ in range(2): v -= orthogonal_to@(orthogonal_to.T@(mass@v))
            squared = float(v@(mass@v))
            if squared < 0 or not np.isfinite(squared): raise ArithmeticError('invalid mass norm')
            residuals.append(v); scores.append(squared)
        k = first_numerical_max(scores, remaining)
        if scores[k] <= 0: raise ArithmeticError('fixed sampled rank cannot be realized by mass QR')
        q = np.c_[q, residuals[k]/np.sqrt(scores[k])]
        pivots.append(remaining.pop(k))
    permutation = pivots+remaining
    permuted = c[:, permutation]
    r = q.T@(mass@permuted)
    defect = permuted-q@r
    squared_defect = float(np.sum(defect*(mass@defect)))
    if squared_defect < 0: raise ArithmeticError('invalid QR backward mass norm')
    backward = float(normalized_error(np.sqrt(squared_defect), reference_scale, 'QR backward'))
    gram = q.T@(mass@q)
    orthogonality = float(np.linalg.norm(gram-np.eye(rank), 2)) if rank else 0.
    report = {'rank_from_original_sample_spectrum': rank, 'snapshot_pivots': pivots,
              'permutation': permutation, 'factorization': 'C[:,permutation] = Q R',
              'pivoting_is_rank_cutoff': False, 'protected_constant_count': protected.shape[1], 'passes': 2, 'normalized_backward_error': float(backward),
              'backward_normalization': 'original_rank_reference_scale',
              'gram_spectral_error': orthogonality, 'gram': gram.tolist(),
              'reference_scale': float(reference_scale), 'tolerances': dict(TOL)}
    if backward > TOL['backward'] or orthogonality > TOL['gram']:
        raise ArithmeticError('mass QR verification failed: '+json.dumps(report, allow_nan=False))
    return q, r, report


def mass_pod(coefficients, snapshots, weights, mass, max_rank, registered_cuts,
             rank_reference=None, protected=None):
    sampled = weighted_matrix(snapshots, weights)
    sample_singular = np.linalg.svd(sampled, compute_uv=False)
    reference = float(sample_singular[0]) if rank_reference is None else float(rank_reference)
    if reference < 0: raise ValueError('nonnegative original rank reference required')
    threshold = TOL['rank_relative']*reference
    rank = int(np.sum(sample_singular > threshold)); retained = min(rank, max_rank)
    cuts = [rank, retained]+[c for c in registered_cuts if c <= retained]
    sample_clusters = check_cluster_cuts(sample_singular, cuts)
    q, r, qr = metric_qr(coefficients, mass, rank, reference, protected)
    left, singular, _ = np.linalg.svd(r, full_matrices=False)
    padded = np.zeros_like(sample_singular); padded[:len(singular)] = singular
    singular_gap = float(normalized_error(np.max(np.abs(padded-sample_singular), initial=0), reference, 'singular gap'))
    if singular_gap > TOL['singular'] or np.sum(padded > threshold) != rank:
        raise ArithmeticError('mass-R and sampled singular spectrum/rank disagree')
    qr_clusters = check_cluster_cuts(padded, cuts)
    modes = q@left[:, :retained]
    groups, _ = spectrum_clusters(padded)
    modes, orientation = orient_clusters(modes, groups)
    return modes, {'numerical_rank': rank, 'retained_rank': retained,
                   'rank_reference': reference, 'rank_threshold': threshold, 'zero_input': reference == 0,
                   'zero_input_policy': 'zero error only; no eligible velocity prefix for zero rank',
                   'sample_singular_values': sample_singular.tolist(),
                   'small_R_singular_values_padded': padded.tolist(),
                   'normalized_singular_gap': singular_gap, 'QR': qr,
                   'sample_clusters': sample_clusters, 'QR_clusters': qr_clusters,
                   'orientation': orientation, 'tolerances': dict(TOL)}


def coefficient_pod_a6(velocity_coefficients, velocity_samples, pressure_coefficients,
                       pressure_samples, constant_coefficients, weights, velocity_mass, pressure_mass):
    vc = np.asarray(velocity_coefficients); pc = np.asarray(pressure_coefficients)
    p = np.asarray(pressure_samples); one = np.asarray(constant_coefficients)
    cu, um = mass_pod(vc, velocity_samples, weights, velocity_mass, 8, (2, 4, 8))
    eligible = [r for r in (2, 4, 8) if r <= cu.shape[1]]
    if not eligible: raise ValueError('no registered velocity prefix passes unchanged sampled rank')
    w = positive_weights(weights, p.shape[1]); means = p@w/w.sum()
    centered = p-means[:, None]
    center_coeff = pc-one[:, None]*means[None, :]
    reference = max(1., float(np.linalg.svd(weighted_matrix(p, w), compute_uv=False)[0]))
    cp, pm = mass_pod(center_coeff, centered, w, pressure_mass, 7, (1, 3, 7), reference,
                      protected=one[:, None]/np.sqrt(w.sum()))
    cp = np.c_[one/np.sqrt(w.sum()), cp]
    um['eligible_velocity_prefixes'] = eligible
    pm.update(pressure_rank=cp.shape[1], constant_first=True, physical_gauge_shift=False,
              snapshot_means=means.tolist(), rank_reference_definition='max(1,original weighted pressure sigma1)')
    return {'velocity_coefficients': cu, 'pressure_coefficients': cp,
            'metadata': {'construction': 'A6 coefficient mass CGS2 and small-R SVD',
                         'velocity': um, 'pressure': pm}}


def validate_actual_pod(actual_modes, original_snapshots, weights, metadata, pressure=False):
    """Actual field Gram and every registered prefix versus independent sampled-SVD prefix residuals."""
    a = np.asarray(actual_modes); y = weighted_matrix(original_snapshots, weights)
    sw = np.repeat(np.sqrt(weights), int(np.prod(a.shape[1:-1])))
    matrix = a.reshape(-1, a.shape[-1])*sw[:, None]
    gram = matrix.T@matrix
    gram_error = float(np.linalg.norm(gram-np.eye(a.shape[-1]), 2))
    reference = metadata['rank_reference']
    if pressure:
        target_center = np.asarray(original_snapshots)-np.asarray(metadata['snapshot_means'])[:, None]
        left, sv, _ = np.linalg.svd(weighted_matrix(target_center, weights), full_matrices=False)
        constant = np.sqrt(weights)/np.sqrt(np.sum(weights))
        prefixes = sorted({min(r, a.shape[-1]) for r in (2, 4, 8)})
    else:
        left, sv, _ = np.linalg.svd(y, full_matrices=False)
        prefixes = [r for r in (2, 4, 8) if r <= a.shape[-1]]
    padded = np.asarray(metadata['small_R_singular_values_padded'])
    singular_gap = float(normalized_error(np.max(np.abs(sv-padded), initial=0), reference, 'actual singular gap'))
    prefix_reports = []
    for prefix in prefixes:
        basis = matrix[:, :prefix]
        coef, _, rank, _ = np.linalg.lstsq(basis, y, rcond=TOL['rank_relative'])
        if rank != prefix: raise ArithmeticError('actual evaluated registered prefix lost rank')
        residual = y-basis@coef
        ideal = np.c_[constant, left[:, :prefix-1]] if pressure else left[:, :prefix]
        ideal_coef, _, _, _ = np.linalg.lstsq(ideal, y, rcond=TOL['rank_relative'])
        ideal_residual = y-ideal@ideal_coef
        norms = np.linalg.norm(residual, axis=0); optimum = np.linalg.norm(ideal_residual, axis=0)
        differences = normalized_error(np.abs(norms-optimum), reference, 'per-case prefix residual difference')
        prefix_reports.append({'prefix': prefix, 'actual_residual_norms': norms.tolist(),
                               'sampled_SVD_prefix_residual_norms': optimum.tolist(),
                               'aggregate_actual_Frobenius': float(np.linalg.norm(residual)),
                               'aggregate_sampled_POD_optimal_Frobenius': float(np.linalg.norm(ideal_residual)),
                               'normalized_floor_differences': differences.tolist()})
    report = {'actual_gram': gram.tolist(), 'actual_gram_spectral_error': gram_error,
              'normalized_singular_gap': singular_gap, 'prefixes': prefix_reports,
              'reference_scale': reference, 'floor_normalization': 'original rank-reference scale',
              'tolerances': dict(TOL), 'rank_or_cutoff_changed': False,
              'optimality_scope': 'aggregate weighted Frobenius only; per-case values are SVD-prefix comparisons'}
    passed = (gram_error <= TOL['gram'] and singular_gap <= TOL['singular'] and
              all(max(r['normalized_floor_differences']) <= TOL['prefix_floor'] for r in prefix_reports))
    report['passed'] = bool(passed)
    if not passed: raise ArithmeticError('A6 actual POD verification failed: '+json.dumps(report, allow_nan=False))
    return gram, report


def expand_components(scalar_modes, component_maps, vector_ndof):
    scalar = np.asarray(scalar_modes); maps = np.asarray(component_maps)
    if (maps.shape != (2, len(scalar)) or not np.issubdtype(maps.dtype, np.integer) or
            sorted(maps.ravel().tolist()) != list(range(vector_ndof))):
        raise ValueError('component maps must be a bijection of all vector DOFs')
    result = np.zeros((vector_ndof, 2*scalar.shape[1]))
    for j in range(scalar.shape[1]):
        for component in range(2): result[maps[component], 2*j+component] = scalar[:, j]
    return result


def check_component_blocks(vector_h1, vector_mass, scalar_h1, scalar_mass, component_maps):
    from scipy.sparse.linalg import norm as sparse_norm
    maps = np.asarray(component_maps); reports = {}
    for label, vector, scalar in [('H1', vector_h1, scalar_h1), ('L2', vector_mass, scalar_mass)]:
        for i in range(2):
            diagonal = vector[maps[i]][:, maps[i]]
            denominator = float(sparse_norm(diagonal))
            if denominator <= 0: raise ValueError('nonzero matching diagonal block norm required')
            reports[f'{label}_{i}_diagonal_norm'] = denominator
            for j in range(2):
                block = vector[maps[i]][:, maps[j]]
                difference = block-scalar if i == j else block
                gap = float(sparse_norm(difference)/denominator)
                reports[f'{label}_{i}_{j}'] = gap
                if gap > TOL['component_matrix']:
                    raise ArithmeticError('vector/scalar component block mismatch: '+str(reports))
    return {'block_checks': reports, 'normalization': 'nonzero actual matching row-component diagonal-block Frobenius norm',
            'tolerance': TOL['component_matrix']}
