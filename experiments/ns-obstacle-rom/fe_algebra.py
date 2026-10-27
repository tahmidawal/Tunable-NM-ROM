"""Small algebra for a training-only FE bridge; no NGSolve import or FE execution."""
from __future__ import annotations
import hashlib
import numpy as np
from spaces import (canonicalize_eigenmodes, checked_cholesky, pressure_basis,
                    weighted_gram, weighted_pod)


def array_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def canonical_pod(coefficients, values, singular_values, weights):
    """Orient POD blocks using frozen FE coefficient order, keeping sampled fields exact."""
    c = np.asarray(coefficients, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    # Negative singular values are ascending. Cluster/sign rule is the existing rule.
    oriented, meta = canonicalize_eigenmodes(-np.asarray(singular_values), c,
                                            weighted_gram(v, weights))
    rotation, _, rank, _ = np.linalg.lstsq(c, oriented, rcond=1e-12)
    if rank != c.shape[1] or not np.allclose(c @ rotation, oriented, rtol=1e-10, atol=1e-12):
        raise ValueError("POD coefficient orientation lost rank")
    meta['rotation'] = rotation.tolist()
    return oriented, v @ rotation, meta


def coefficient_pod(velocity_coefficients, velocity_samples, pressure_coefficients,
                    pressure_samples, constant_coefficients, weights):
    """All inputs are dimensionless TRAINING corrections on one FE mesh.

    Coefficients [dof,case], velocity samples [case,point,component], pressure
    samples [case,point]. Returned basis coefficients preserve FE evaluation anywhere.
    """
    vc = np.asarray(velocity_coefficients, dtype=np.float64)
    pc = np.asarray(pressure_coefficients, dtype=np.float64)
    one = np.asarray(constant_coefficients, dtype=np.float64)
    u, um = weighted_pod(velocity_samples, weights, max_rank=8)
    cu = vc @ np.asarray(um['snapshot_combinations'])
    cu, u, orient_u = canonical_pod(cu, u, um['singular_values'][:u.shape[-1]], weights)
    um['eligible_velocity_prefixes'] = [r for r in (2, 4, 8) if r <= u.shape[-1]]
    if not um['eligible_velocity_prefixes']:
        raise ValueError('no registered velocity prefix passes the POD rank guard')
    p, pm = pressure_basis(pressure_samples, weights, 8)
    cp = one[:, None] / np.sqrt(np.sum(weights))
    if p.shape[-1] > 1:
        center_coeff = pc - one[:, None] * np.asarray(pm['snapshot_means'])[None, :]
        tail = center_coeff @ np.asarray(pm['centered_pod']['snapshot_combinations'])
        tail, p_tail, orient_p = canonical_pod(
            tail, p[:, 1:], pm['centered_pod']['singular_values'][:p.shape[-1]-1], weights)
        cp = np.c_[cp, tail]
        p = np.c_[p[:, 0], p_tail]
        pm['orientation'] = orient_p
    um['orientation'] = orient_u
    return {'velocity_coefficients': cu, 'pressure_coefficients': cp,
            'velocity_values': u, 'pressure_values': p,
            'metadata': {'velocity': um, 'pressure': pm}}


def reference_inf_sup(rhs, supremizers, pressure_gram):
    """Full-free-velocity H1 / SAME reduced pressure-L2 reference, not reduced admission."""
    gram = np.asarray(rhs).T @ np.asarray(supremizers)
    lp = checked_cholesky(pressure_gram)
    whitened = np.linalg.solve(lp, gram) @ np.linalg.inv(lp.T)
    if not np.allclose(whitened, whitened.T, rtol=1e-10, atol=1e-12):
        raise ValueError("supremizer reference Gram is not symmetric")
    eigenvalues = np.linalg.eigvalsh((whitened + whitened.T) / 2)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise ValueError("full-velocity pressure coupling is singular")
    return {'beta': float(np.sqrt(eigenvalues[0])), 'squared_singular_values': eigenvalues.tolist(),
            'norms': 'dimensionless velocity H1 / pressure L2',
            'schur_gram': gram.tolist(), 'pressure_gram': np.asarray(pressure_gram).tolist()}
