"""Actual decoder-output fixed-support NNLS equivalence/performance probe."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from common.decoders import DecoderConfig, decode_grid
from common.quadrature import solve_nnls, nnls_diagnostics
from physics import Grid, smooth_tests, provenance
from data import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--rule-dir', type=Path, required=True)
    ap.add_argument('--boundary', choices=('dirichlet', 'absorbing'), default='dirichlet')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with args.checkpoint.open('rb') as f:
        ck = pickle.load(f)
    dc = DecoderConfig(**ck['config'])
    p = jax.tree.map(jnp.asarray, ck['params'])
    audit = json.loads((args.rule_dir/'audit.json').read_text())
    with np.load(args.rule_dir/'rule.npz') as raw:
        ids = raw['active_ids']
        old_weight = raw['weight']
    grid = Grid(audit['intervals'], args.boundary, args.boundary)
    phi, _, _ = smooth_tests(grid, audit['weak_modes'])
    z = jnp.asarray(ck['Z'][audit['fit_code_indices']])
    decode = jax.jit(lambda pp, zz: jax.lax.map(lambda a: decode_grid(pp, a, grid.n, dc), zz))
    fields = np.asarray(decode(p, z))
    if args.boundary == 'dirichlet':
        fields = fields[:, 1:-1, 1:-1]
    fields = fields.reshape(len(z), -1, 2)
    masses = grid.mass().ravel()
    target = np.einsum('pm,spc,p->scm', phi, fields, masses, optimize=True).reshape(-1)
    scale = np.sqrt(np.einsum('pm,spc,p->scm', phi*phi, fields*fields, masses, optimize=True)/masses.sum()).reshape(-1)
    scale = np.maximum(scale, max(float(np.max(scale))*1e-12, 1e-14))
    design = np.einsum('pm,spc->scmp', phi[ids], fields[:, ids], optimize=True).reshape(len(target), len(ids))/scale[:, None]
    A = np.vstack((np.ones((1, len(ids))), design))
    b = np.r_[masses.sum(), target/scale]
    result = {'provenance': provenance(), 'matrix_shape': list(A.shape),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'fixed_support': True, 'existing_weights': nnls_diagnostics(A, b, old_weight), 'methods': {}}
    weights = {}
    for method in ('direct', 'qr'):
        start = time.perf_counter()
        w, residual = solve_nnls(A, b, method=method, maxiter=20*len(ids))
        seconds = time.perf_counter()-start
        weights[method] = w
        result['methods'][method] = {'seconds': seconds, 'returned_residual': residual, **nnls_diagnostics(A, b, w)}
    direct_r = np.linalg.norm(A@weights['direct']-b)
    qr_r = np.linalg.norm(A@weights['qr']-b)
    relative = abs(qr_r-direct_r)/max(direct_r, 1e-14)
    result['relative_objective_norm_difference'] = relative
    result['relative_weight_difference'] = float(np.linalg.norm(weights['qr']-weights['direct'])/np.linalg.norm(weights['direct']))
    result['passed'] = bool(relative < 1e-8 and all(np.min(w) >= 0 and np.all(np.isfinite(w)) for w in weights.values()))
    # Preserve the actual design's near-dependent columns in an extra bounded
    # probe. Compare fitted values/objectives, since weights may be nonunique.
    near = A[:, :32].copy()
    near[:, -1] = near[:, 0]+1e-9*near[:, -1]
    near_b = near@np.linspace(.001, .002, near.shape[1])+.00001*np.sin(np.arange(len(b)))
    near_results, near_weights = {}, {}
    for method in ('direct', 'qr'):
        w, _ = solve_nnls(near, near_b, method=method, maxiter=10000)
        near_weights[method] = w
        near_results[method] = nnls_diagnostics(near, near_b, w)
    difference = float(np.linalg.norm(near@(near_weights['direct']-near_weights['qr']))/np.linalg.norm(near_b))
    near_results['relative_fitted_value_difference'] = difference
    near_results['passed'] = bool(difference < 1e-8 and all(v['scaled_kkt'] < 1e-10 for k, v in near_results.items() if isinstance(v, dict)))
    result['near_dependent_actual_design'] = near_results
    result['passed'] &= near_results['passed']
    np.savez_compressed(args.out/'weights.npz', direct=weights['direct'], qr=weights['qr'], original=old_weight, active_ids=ids)
    save_json(args.out/'result.json', result)
    if not result['passed']:
        raise RuntimeError('Actual decoder-output NNLS probe failed equivalence')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
