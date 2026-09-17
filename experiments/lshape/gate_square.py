"""G-FOM-6: square fidelity gate (DESIGN.md section 2), run locally before the first job.

With the geometry switched to the plain square at 255 intervals, this cell's sparse-direct
field solve and QR-based bank projection floor, applied to the parent lane's incumbent
checkpoint with the parent's `bc_poly` factor, must reproduce the parent's audited per-case
development bank floors (pbh01 `D1_bank_floor_development` at 255 intervals) to <= 1e-9
relative. Also checks the sparse-direct solve against the DST solve of the same source.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.fft import dstn
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import sep_common as sc
import lsh_core as K_


def dst_solve(intervals, f_int):
    p = np.arange(1, intervals)
    l = 4.0 * intervals ** 2 * np.sin(np.pi * p / (2 * intervals)) ** 2
    lam = l[:, None] + l[None, :]
    c = dstn(f_int.reshape(intervals - 1, intervals - 1), type=1, norm='ortho', workers=1)
    return dstn(c / lam, type=1, norm='ortho', workers=1).ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--incumbent', required=True)
    ap.add_argument('--reference', required=True, help='pbh01 result.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    ref = json.loads(Path(a.reference).read_text())
    mesh = next(m for m in ref['diagnosis'][0]['meshes'] if m['intervals'] == 255)
    expected = np.asarray(mesh['D1_bank_floor_development']['per_case'])
    dev = np.asarray(ref['development']['parameters'])
    assert K_.sha_array(dev) == ref['development']['sha256']
    n = 255
    geom = K_.Geometry(n, 'square')
    fom = K_.FOM(geom, build_ilu=False)
    gates = fom.gates()
    U = K_.fields(fom, dev)
    F = np.stack([K_.source_interior(geom, q) for q in dev])
    dst = np.stack([dst_solve(n, f) for f in F])
    dst_parity = float(max(K_.relative(U[i], dst[i]) for i in range(len(dev))))
    params, Z, ck = sc.load_pkl(a.incumbent)
    assert K_.weights_sha(params) == ref['diagnosis'][0]['weights_sha256']
    features = K_.make_features('poly', 0)
    G = K_.bank_of(features, params, geom)
    Rg, rank = K_.bank_r(G)
    assert rank['rank_valid'], rank
    got = K_.bank_floor(G, Rg, U)
    worst = float(np.max(np.abs(got - expected) / expected))
    # pure-NumPy twin of the floor for the audit path
    Gn = K_.features_np(jax.tree_util.tree_map(np.asarray, params), geom.coords, 'poly', 0)
    Qn, _ = np.linalg.qr(Gn)
    got_np = np.linalg.norm(U - (U @ Qn) @ Qn.T, axis=1) / np.linalg.norm(U, axis=1)
    worst_np = float(np.max(np.abs(got_np - expected) / expected))
    out = dict(gate='G-FOM-6 square fidelity', intervals=n, operator_gates=gates, dst_parity=dst_parity,
               expected_per_case=expected.tolist(), got_per_case=got.tolist(), got_numpy_per_case=got_np.tolist(),
               worst_relative_difference=worst, worst_relative_difference_numpy=worst_np, tolerance=1e-9,
               passed=bool(worst <= 1e-9 and worst_np <= 1e-9 and dst_parity <= 1e-10 and gates['symmetric']
                           and gates['independent_assembly_agrees']),
               incumbent_weights_sha256=K_.weights_sha(params), bank_rank=rank['rank'])
    K_.dump(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if 'per_case' not in k}, indent=1))
    assert out['passed']


if __name__ == '__main__':
    main()
