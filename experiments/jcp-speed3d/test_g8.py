"""jcp-speed3d gate G8 (DESIGN A2.3): on Step 2's stored c3grid outputs, the lattice distance of this lane equals Step 2's
Gram-metric distance (||G^T (a - b)|| via the mesh Cholesky factor, relative to ||u0||) to 1e-12 relative, at n = 65.

    python test_g8.py --out checks/test_g8.json
"""
import offline  # noqa: F401

import argparse
import json

import numpy as np

import c3common as C3
import s3common as S

W2 = ('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/c3grid/'
      'archive/output/W_grid.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    z = np.load(W2)
    ka, kb = 'W256|gl24|fs|BE|0.01|K1', 'W256|gl24|ad|LSPG|BE|0.01|tight'
    A_, B_ = z[ka][:4], z[kb][:4]
    bank = C3.load_bank('W1024')
    tb = C3.Tables(65, bank, [256], log=print)
    lean = tb.lean(256)
    tab = C3.C.table(923801, 64)
    n0 = np.array([np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j))) for j in range(4)])
    Lt = np.asarray(lean['L']).T
    gram = (np.linalg.norm(np.einsum('cki,ji->ckj', A_ - B_, Lt), axis=2) / n0[:, None])[:, 1:].max(1)
    G = np.concatenate([np.asarray(g) for g in lean['GTb']], 0)
    lat = S.lattice_dist(A_, B_, G, n0)
    rel = float(np.max(np.abs(gram - lat) / np.maximum(gram, 1e-300)))
    out = dict(gram=gram.tolist(), lattice=lat.tolist(), max_rel=rel, pass_=rel <= 1e-12)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(out)
    if not out['pass_']:
        raise SystemExit(5)


if __name__ == '__main__':
    main()
