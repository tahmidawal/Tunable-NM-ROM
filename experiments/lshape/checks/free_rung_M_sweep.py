"""Untimed error of the q = R free-bank rung on the REAL lsh02 banks at N=256, as a function
of the test-mode count M, in pure NumPy/SciPy (DESIGN.md A6, run before the M=513 job).

The free rung is the least-squares solution of B c = f_m with B = Lambda^{-1} Phi^T A G
(M x R) over ALL R bank coefficients; it is head-independent, so this is exactly the error
the solve job will report for it, computed ahead of the job.  One shift-invert eigensolve
at k = 1024 gives every prefix M at once (the lowest M eigenpairs are the same set).
"""
import json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse.linalg as spla
sys.path.insert(0, 'experiments/lshape')
import lsh_audit_np as L

# result.json is the archived copy; the bank checkpoints sit in the collected output tree
out = Path('experiments/lshape/runs/lsh02/archive/output'); d = json.loads((out / 'result.json').read_text())
cfg = d['config']; N = cfg['training_intervals']
g = L.Geom(N); S = L.Solver(g); A = S.A
dc = cfg['cohorts']['development']; dev = L.cohort(seed=dc['seed'], draw=dc['draw'], count=dc['count'])
F = np.stack([L.source_int(g, q) for q in dev]); U = np.stack([S.ref(f)[0] for f in F])
KMAX = 1024
t0 = time.time()
cache = Path(sys.argv[1]) if len(sys.argv) > 1 else None   # optional npz cache of the eigenpairs
if cache is not None and cache.exists():
    z = np.load(cache); lam, phi = z['lam'], z['phi']
else:
    opinv = spla.LinearOperator((g.n, g.n), matvec=S.lu.solve, dtype=float)
    lam, phi = spla.eigsh(A, k=KMAX, sigma=0.0, which='LM', OPinv=opinv, v0=np.random.default_rng(0).standard_normal(g.n), tol=0)
    o = np.argsort(lam); lam, phi = lam[o], phi[:, o]
    if cache is not None:
        np.savez(cache, lam=lam, phi=phi)
print(f'eigsh k={KMAX}: {time.time()-t0:.0f}s, lambda_1={lam[0]:.6f}, lambda_{KMAX}={lam[-1]:.1f}, '
      f'eig resid {np.linalg.norm(A @ phi - phi * lam) / np.linalg.norm(phi * lam):.1e}', flush=True)
rows = []
for arm in ('sdf_R512', 'smooth_R512', 'enrich_R512'):
    ck = next(x for x in d['checkpoints'] if x['id'] == arm)
    params, Z, c = L.load(out / ck['path'])
    G = L.features(params, g.coords, c['factor'], c['n_enrich']); R = G.shape[1]
    Q, _ = np.linalg.qr(G)
    floor = np.linalg.norm(U - (U @ Q) @ Q.T, axis=1) / np.linalg.norm(U, axis=1)
    AG = A @ G
    for M in (R + 1, 640, 768, 1024):
        P = (phi[:, :M] / lam[None, :M]).T
        B = P @ AG; Fm = F @ P.T
        C = np.linalg.lstsq(B, Fm.T, rcond=None)[0]
        err = np.linalg.norm(U - C.T @ G.T, axis=1) / np.linalg.norm(U, axis=1)
        s = np.linalg.svd(B, compute_uv=False)
        rows.append(dict(arm=arm, R=R, M=M, free_worst=float(err.max()), free_median=float(np.median(err)),
                         floor_worst=float(floor.max()), floor_median=float(np.median(floor)),
                         ratio_worst=float(err.max() / floor.max()), cond_B=float(s[0] / s[-1])))
        r = rows[-1]
        print(f"{arm:<12} R={R} M={M:>5}  free worst {r['free_worst']*100:8.4f}%  median {r['free_median']*100:8.4f}%  "
              f"| floor worst {r['floor_worst']*100:7.4f}%  ratio {r['ratio_worst']:6.2f}x  cond(B) {r['cond_B']:.2e}", flush=True)
json.dump(rows, open('experiments/lshape/checks/free_rung_M_sweep.json', 'w'), indent=1)
print('DONE')
