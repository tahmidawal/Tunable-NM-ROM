"""CPU check of linear reducibility: POD projection floor and POD-Galerkin error vs rank (2D, own-mesh truth).

Snapshots and truth on the same mesh (N, beta) so this isolates the reduction error (no discretisation error).
Usage: pod_floor2d.py <family> <N> <ntrain> <out.json>
"""
import json, sys, time
import numpy as np, scipy.sparse.linalg as spla
import aniso as an

fam, N, ntr, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
bx = by = 0.0
def solve(mu):
    x, y, A, b = an.problem(fam, mu, N, bx, by)
    return spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)).solve(b), A, b
r = an.RANGES[fam]
rng = np.random.default_rng(an.SEEDS["train"])
lo = np.array([r["theta"][0], r["loge"][0], r["c"][0], r["c"][0]]); hi = np.array([r["theta"][1], r["loge"][1], r["c"][1], r["c"][1]])
P = lo + rng.random((ntr, 4)) * (hi - lo)
t = time.time(); S = np.stack([solve(mu)[0] for mu in P]); print("snapshots", S.shape, time.time() - t, flush=True)
w = np.outer(an.lumped1d(an.nodes(N, bx))[1:-1], an.lumped1d(an.nodes(N, by))[1:-1]).ravel()
G = (S * w) @ S.T
ev, V = np.linalg.eigh(G); ev, V = ev[::-1], V[:, ::-1]
Phi = (S.T @ V) / np.sqrt(np.maximum(ev, 1e-300))          # w-orthonormal modes
res = dict(family=fam, N=N, ntrain=ntr, rows=[], sv=np.sqrt(np.maximum(ev[:600], 0) / ev[0]).tolist())
for cohort in ("dev",):
    T = an.sample_params(fam, cohort, 16)
    sols = [solve(mu) for mu in T]
    for rk in (16, 32, 64, 128, 256, 384, 512, 768):
        if rk > ntr: break
        B = Phi[:, :rk]
        ep, eg = [], []
        for (u, A, b) in sols:
            nrm = np.sqrt((w * u * u).sum())
            up = B @ (B.T @ (w * u)); ep.append(np.sqrt((w * (up - u) ** 2).sum()) / nrm)
            Ar = B.T @ (A @ B); ug = B @ np.linalg.solve(Ar, B.T @ b); eg.append(np.sqrt((w * (ug - u) ** 2).sum()) / nrm)
        res["rows"].append(dict(cohort=cohort, rank=rk, proj_mean=float(np.mean(ep)), proj_max=float(np.max(ep)), gal_mean=float(np.mean(eg)), gal_max=float(np.max(eg))))
        print(res["rows"][-1], flush=True)
json.dump(res, open(out, "w"))
