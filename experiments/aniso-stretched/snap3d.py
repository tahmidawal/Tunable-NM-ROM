"""Training snapshots for the 3D ROM arms: truth = degree-p, N^3 elements, FD-PCG to 1e-9.

Usage: snap3d.py <p> <N> <ntrain>   -> runs/rom3/snap_p{p}N{N}_{ntrain}.npy (float32), .json (timing)
"""
import sys, os, time, json
import numpy as np, jax
import aniso3d as a3, solve3d as s3
HERE = os.path.dirname(os.path.abspath(__file__)); OD = f"{HERE}/runs/rom3"; os.makedirs(OD, exist_ok=True)
p, N, nt = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
P = a3.sample_params("train", nt); S = s3.Solver(N, p)
X = np.zeros((nt, S.n ** 3), np.float32); its = []
S.solve(P[0], tol=1e-9)  # compile
t0 = time.time()
for i in range(nt):
    x, k, res = S.solve(P[i], tol=1e-9, prec="fd"); X[i] = np.asarray(x).ravel(); its.append(k)
    if i % 64 == 0: print(i, k, "%.0fs" % (time.time() - t0), flush=True)
sec = time.time() - t0
np.save(f"{OD}/snap_p{p}N{N}_{nt}.npy", X)
json.dump(dict(p=p, N=N, ntrain=nt, dofs=S.n ** 3, backend=jax.default_backend(), seconds=sec, iters_median=float(np.median(its)), iters_max=int(max(its))),
          open(f"{OD}/snap_p{p}N{N}_{nt}.json", "w"), indent=1)
print("done", sec)
