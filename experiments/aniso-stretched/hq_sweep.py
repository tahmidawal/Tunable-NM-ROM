"""Higher-order control (e): degree-p elements, direct solve, error vs stored reference + warm CPU time.

Usage: hq_sweep.py <family> <cohort> <out.json> [configs "p:N:bx:by,..."]   (default: full ladder)
"""
import json, sys, time, os
import numpy as np, scipy.sparse.linalg as spla
import aniso as an, hq
HERE = os.path.dirname(os.path.abspath(__file__))
fam, cohort, out = sys.argv[1], sys.argv[2], sys.argv[3]
if len(sys.argv) > 4:
    cfgs = [(int(a), int(b), float(c), float(d)) for a, b, c, d in (s.split(":") for s in sys.argv[4].split(","))]
else:
    cfgs = [(p, N, bx, by) for p in (2, 3, 4, 6) for N in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64) for (bx, by) in ((0, 0), (0, 1.5), (1.5, 1.5), (0, 2.5))
            if N * p <= 192]
ev = an.Evaluator(); params = an.sample_params(fam, cohort, 16)
def splu(A): return spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
def med(f, rep=5):
    f(); ts = []
    for _ in range(rep):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return r, dict(med_ms=1e3 * float(np.median(ts)), min_ms=1e3 * min(ts), max_ms=1e3 * max(ts))
rows = []
refs = [np.load(f"{HERE}/runs/ref/{fam}_{cohort}_{i:02d}.npy").astype(np.float64) for i in range(16)]
for (p, N, bx, by) in cfgs:
    Px, Py = hq.interp1d(N, bx, p, ev.x), hq.interp1d(N, by, p, ev.x)
    for i in range(16):
        t = time.perf_counter(); A, F, n1 = hq.problem(fam, params[i], N, bx, by, p); tasm = time.perf_counter() - t
        Ac = A.tocsc()
        lu, tf = med(lambda: splu(Ac), 3)
        u, ts = med(lambda: lu.solve(F), 3)
        (Ue, tint) = med(lambda: hq.to_eval(u, n1, Px, Py), 1)
        rows.append(dict(family=fam, cohort=cohort, case=i, p=p, N=N, bx=bx, by=by, dofs=int(A.shape[0]), nnz=int(A.nnz), assemble_ms=1e3 * tasm,
                         factor=tf, solve=ts, total_ms=tf["med_ms"] + ts["med_ms"], interp_ms=tint["med_ms"], **ev.errors(Ue, refs[i])))
    r = [x for x in rows if (x["p"], x["N"], x["bx"], x["by"]) == (p, N, bx, by)]
    print(fam, "p=%d N=%d b=(%g,%g) dofs %d  l2w mean %.5f max %.5f  time med %.2f ms" % (p, N, bx, by, r[0]["dofs"], np.mean([x["l2w"] for x in r]), np.max([x["l2w"] for x in r]), np.median([x["total_ms"] for x in r])), flush=True)
json.dump(rows, open(out, "w"))
