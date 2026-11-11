"""3D pilot (PROTOCOL amendment 1): references, then error + warm time for every (p, N, preconditioner, tol).

Usage: sweep3d.py ref <cohort>
       sweep3d.py time <cohort> <out.jsonl> [configs "p:N,..."] [precs "fd,jacobi,none"] [tols "1e-2,..."]
Backend is whatever JAX runs on (recorded in every row).
"""
import json, sys, os, time
import numpy as np, jax
import aniso3d as a3, solve3d as s3
HERE = os.path.dirname(os.path.abspath(__file__)); RD = f"{HERE}/runs/ref3"; os.makedirs(RD, exist_ok=True)
mode, cohort = sys.argv[1], sys.argv[2]
P = a3.sample_params(cohort, 16); ev = a3.Evaluator3(128); backend = jax.default_backend()
if mode == "ref":
    S32, S24 = s3.Solver(32, 4), s3.Solver(24, 4); rows = []
    for i in range(16):
        t = time.time(); x, k, res = S32.solve(P[i], tol=1e-10, prec="fd"); U = ev.field(S32.sp, np.asarray(x))
        np.save(f"{RD}/A3_{cohort}_{i:02d}.npy", U.astype(np.float32))
        x2, k2, _ = S24.solve(P[i], tol=1e-10, prec="fd"); e = ev.errors(ev.field(S24.sp, np.asarray(x2)), U)
        rows.append(dict(case=i, mu=P[i].tolist(), iters=k, res=res, p4N24_vs_ref=e, seconds=time.time() - t))
        print(i, "it", k, "p4N24 vs ref l2w %.5f linf %.5f" % (e["l2w"], e["linf"]), "%.0fs" % (time.time() - t), flush=True)
    json.dump(rows, open(f"{HERE}/runs/ref3_{cohort}.json", "w"), indent=1)
    sys.exit()
out = sys.argv[3]
cfgs = [tuple(map(int, s.split(":"))) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 and sys.argv[4] else \
    [(1, N) for N in (16, 24, 32, 48, 64, 96, 128)] + [(2, N) for N in (8, 12, 16, 24, 32, 48)] + [(4, N) for N in (4, 6, 8, 12, 16, 24)]
precs = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else ["fd", "jacobi", "none"]
tols = [float(s) for s in sys.argv[6].split(",")] if len(sys.argv) > 6 else [1e-2, 1e-3, 1e-4, 1e-6]
refs = [np.load(f"{RD}/A3_{cohort}_{i:02d}.npy").astype(np.float64) for i in range(16)]
fo = open(out, "a")
import jax.numpy as jnp
dts = os.environ.get("DTYPES", "float32,float64").split(",")
for (p, N), dt in [(c, d) for c in cfgs for d in dts]:
    S = s3.Solver(N, p, dtype=getattr(jnp, dt)); sfx = "_f32" if dt == "float32" else ""
    for prec in precs:
        if dt == "float32" and prec == "none": continue
        if prec != "fd" and S.n ** 3 > 300000: continue
        for tol in tols:
            rr = []
            if dt == "float32" and tol < 1e-4: continue
            for i in range(16):
                x, k, tm = S.timed(P[i], tol, prec, rep=5, maxit=5000)
                t = time.perf_counter(); U = ev.field(S.sp, x); ti = time.perf_counter() - t
                r = dict(family="A3", cohort=cohort, case=i, backend=backend, p=p, N=N, dofs=S.n ** 3, solver=f"{prec}_pcg{sfx}", tol=tol, iters=k,
                         solve=tm, total_ms=tm["med_ms"], interp_ms=1e3 * ti, **ev.errors(U, refs[i]))
                rr.append(r); fo.write(json.dumps(r) + "\n"); fo.flush()
            print("p=%d N=%-3d dofs %-8d %-10s tol %-6g it med %4.0f  l2w mean %.5f max %.5f  t med %.1f ms" % (
                p, N, S.n ** 3, prec + sfx, tol, np.median([r["iters"] for r in rr]), np.mean([r["l2w"] for r in rr]), np.max([r["l2w"] for r in rr]),
                np.median([r["total_ms"] for r in rr])), flush=True)
