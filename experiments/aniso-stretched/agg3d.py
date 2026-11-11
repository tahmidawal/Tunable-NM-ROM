"""Aggregate 3D timing rows (JSON lines). Usage: agg3d.py <out.json> <ncases> <files...>
Groups by (backend, solver, tol, p, N); needs ncases cases; per target picks the fastest configuration (median over
cases of the per-case median warm time) with cohort-mean l2w <= target, overall per backend and per solver."""
import sys, json, collections
import numpy as np
out, nc, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
G = collections.defaultdict(dict)
for f in files:
    for line in open(f):
        r = json.loads(line)
        G[(r["backend"], r["solver"], r.get("tol"), r["p"], r["N"])][r["case"]] = r
cfgs = []
for k, d in G.items():
    if len(d) < nc: continue
    rr = list(d.values()); t = np.array([r["total_ms"] for r in rr]); e = np.array([r["l2w"] for r in rr])
    cfgs.append(dict(backend=k[0], solver=k[1], tol=k[2], p=k[3], N=k[4], dofs=rr[0]["dofs"], ncases=len(rr),
                     l2w_mean=float(e.mean()), l2w_max=float(e.max()), t_med=float(np.median(t)), t_min=float(t.min()),
                     t_max=float(t.max()), iters_med=float(np.median([r.get("iters") or 0 for r in rr])),
                     interp_ms=float(np.median([r.get("interp_ms", np.nan) for r in rr]))))
res = dict(configs=cfgs, best={})
for tgt in (0.03, 0.01, 0.003):
    ok = [c for c in cfgs if c["l2w_mean"] <= tgt]
    for b in sorted({c["backend"] for c in cfgs}):
        okb = sorted([c for c in ok if c["backend"] == b], key=lambda c: c["t_med"])
        if not okb: print(f"target {tgt} backend {b}: none"); continue
        per = {}
        for c in okb: per.setdefault(c["solver"], c)
        res["best"][f"{tgt}|{b}"] = dict(overall=okb[0], per_solver=per)
        print(f"target {tgt} backend {b}:")
        for s, c in per.items():
            print("   %-16s p=%d N=%-3d dofs %-8d tol %-6s  %.2f ms [%.2f, %.2f]  it %g  l2w mean %.4f max %.4f" % (
                s, c["p"], c["N"], c["dofs"], c["tol"], c["t_med"], c["t_min"], c["t_max"], c["iters_med"], c["l2w_mean"], c["l2w_max"]))
json.dump(res, open(out, "w"), indent=1)
