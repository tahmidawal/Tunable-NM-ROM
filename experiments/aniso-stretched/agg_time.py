"""Select, per family and target, the fastest accuracy-matched configuration (PROTOCOL comparator rule).

Usage: agg_time.py <out.json> <file1> [file2 ...]   (files: JSON lists or logs with one JSON row per line)
Group key: (solver, tol, p, N, bx, by). Requires all 16 cases present. Mean l2w over cases must meet target.
Time of a config = median over cases of the per-case median warm total_ms.
"""
import json, sys, collections
import numpy as np
TARGETS = (0.03, 0.01, 0.003, 0.001)
rows = []
for f in sys.argv[2:]:
    txt = open(f).read()
    if txt.lstrip().startswith("["):
        rows += json.loads(txt)
    else:
        for line in txt.splitlines():
            if line.startswith("{"):
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
g = collections.defaultdict(dict)
for r in rows:
    key = (r["family"], r.get("solver", "hq_direct_p%d" % r.get("p", 1)), r.get("tol"), r.get("p", 1), r["N"], r["bx"], r["by"])
    g[key][r["case"]] = r
out = {}
for fam in sorted({k[0] for k in g}):
    cfg = []
    for k, d in g.items():
        if k[0] != fam or len(d) < 16: continue
        v = list(d.values())
        if any(x.get("info", 0) != 0 for x in v): continue
        t = np.array([x["total_ms"] for x in v]); e = np.array([x["l2w"] for x in v])
        cfg.append(dict(solver=k[1], tol=k[2], p=k[3], N=k[4], bx=k[5], by=k[6], dofs=v[0].get("dofs", (k[4] - 1) ** 2),
                        l2w_mean=float(e.mean()), l2w_max=float(e.max()), l2n_mean=float(np.mean([x["l2n"] for x in v])),
                        t_med=float(np.median(t)), t_min=float(t.min()), t_max=float(t.max()),
                        iters_med=(float(np.median([x["iters"] for x in v])) if "iters" in v[0] else None),
                        assemble_ms=float(np.median([x["assemble_ms"] for x in v]))))
    out[fam] = {}
    for tg in TARGETS:
        ok = sorted([c for c in cfg if c["l2w_mean"] <= tg], key=lambda c: c["t_med"])
        best = {}
        for c in ok:   # best per solver class
            cls = "hq" if c["solver"].startswith("hq") else c["solver"].split("_")[0] if not c["solver"].startswith("frozen") else "frozen"
            best.setdefault(cls, c)
        out[fam][str(tg)] = dict(best=ok[0] if ok else None, per_class=best)
        print(f"{fam} target {tg:g}:")
        for cls, c in best.items():
            print("   %-7s %-18s tol %-6s p%d N=%-4d b=(%g,%g) dofs %-6d  l2w mean %.4f max %.4f  t med %.2f ms [%.2f, %.2f]%s" % (
                cls, c["solver"], c["tol"], c["p"], c["N"], c["bx"], c["by"], c["dofs"], c["l2w_mean"], c["l2w_max"], c["t_med"], c["t_min"], c["t_max"],
                "" if c["iters_med"] is None else "  it %.0f" % c["iters_med"]))
json.dump(out, open(sys.argv[1], "w"), indent=1)
