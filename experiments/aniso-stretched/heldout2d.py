"""Dev-selected 2D configurations scored on the held-out cohort (no re-selection on held-out).
Usage: heldout2d.py <agg_dev.json> <out.json> <test files...>"""
import json, sys, collections
import numpy as np
dev = json.load(open(sys.argv[1])); rows = []
for f in sys.argv[3:]:
    txt = open(f).read()
    rows += json.loads(txt) if txt.lstrip().startswith("[") else [json.loads(l) for l in txt.splitlines() if l.startswith("{")]
g = collections.defaultdict(dict)
for r in rows:
    g[(r["family"], r.get("solver", "hq_direct_p%d" % r.get("p", 1)), r.get("tol"), r.get("p", 1), r["N"], float(r["bx"]), float(r["by"]))][r["case"]] = r
out = []
for fam in dev:
    for tg, d in dev[fam].items():
        for cls, c in d["per_class"].items():
            k = (fam, c["solver"], c["tol"], c["p"], c["N"], float(c["bx"]), float(c["by"])); v = list(g.get(k, {}).values())
            if len(v) < 16: continue
            t = np.array([x["total_ms"] for x in v]); e = np.array([x["l2w"] for x in v])
            row = dict(family=fam, target=float(tg), cls=cls, solver=c["solver"], tol=c["tol"], p=c["p"], N=c["N"], bx=c["bx"], by=c["by"], dofs=c["dofs"],
                       dev_l2w_mean=c["l2w_mean"], dev_t_med=c["t_med"], test_l2w_mean=float(e.mean()), test_l2w_max=float(e.max()),
                       test_t_med=float(np.median(t)), test_t_min=float(t.min()), test_t_max=float(t.max()),
                       test_assemble_ms=float(np.median([x["assemble_ms"] for x in v])), dev_rank=list(d["per_class"]).index(cls))
            out.append(row)
            print("%s %-5g %-7s %-18s tol %-6s p%d N=%-4d b=(%g,%g) dofs %-6d dev %.4f/%.2f ms | test l2w mean %.4f max %.4f  t %.2f ms [%.2f, %.2f] asm %.2f" % (
                fam, float(tg), cls, c["solver"], c["tol"], c["p"], c["N"], c["bx"], c["by"], c["dofs"], c["l2w_mean"], c["t_med"],
                row["test_l2w_mean"], row["test_l2w_max"], row["test_t_med"], row["test_t_min"], row["test_t_max"], row["test_assemble_ms"]))
json.dump(out, open(sys.argv[2], "w"), indent=1)
