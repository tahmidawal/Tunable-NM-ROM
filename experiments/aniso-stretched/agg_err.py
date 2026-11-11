"""Aggregate the discretisation-error sweep: per family/mesh, mean and worst error over a cohort."""
import json, glob, sys, collections
import numpy as np
cohort = sys.argv[1] if len(sys.argv) > 1 else "dev"
out = {}
for fam in "AB":
    files = sorted(glob.glob(f"runs/err/{fam}_{cohort}_*.json"))
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        for r in json.load(open(f))["rows"]:
            for k in ("l2w", "l2n", "linf"):
                d[(r["N"], r["bx"], r["by"])][k].append(r[k])
    rows = [dict(N=k[0], bx=k[1], by=k[2], n=len(v["l2w"]), **{m + "_mean": float(np.mean(v[m])) for m in v}, **{m + "_max": float(np.max(v[m])) for m in v}) for k, v in d.items()]
    out[fam] = rows
    print(f"family {fam}: {len(files)} cases")
    for metric in ("l2w", "l2n"):
        for tgt in (0.03, 0.01, 0.003, 0.001):
            ok = sorted([r for r in rows if r[metric + "_mean"] <= tgt], key=lambda r: r["N"])
            s = "; ".join("N=%d b=(%g,%g) mean %.4f max %.4f" % (r["N"], r["bx"], r["by"], r[metric + "_mean"], r[metric + "_max"]) for r in ok[:3])
            print(f"  {metric} mean<={tgt}: {s}")
json.dump(out, open(f"runs/agg_err_{cohort}.json", "w"), indent=0)
