"""Reduce saved sealed-closeout panel results to per-case scalars (post-hoc diagnosis; reads only).

Run on the cluster: python3 reduce_cluster.py <archive_root> <out.json>. Copies no fields, seeds or inputs.
"""
import json, sys
from pathlib import Path

PANELS = ["sealed2d256", "sealed2d512", "sealed2d1024", "sealed2d2048", "sealed2d4096",
          "sealed3d65", "sealed3d129", "sealed3d257"]
KEEP_T = {"gauss256", "fib46368", "fib987", "fib1597", "gauss45", "gauss50", "gauss55",
          "lat4096", "lat32768", "gl20", "gl32"}
root, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for p in PANELS:
    d = json.loads((root / p / "output" / "result.json").read_text())
    cfg = d["config"]
    arms = {}
    for r in d["rows"]:
        a = arms.setdefault(r["arm"], {"family": r["family"], "m": r.get("m"), "cases": {}})
        rec = {"finite": r["finite"]}
        for k in ("S", "ST"):
            rec[k] = r[k]["e"]
            if r["arm"] in KEEP_T:
                rec[k + "_t"] = r[k]["e_t"]
        a["cases"][r["case"]] = rec
    foms = {}
    for r in d["fom_rows"]:
        f = foms.setdefault(r["fom"], {})
        f[r["case"]] = {"converged": r["converged"], "S": r["S"]["e"], "ST": r["ST"]["e"],
                        "S_t": r["S"]["e_t"], "ST_t": r["ST"]["e_t"]}
    res[p] = dict(dim=cfg["dim"], mesh=cfg["mesh"], dt=d["setup"]["dt"], conv_arm=cfg["conv_arm"],
                  anchor_check_arm=cfg["anchor_check_arm"], job_id=d["job_id"], cases=d["cases"],
                  references=d["references"], arms=arms, foms=foms,
                  fom_config=cfg["foms"])
out.write_text(json.dumps(res))
print("wrote", out, out.stat().st_size)
