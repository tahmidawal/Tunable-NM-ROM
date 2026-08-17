"""Choose the handful of configurations that the SINGLE-GPU CONSOLIDATION job
must re-time.

Why this exists.  The (k, N) surface is measured by one job per (PDE, mesh)
PANEL, all submitted at once.  Inside a panel every timing shares one GPU, so
the per-(PDE, N) Pareto frontier -- whose dominance is computed within a panel
-- is valid as measured.  The SCALING figure, however, compares timings ACROSS
meshes, and the panels landed on different physical GPUs.  So the argmin
configurations that define that figure are re-timed sequentially in ONE job on
ONE GPU, and only those consolidated timings may be used for it.

What is selected, per (pde, method, N):
  * the cheapest configuration reaching each error target in TARGETS,
  * the configuration with the smallest error (the frontier's accuracy end),
  * the configuration with the smallest time (the frontier's speed end).
Duplicates are removed.  Accuracy is GPU-independent, so the SELECTION is made
from the panel results; only the TIME is re-measured.

Usage: python ctol_pick_configs.py [--runs runs] [--out cluster/stage/consolidate_configs.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import ctol_tables as T

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "cluster", "stage",
                                                  "consolidate_configs.json"))
    args = ap.parse_args()
    data = T.load(args.runs)
    pts = T.to_points(data)
    primary = [p for p in pts if p.get("arm") in (None, "primary")]
    picked, seen = [], set()

    def add(p, why):
        key = (p["pde"], p["method"], p["N"], p["k"], p["M"], p["m"], p["tau"])
        if key in seen:
            return
        seen.add(key)
        picked.append(dict(pde=p["pde"], method=p["method"], N=p["N"], k=p["k"],
                           M=p["M"], m=p["m"], tau=p["tau"], why=why,
                           panel_time_ms=p["time_ms"], panel_err=p["err_rel_l2"]))

    for pde, d in sorted(data.items()):
        for N in d["config"]["ns"]:
            for method in ("coord", "pod"):
                sel = [p for p in primary if p["pde"] == pde and p["method"] == method
                       and p["N"] == N and p.get("err_rel_l2") is not None
                       and math.isfinite(p["err_rel_l2"])
                       and p.get("time_ms") is not None]
                if not sel:
                    continue
                for target in T.TARGETS[pde]:
                    b = T.cheapest_reaching(sel, target)
                    if b is not None:
                        add(b, f"cheapest reaching {target:.0e}")
                add(min(sel, key=lambda p: p["err_rel_l2"]), "most accurate")
                add(min(sel, key=lambda p: p["time_ms"]), "fastest")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(picked, open(args.out, "w"), indent=1)
    n_p = sum(1 for p in picked if p["pde"] == "poisson2d")
    print(f"  wrote {args.out}: {len(picked)} configurations "
          f"({n_p} poisson, {len(picked)-n_p} burgers)")
    for p in picked:
        print(f"    {p['pde']:10s} {p['method']:5s} N={p['N']:4d} k={p['k']:2d} "
              f"M={p['M']:3d} m={p['m']:4d} tau={p['tau']:.0e}  ({p['why']}; panel "
              f"{p['panel_time_ms']:.2f} ms, err {p['panel_err']:.3e})")


if __name__ == "__main__":
    main()
