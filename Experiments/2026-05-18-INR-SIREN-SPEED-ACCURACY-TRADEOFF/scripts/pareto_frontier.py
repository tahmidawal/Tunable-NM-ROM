#!/usr/bin/env python
"""Extract the Pareto frontier from a results CSV (max speedup at each
accuracy level, or equivalently min rel-L2 at each speedup level).

Usage: python -m scripts.pareto_frontier --csv runs/pareto/v1/results_pareto_v1.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--target-rel-l2", type=float, default=1e-2,
                   help="cells with rel_l2_median > this are dropped (default 1e-2)")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    # Filter to passable accuracy.
    filt = [r for r in rows if float(r["rel_l2_median"]) <= args.target_rel_l2]
    if not filt:
        print(f"No rows with rel_l2_median <= {args.target_rel_l2}")
        return
    # A point (rel_l2, time) is on the Pareto frontier if no other point
    # has both smaller rel_l2 AND smaller time. Compute frontier.
    pts = [(float(r["rel_l2_median"]), float(r["rom_time_ms"]), r) for r in filt]
    frontier = []
    for rl, tm, r in pts:
        dominated = False
        for rl2, tm2, _ in pts:
            if rl2 < rl and tm2 < tm:
                dominated = True
                break
        if not dominated:
            frontier.append((rl, tm, r))
    # Sort frontier by speedup ascending (= rom_time descending).
    frontier.sort(key=lambda x: x[1])
    print(f"# Pareto frontier (rel_l2_median <= {args.target_rel_l2}), "
          f"{len(frontier)} of {len(rows)} cells")
    print(f"{'time_ms':>10} {'speedup':>8} {'rel_l2':>12} {'p90':>12} "
          f"{'c_neq':>6} {'f_neq':>6} {'c_it':>5} {'f_it':>5}")
    for rl, tm, r in frontier:
        print(f"{tm:10.1f} {float(r['speedup_vs_fom']):8.3f} "
              f"{rl:12.3e} {float(r['rel_l2_p90']):12.3e} "
              f"{r['coarse_neq']:>6} {r['fine_neq']:>6} "
              f"{r['coarse_iters']:>5} {r['fine_iters']:>5}")


if __name__ == "__main__":
    main()
