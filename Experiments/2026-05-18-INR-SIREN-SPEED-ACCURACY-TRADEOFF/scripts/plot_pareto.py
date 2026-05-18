#!/usr/bin/env python
"""Render a single Pareto-frontier plot combining multiple CSVs.

Usage:
  python -m scripts.plot_pareto \
    --csv runs/pareto/v1/results_pareto_v1.csv:baseline \
    --csv runs/pareto/fast/results_pareto_fast.csv:fast \
    --extra runs/rom/affine_v1/coarse_500.json:affine_v1 \
    --out runs/plots/pareto.png

X-axis: speedup vs FOM (log scale)
Y-axis: rel-L2 median (log scale)
One marker per cell; non-dominated frontier highlighted as a line.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path, max_rel_l2=1.0):
    pts = []
    for r in csv.DictReader(open(path)):
        rl = float(r["rel_l2_median"]); su = float(r["speedup_vs_fom"])
        if rl > max_rel_l2: continue
        pts.append((su, rl, r))
    return pts


def load_json(path: Path):
    d = json.loads(path.read_text())
    return (d["speedup_median"], d["rom_relL2_median"], d)


def pareto(pts):
    """Return non-dominated points (maximize speedup, minimize rel_l2)."""
    out = []
    for su, rl, _ in pts:
        dom = any(su2 > su and rl2 < rl for su2, rl2, _ in pts)
        if not dom: out.append((su, rl))
    out.sort()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="append", default=[], help="path:label")
    p.add_argument("--extra", action="append", default=[], help="JSON file path:label")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-rel-l2", type=float, default=1.0)
    p.add_argument("--fom-marker", action="store_true",
                   help="Add an FOM-reference vertical line at speedup=1")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    ci = 0
    for spec in args.csv:
        path, label = spec.rsplit(":", 1)
        pts = load_csv(Path(path), max_rel_l2=args.max_rel_l2)
        if not pts: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, alpha=0.4, s=25, color=colors[ci], label=label)
        fr = pareto(pts)
        if len(fr) >= 2:
            ax.plot([f[0] for f in fr], [f[1] for f in fr], "-",
                    color=colors[ci], lw=2)
        ci = (ci + 1) % len(colors)
    for spec in args.extra:
        path, label = spec.rsplit(":", 1)
        su, rl, _ = load_json(Path(path))
        ax.scatter([su], [rl], marker="*", s=200, color=colors[ci],
                   label=label, edgecolor="black", zorder=5)
        ci = (ci + 1) % len(colors)
    if args.fom_marker:
        ax.axvline(1.0, color="gray", lw=1, ls="--", label="FOM (1x)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("speedup vs FOM CG")
    ax.set_ylabel("rel-L2 median")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("INR-SIREN cold-start NM-ROM — speed/accuracy trade-off")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
