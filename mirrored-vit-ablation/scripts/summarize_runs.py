#!/usr/bin/env python
"""Summarize all completed mirrored-ViT runs (heat + poisson).

Walks heat/runs/* and poisson/runs/* for train.log and rom.log files,
extracts best val rel-L2 and median speedup, prints a table sorted by
val rel-L2 within each PDE.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_train(log: Path):
    if not log.exists():
        return None
    text = log.read_text()
    m = re.search(r"best val:\s*([0-9.eE+-]+)", text)
    return float(m.group(1)) if m else None


def parse_rom(log: Path):
    if not log.exists():
        return {"mean_rel_l2": None, "median_speedup": None}
    text = log.read_text()
    m1 = re.search(r"Mean rel_l2:\s*([0-9.eE+-]+)", text)
    m2 = re.search(r"Median speedup:\s*([0-9.]+)x", text)
    return {
        "mean_rel_l2": float(m1.group(1)) if m1 else None,
        "median_speedup": float(m2.group(1)) if m2 else None,
    }


def walk(pkg: str):
    rows = []
    pkg_runs = ROOT / pkg / "runs"
    if not pkg_runs.exists():
        return rows
    for d in sorted(pkg_runs.iterdir()):
        if not d.is_dir():
            continue
        train_v = parse_train(d / "train.log")
        rom = parse_rom(d / "rom.log")
        rows.append({"config": d.name, "best_val": train_v, **rom})
    return rows


def main():
    for pkg, baseline in [("poisson", (6.62e-3, 111.0)), ("heat", (1.76e-2, 269.0))]:
        rows = walk(pkg)
        if not rows:
            print(f"\n=== {pkg.upper()} === (no runs)\n")
            continue
        print(f"\n=== {pkg.upper()} (baseline LinearCP/CP: rel-L2={baseline[0]:.3e}, speedup={baseline[1]:.1f}x) ===")
        print(f"{'config':<35} {'best_val':>12} {'rom_rel_l2':>12} {'median_x':>10}")
        rows_sorted = sorted(rows, key=lambda r: r["best_val"] or float("inf"))
        for r in rows_sorted:
            bv = f"{r['best_val']:.3e}" if r["best_val"] is not None else "—"
            rl = f"{r['mean_rel_l2']:.3e}" if r["mean_rel_l2"] is not None else "—"
            sp = f"{r['median_speedup']:.1f}x" if r["median_speedup"] is not None else "—"
            print(f"{r['config']:<35} {bv:>12} {rl:>12} {sp:>10}")
    print()


if __name__ == "__main__":
    main()
