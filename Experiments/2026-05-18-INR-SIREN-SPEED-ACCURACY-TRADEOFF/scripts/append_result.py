#!/usr/bin/env python
"""Append a row to results.csv from a ROM-eval JSON file.

Usage:
  python -m scripts.append_result --json runs/rom/X.json --label foo --desc "..."
"""
from __future__ import annotations
import argparse, csv, json, subprocess
from pathlib import Path


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--desc", required=True)
    p.add_argument("--csv", type=Path,
                   default=Path(__file__).parent.parent / "results.csv")
    args = p.parse_args()

    d = json.loads(args.json.read_text())
    row = {
        "commit": git_commit(),
        "label": args.label,
        "speedup_vs_fom": f"{d.get('speedup_median', float('nan')):.3f}",
        "rel_l2_median": f"{d.get('rom_relL2_median', float('nan')):.3e}",
        "rel_l2_p90": f"{d.get('rom_relL2_p90', float('nan')):.3e}",
        "frac_le_1e-2": f"{d.get('frac_le_1e-2', float('nan')):.3f}",
        "frac_le_5e-3": f"{d.get('frac_le_5e-3', float('nan')):.3f}",
        "rom_time_ms": f"{1000 * d.get('rom_time_median', float('nan')):.1f}",
        "fom_time_ms": f"{1000 * d.get('fom_time_median', float('nan')):.1f}",
        "iters_coarse_med": d.get("iters_coarse_median", ""),
        "iters_fine_med": d.get("iters_fine_median", ""),
        "change_description": args.desc,
    }
    cols = list(row.keys())
    with open(args.csv, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writerow(row)
    print(f"appended: {row['label']}  rel_l2={row['rel_l2_median']}  "
          f"speedup={row['speedup_vs_fom']}  desc={row['change_description']}")


if __name__ == "__main__":
    main()
