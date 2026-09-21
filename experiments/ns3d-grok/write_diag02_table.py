"""Write the diag02 development table from summary.json."""
from __future__ import annotations

import json
from pathlib import Path


def pct(value):
    return f"{100 * value:.3f}%"


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "runs" / "diag02" / "output" / "summary.json").read_text())
    lines = [
        "# diag02 frozen initial center",
        "",
        "Generated from `runs/diag02/output/summary.json`. "
        f"Job {summary['job_id']}, commit `{summary['source_commit']}`. "
        "Development only. The per-time oracle recenters every saved truth. "
        "The solved arms shift by the centroid of the initial field and do not look at later truth.",
        "",
        "| rank | oracle worst | frozen-projection worst | Galerkin dt=0.001 | dt=0.004 | dt=0.01 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in sorted(summary["ranks"].items(), key=lambda item: int(item[0])):
        oracle = row["per_time_oracle"]["stats"]["evolved_worst"]
        frozen = row["frozen_c0"]["stats"]["evolved_worst"]
        gals = []
        for key in ("0p0010", "0p0040", "0p0100"):
            gals.append(pct(row["galerkin"][key]["stats"]["evolved_worst"]))
        lines.append(f"| {rank} | {pct(oracle)} | {pct(frozen)} | " + " | ".join(gals) + " |")
    lines += ["", "Frozen-projection worst at each saved time, starting at $t=0$:", ""]
    for rank, row in sorted(summary["ranks"].items(), key=lambda item: int(item[0])):
        series = row["frozen_c0"]["stats"]["per_time_worst"]
        lines.append(f"- rank {rank}: " + ", ".join(pct(value) for value in series))
    lines += ["", "| arm | median ms |", "|---|---:|"]
    for name, block in summary["timing"]["fom"].items():
        lines.append(f"| FOM dt={block['dt']} | {block['median_ms']:.3f} |")
    for name, block in summary["timing"]["shifted_galerkin"].items():
        lines.append(f"| shifted Galerkin {name} | {block['median_ms']:.3f} |")
    lines.append("")
    destination = root / "results" / "diag02.md"
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
