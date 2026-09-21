"""Write the diag03 development table from summary.json."""
from __future__ import annotations

import json
from pathlib import Path


def pct(value):
    return f"{100 * value:.3f}%"


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "runs" / "diag03" / "output" / "summary.json").read_text())
    lines = [
        "# diag03 every-step center tracking",
        "",
        "Generated from `runs/diag03/output/summary.json`. "
        f"Job {summary['job_id']}, commit `{summary['source_commit']}`. "
        "Development only. `rom` recenters from the decoded field. "
        "`truth` recenters from the true field and is not a model. "
        "`interval` recenters only at the saved output times. "
        "Each substep is the Galerkin startup step, not multistep CNAB2. "
        "The timing column is that Python loop, not a fused query.",
        "",
        "| rank | mode | dt | evolved worst | cases over 5% |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in sorted(summary["ranks"].items(), key=lambda item: int(item[0])):
        for mode, by_dt in row.items():
            for block in by_dt.values():
                stats = block["stats"]
                lines.append(
                    f"| {rank} | {mode} | {block['dt']} | {pct(stats['evolved_worst'])} "
                    f"| {stats['cases_evolved_over_target']}/{stats['cases']} |"
                )
    lines += ["", "| arm | median ms |", "|---|---:|"]
    for name, block in summary["timing"].items():
        lines.append(f"| Python ROM tracker rank {block['rank']} dt={block['dt']} | {block['median_ms']:.3f} |")
    lines.append("")
    destination = root / "results" / "diag03.md"
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
