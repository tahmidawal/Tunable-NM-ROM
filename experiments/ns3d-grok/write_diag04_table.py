"""Write the diag04 development table from summary.json."""
from __future__ import annotations

import json
from pathlib import Path


def pct(value):
    return f"{100 * value:.3f}%"


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "runs" / "diag04" / "output" / "summary.json").read_text())
    lines = [
        "# diag04 fused center tracker, development",
        "",
        "Generated from `runs/diag04/output/summary.json`. "
        f"Job {summary['job_id']}, commit `{summary['source_commit']}`. "
        "Development only. Errors match the diag03 ROM rows. "
        "Times are medians of retained repetitions after one discarded warmup, "
        "on this job. They are separate calls from the accuracy loop.",
        "",
        "| rank | dt | evolved worst | cases over 5% | median ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rank, row in sorted(summary["ranks"].items(), key=lambda item: int(item[0])):
        for tag, block in row.items():
            stats = block["stats"]
            timing = summary["timing"]["tracked"][f"r{rank}_{tag}"]["median_ms"]
            lines.append(
                f"| {rank} | {block['dt']} | {pct(stats['evolved_worst'])} "
                f"| {stats['cases_evolved_over_target']}/{stats['cases']} | {timing:.3f} |"
            )
    lines += ["", "| FOM dt | median ms |", "|---:|---:|"]
    for block in summary["timing"]["fom"].values():
        lines.append(f"| {block['dt']} | {block['median_ms']:.3f} |")
    lines.append("")
    destination = root / "results" / "diag04.md"
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
