"""Write the sealed diag05 table and pick the FOM comparator from summary.json."""
from __future__ import annotations

import json
from pathlib import Path


def pct(value):
    return f"{100 * value:.3f}%"


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "runs" / "diag05" / "output" / "summary.json").read_text())
    rom = summary["rom"]
    rom_worst = rom["stats"]["evolved_worst"]
    eligible = []
    for block in summary["fom"].values():
        if block["stats"]["evolved_worst"] <= rom_worst:
            eligible.append(block)
    if not eligible:
        raise SystemExit("no CNAB2 setting is at least as accurate as the tracker")
    comparator = min(eligible, key=lambda block: block["median_ms"])
    speedup = comparator["median_ms"] / rom["median_ms"]
    stats = rom["stats"]
    lines = [
        "# diag05 sealed center tracker",
        "",
        "Generated from `runs/diag05/output/summary.json`. "
        f"Job {summary['job_id']}, commit `{summary['source_commit']}`. "
        f"Device: {summary['device']}. "
        "Seed 202609203, 32 cases, opened once. "
        "Setting frozen before this job: centered POD rank "
        f"{summary['frozen_rank']}, ROM-centroid shift every startup step, "
        f"dt={summary['frozen_dt']}. "
        "Error and time for each method come from the same timed calls. "
        "The comparator is the fastest tested CNAB2 step whose evolved worst "
        "is no larger than the tracker's.",
        "",
        "| method | evolved worst | evolved median | cases over 5% | median ms |",
        "|---|---:|---:|---:|---:|",
        (
            f"| tracked ROM | {pct(stats['evolved_worst'])} | {pct(stats['evolved_median'])} "
            f"| {stats['cases_evolved_over_target']}/{stats['cases']} | {rom['median_ms']:.3f} |"
        ),
    ]
    for block in sorted(summary["fom"].values(), key=lambda item: item["dt"]):
        fst = block["stats"]
        mark = " chosen" if block is comparator else ""
        lines.append(
            f"| CNAB2 dt={block['dt']}{mark} | {pct(fst['evolved_worst'])} "
            f"| {pct(fst['evolved_median'])} | {fst['cases_evolved_over_target']}/{fst['cases']} "
            f"| {block['median_ms']:.3f} |"
        )
    lines += [
        "",
        f"Paired speedup (chosen CNAB2 ms / tracked ms): {speedup:.6f}.",
        "",
        "Tracked per-time worst, including t=0:",
        "",
        "| time index | worst |",
        "|---:|---:|",
    ]
    for index, value in enumerate(stats["per_time_worst"]):
        lines.append(f"| {index} | {pct(value)} |")
    lines.append("")
    destination = root / "results" / "diag05.md"
    destination.write_text("\n".join(lines) + "\n")
    choice = {
        "comparator_dt": comparator["dt"],
        "comparator_median_ms": comparator["median_ms"],
        "comparator_evolved_worst": comparator["stats"]["evolved_worst"],
        "rom_median_ms": rom["median_ms"],
        "rom_evolved_worst": rom_worst,
        "speedup": speedup,
    }
    (root / "results" / "diag05_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
