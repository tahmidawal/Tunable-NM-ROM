"""Write the sealed diag07 table and choose the CNAB2 comparator from summary.json."""
from __future__ import annotations

import json
from pathlib import Path


def pct(value):
    return f"{100 * value:.3f}%"


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "runs" / "diag07" / "output" / "summary.json").read_text())
    coeff = summary["coeff"]
    rom_worst = coeff["stats"]["evolved_worst"]
    eligible = [
        block for block in summary["fom"].values()
        if block["stats"]["evolved_worst"] <= rom_worst
    ]
    if not eligible:
        raise SystemExit("no CNAB2 setting is at least as accurate as the coefficient ROM")
    comparator = min(eligible, key=lambda block: block["median_ms"])
    speedup = comparator["median_ms"] / coeff["median_ms"]
    stats = coeff["stats"]
    lines = [
        "# diag07 sealed coefficient tracker",
        "",
        "Generated from `runs/diag07/output/summary.json`. "
        f"Job {summary['job_id']}, commit `{summary['source_commit']}`. "
        f"Device: {summary['device']}. "
        "Seed 202609211, 32 cases, opened once. "
        "Setting frozen from diag06: centered POD rank 64, startup step, "
        f"dt={summary['frozen']['dt']}, Fourier tail {summary['frozen']['tail']}. "
        "Error and time for each method come from the same timed calls. "
        "The comparator is the fastest tested CNAB2 step whose evolved worst "
        "is no larger than the coefficient ROM's.",
        "",
        "| method | evolved worst | evolved median | cases over 5% | median ms | parity vs grid |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| coefficient ROM | {pct(stats['evolved_worst'])} | {pct(stats['evolved_median'])} "
            f"| {stats['cases_evolved_over_target']}/{stats['cases']} | {coeff['median_ms']:.3f} "
            f"| {coeff['parity_worst']:.3e} |"
        ),
        (
            f"| grid tracker | {pct(summary['tracker']['stats']['evolved_worst'])} "
            f"| {pct(summary['tracker']['stats']['evolved_median'])} "
            f"| {summary['tracker']['stats']['cases_evolved_over_target']}/"
            f"{summary['tracker']['stats']['cases']} | {summary['tracker']['median_ms']:.3f} | |"
        ),
    ]
    for block in sorted(summary["fom"].values(), key=lambda item: item["dt"]):
        fst = block["stats"]
        mark = " chosen" if block is comparator else ""
        lines.append(
            f"| CNAB2 dt={block['dt']}{mark} | {pct(fst['evolved_worst'])} "
            f"| {pct(fst['evolved_median'])} | {fst['cases_evolved_over_target']}/{fst['cases']} "
            f"| {block['median_ms']:.3f} | |"
        )
    lines += [
        "",
        f"Paired speedup (chosen CNAB2 ms / coefficient ms): {speedup:.6f}.",
        "",
        "Coefficient ROM per-time worst, including t=0:",
        "",
        "| time index | worst |",
        "|---:|---:|",
    ]
    for index, value in enumerate(stats["per_time_worst"]):
        lines.append(f"| {index} | {pct(value)} |")
    lines.append("")
    destination = root / "results" / "diag07.md"
    destination.write_text("\n".join(lines) + "\n")
    choice = {
        "comparator_dt": comparator["dt"],
        "comparator_median_ms": comparator["median_ms"],
        "comparator_evolved_worst": comparator["stats"]["evolved_worst"],
        "rom_median_ms": coeff["median_ms"],
        "rom_evolved_worst": rom_worst,
        "speedup": speedup,
    }
    (root / "results" / "diag07_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
