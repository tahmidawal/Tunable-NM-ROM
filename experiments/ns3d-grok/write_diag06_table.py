"""Write the diag06 development table from summary.json."""
from __future__ import annotations

import json
from pathlib import Path


def pct(value):
    return f"{100 * value:.3f}%"


def method_rows(summary):
    rows = [summary["coeff_full"], summary["f32"], *summary["truncations"]]
    return rows


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "runs" / "diag06" / "output" / "summary.json").read_text())
    selection = summary["selection"]
    chosen = next(row for row in method_rows(summary) if row["name"] == selection["name"])
    eligible = [
        block for block in summary["fom"].values()
        if block["stats"]["evolved_worst"] <= chosen["stats"]["evolved_worst"]
    ]
    comparator = min(eligible, key=lambda block: block["median_ms"])
    speedup = comparator["median_ms"] / chosen["median_ms"]
    lines = [
        "# diag06 coefficient-space tracker, development",
        "",
        "Generated from `runs/diag06/output/summary.json`. "
        f"Job {summary['job_id']}, commit `{summary['source_commit']}`. "
        "Development seed 202609202 only. The sealed seed was not opened. "
        "A rewrite is kept when its relative field gap versus the grid tracker "
        "is at most 1e-6 and every case stays within 5%. "
        "Times and errors for each method come from the same timed calls.",
        "",
        "| method | kept | parity worst | evolved worst | cases over 5% | median ms |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| grid tracker | reference | {summary['tracker']['parity_worst']:.3e} "
            f"| {pct(summary['tracker']['stats']['evolved_worst'])} "
            f"| {summary['tracker']['stats']['cases_evolved_over_target']}/"
            f"{summary['tracker']['stats']['cases']} | {summary['tracker']['median_ms']:.3f} |"
        ),
    ]
    for row in method_rows(summary):
        mark = " selected" if row["name"] == selection["name"] else ""
        extra = ""
        if "n_freq" in row:
            extra = f" ({row['n_freq']} freq)"
        lines.append(
            f"| {row['name']}{extra}{mark} | {row['kept']} | {row['parity_worst']:.3e} "
            f"| {pct(row['stats']['evolved_worst'])} "
            f"| {row['stats']['cases_evolved_over_target']}/{row['stats']['cases']} "
            f"| {row['median_ms']:.3f} |"
        )
    lines += [
        "",
        "| FOM dt | evolved worst | median ms |",
        "|---:|---:|---:|",
    ]
    for block in sorted(summary["fom"].values(), key=lambda item: item["dt"]):
        mark = " chosen" if block is comparator else ""
        lines.append(
            f"| {block['dt']}{mark} | {pct(block['stats']['evolved_worst'])} | {block['median_ms']:.3f} |"
        )
    lines += [
        "",
        f"Development paired speedup (chosen CNAB2 ms / selected ms): {speedup:.6f}.",
        "",
        "Piece medians for one trajectory length, separate from the accuracy calls:",
        "",
        "| piece | median ms |",
        "|---|---:|",
    ]
    for name, block in summary["profile"].items():
        lines.append(f"| {name} | {block['median_ms']:.3f} |")
    lines += ["", "Centered-snapshot floor of a low-wavenumber Fourier basis:", "",
              "| rank | evolved worst | cases over 5% |", "|---:|---:|---:|"]
    for rank, block in sorted(summary["fourier_floor"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {rank} | {pct(block['evolved_worst'])} | {block['cases_over']}/16 |"
        )
    lines.append("")
    destination = root / "results" / "diag06.md"
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
