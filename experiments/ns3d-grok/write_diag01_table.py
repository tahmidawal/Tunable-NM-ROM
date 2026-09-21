"""Write the diag01 development table from summary.json. No hand-typed cells."""
from __future__ import annotations

import json
from pathlib import Path


def pct(value):
    return f"{100 * value:.3f}%"


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "runs" / "diag01" / "output" / "summary.json").read_text())
    lines = [
        "# diag01 development diagnosis",
        "",
        "Generated from `runs/diag01/output/summary.json`. "
        f"Job {summary['job_id']}, commit `{summary['source_commit']}`, {summary['gpu']}. "
        "Development seed only. The final cohort was not opened. "
        "Evolved worst is the worst case of the worst evolved time. "
        "Evolved median is the median across cases of that per-case worst.",
        "",
        "| rank | floor worst | floor over 5% | Galerkin dt=0.001 worst | over 5% | one-interval worst |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in sorted(summary["ranks"].items(), key=lambda item: int(item[0])):
        floor = row["floor"]["stats"]
        gal = row["galerkin_dt_truth"]["stats"]
        lines.append(
            f"| {rank} | {pct(floor['evolved_worst'])} | {floor['cases_evolved_over_target']}/{floor['cases']} "
            f"| {pct(gal['evolved_worst'])} | {gal['cases_evolved_over_target']}/{gal['cases']} "
            f"| {pct(row['one_interval']['worst'])} |"
        )
    last = summary["ranks"][str(max(int(rank) for rank in summary["ranks"]))]
    coarse = last["galerkin_dt_rom"]["stats"]
    lines += [
        "",
        f"Largest rank also at dt={last['galerkin_dt_rom']['dt']}: "
        f"Galerkin evolved worst {pct(coarse['evolved_worst'])}, "
        f"{coarse['cases_evolved_over_target']}/{coarse['cases']} over 5%.",
        "",
        "| rank | oracle-shift evolved worst | cases over 5% | centroid travel worst |",
        "|---:|---:|---:|---:|",
    ]
    for rank, row in sorted(summary["oracle_shift"]["ranks"].items(), key=lambda item: int(item[0])):
        stats = row["stats"]
        lines.append(
            f"| {rank} | {pct(stats['evolved_worst'])} | {stats['cases_evolved_over_target']}/{stats['cases']} "
            f"| {row['centroid_travel']['centroid_travel_worst']:.6f} |"
        )
    lines += ["", "| arm | median ms |", "|---|---:|"]
    for name, block in summary["timing"]["fom"].items():
        lines.append(f"| FOM {name} | {block['median_ms']:.3f} |")
    for name, block in summary["timing"]["galerkin_rollout"].items():
        lines.append(f"| Galerkin rank {name}, dt={block['dt']} | {block['median_ms']:.3f} |")
    weak = summary["timing"]["weak_pod"]
    lines.append(
        f"| weak POD rank {weak['rank']}, dt={weak['dt']}, {weak['test_modes']} tests | {weak['median_ms']:.3f} |"
    )
    lines.append("")
    destination = root / "results" / "diag01.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
