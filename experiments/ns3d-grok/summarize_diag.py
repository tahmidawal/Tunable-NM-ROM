"""Print the diag01 decision numbers from summary.json. Does not recompute fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def brief(stats):
    return dict(
        initial_worst=stats["initial_worst"],
        evolved_median=stats["evolved_median"],
        evolved_worst=stats["evolved_worst"],
        evolved_over_target=stats["evolved_over_target"],
        cases=stats["cases"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    rows = []
    for rank, row in summary["ranks"].items():
        item = dict(
            rank=int(rank),
            floor=brief(row["floor"]["stats"]),
            galerkin_dt_truth=brief(row["galerkin_dt_truth"]["stats"]),
            one_interval_worst=row["one_interval"]["worst"],
            one_interval_per_step_worst=row["one_interval"]["per_interval_worst"],
        )
        if "galerkin_dt_rom" in row:
            item["galerkin_dt_rom"] = brief(row["galerkin_dt_rom"]["stats"])
        item["affine_pca"] = {
            width: brief(block["stats"]) for width, block in row["affine_pca_head"].items()
        }
        rows.append(item)
    shift = {
        rank: brief(block["stats"])
        for rank, block in summary.get("oracle_shift", {}).get("ranks", {}).items()
    }
    timing = summary.get("timing", {})
    print(json.dumps(dict(
        final_cohort_opened=summary.get("final_cohort_opened"),
        source_commit=summary.get("source_commit"),
        job_id=summary.get("job_id"),
        gpu=summary.get("gpu"),
        ranks=rows,
        oracle_shift=shift,
        timing_medians_ms=dict(
            fom={name: block.get("median_ms") for name, block in timing.get("fom", {}).items()},
            galerkin={name: block.get("median_ms") for name, block in timing.get("galerkin_rollout", {}).items()},
            weak=timing.get("weak_pod", {}).get("median_ms"),
        ),
    ), indent=2))


if __name__ == "__main__":
    main()
