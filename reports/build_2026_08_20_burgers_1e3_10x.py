"""Generate the Phase-1 Burgers negative-result report, tables, and figures."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.hashsalt"] = "burgers-1e3-10x-2026-08-20"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.join(ROOT, "experiments", "burgers-1e3-10x")
D0_PATH = os.path.join(EXP, "runs", "d0_r2", "out", "d0.json")
DECISION_PATH = os.path.join(EXP, "runs", "d0_r2", "d0_decision.json")
FOM_PATH = os.path.join(EXP, "runs", "fom_cal_r4", "out", "fom_cal.json")
AUDIT_PATH = os.path.join(ROOT, "reports", "generated", "burgers_1e3_10x_audit.json")
TABLE_PATH = os.path.join(ROOT, "reports", "generated", "burgers_1e3_10x_tables.json")
REPORT_PATH = os.path.join(ROOT, "reports", "2026-08-20-burgers-1e3-10x.md")
FIG_DIR = os.path.join(ROOT, "reports", "figures")


def load(path):
    with open(path) as handle:
        return json.load(handle)


def sci(value, digits=3):
    return f"{value:.{digits}e}"


def ms(value):
    return f"{1e3 * value:.3f}"


def table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def normalize_svg(path):
    """Remove backend whitespace while preserving deterministic SVG bytes."""
    with open(path) as handle:
        lines = handle.readlines()
    with open(path, "w") as handle:
        handle.writelines(line.rstrip() + "\n" for line in lines)


def make_figures(decision, fom):
    os.makedirs(FIG_DIR, exist_ok=True)
    names = ["HG4", "HG5"]
    means = [decision["concepts"][name]["representation_oracle_mean"] for name in names]
    worst = [decision["concepts"][name]["representation_oracle_worst"] for name in names]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(names))
    ax.bar(x - 0.16, means, width=0.32, label="oracle mean", color="#4C78A8")
    ax.bar(x + 0.16, worst, width=0.32, label="oracle worst", color="#F58518")
    ax.axhline(2e-4, color="#4C78A8", linestyle="--", linewidth=1.2, label="mean gate")
    ax.axhline(7e-4, color="#F58518", linestyle=":", linewidth=1.4, label="worst gate")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("trajectory relative L2")
    ax.set_title("Transported Hermite representation floor")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    rep_base = os.path.join(FIG_DIR, "burgers_1e3_10x_representation_floor")
    fig.savefig(rep_base + ".png", dpi=180)
    fig.savefig(rep_base + ".svg", metadata={"Date": None})
    normalize_svg(rep_base + ".svg")
    plt.close(fig)

    ns = np.asarray((256, 512, 1024))
    medians, lower, upper = [], [], []
    for n in ns:
        mesh = fom["meshes"][str(n)]
        summary = mesh["rows"][mesh["selected_fastest_eligible"]]["summary"]
        medians.append(summary["median_elapsed_s"])
        lower.append(summary["clustered_median_elapsed_ci_s"][0])
        upper.append(summary["clustered_median_elapsed_ci_s"][1])
    medians, lower, upper = map(np.asarray, (medians, lower, upper))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.errorbar(ns, 1e3 * medians, yerr=(1e3 * (medians - lower), 1e3 * (upper - medians)),
                marker="o", capsize=4, linewidth=1.8, color="#4C78A8")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("N")
    ax.set_ylabel("50-step FOM median (ms)")
    ax.set_title("Fastest accuracy-eligible cubic/Helmholtz FOM")
    ax.grid(which="both", alpha=0.25)
    fig.tight_layout()
    fom_base = os.path.join(FIG_DIR, "burgers_1e3_10x_fom_scaling")
    fig.savefig(fom_base + ".png", dpi=180)
    fig.savefig(fom_base + ".svg", metadata={"Date": None})
    normalize_svg(fom_base + ".svg")
    plt.close(fig)
    return {
        "representation": os.path.relpath(rep_base + ".png", os.path.dirname(REPORT_PATH)),
        "fom_scaling": os.path.relpath(fom_base + ".png", os.path.dirname(REPORT_PATH)),
    }


def main():
    d0 = load(D0_PATH)
    decision = load(DECISION_PATH)
    fom = load(FOM_PATH)
    audit = load(AUDIT_PATH)
    assert audit["status"] == "pass" and decision["hard_stop_now"]
    figures = make_figures(decision, fom)

    representation_rows = []
    for name in ("HG4", "HG5"):
        row = decision["concepts"][name]
        raw = d0["concepts"][name]
        representation_rows.append({
            "concept": name,
            "k": raw["latent_dimension"],
            "oracle_mean": row["representation_oracle_mean"],
            "oracle_worst": row["representation_oracle_worst"],
            "predictor_mean": row["predictor_decoder_mean"],
            "predictor_worst": row["predictor_decoder_worst"],
            "oracle_to_predictor": row["oracle_to_predictor_mean_ratio"],
            "oracle_wall_fraction": row[
                "nearest_wall_squared_representation_oracle_error_fraction"
            ],
            "representation_pass": row["representation_gate_pass"],
            "wall_license_pass": row["wall_license_pass"],
            "selection_condition_median": row["basis_condition"]["selection_median"],
            "selection_condition_worst": row["basis_condition"]["selection_worst"],
        })

    fom_rows = []
    for n in (256, 512, 1024):
        mesh = fom["meshes"][str(n)]
        label = mesh["selected_fastest_eligible"]
        row = mesh["rows"][label]
        summary = row["summary"]
        setup = row["setup"]
        reference = mesh["reference_health"]
        tighter = mesh["independent_tighter_reference_difference"]
        fom_rows.append({
            "N": n, "selected": label,
            "mean_error": summary["trajectory_error_mean"],
            "worst_error": summary["trajectory_error_worst"],
            "median_s": summary["median_elapsed_s"],
            "clustered_ci_s": summary["clustered_median_elapsed_ci_s"],
            "mean_newton": summary["mean_newton_total"],
            "mean_linear": summary["mean_linear_total"],
            "max_residual": summary["max_returned_relative_residual"],
            "python_build_s": setup["python_build_s"],
            "first_call_s": setup["first_call_compile_and_solve_s"],
            "reference_max_residual": reference["max_returned_relative_residual"],
            "tighter_difference_worst": tighter["worst"],
            "healthy": summary["healthy"],
        })
    n1024 = fom_rows[-1]
    tables = {
        "status": "final_negative_phase_1",
        "representation": representation_rows,
        "fom_scaling": fom_rows,
        "n1024_online_budgets_s": {
            "ten_x_point": n1024["median_s"] / 10.0,
            "eight_x_lower_bound": n1024["median_s"] / 8.0,
        },
        "inherited_pure_nmrom": d0["inherited"],
        "scientific_cells": {
            "D0": 1, "excluded_partial_FOM": 1, "final_FOM": 1,
            "training": 0, "D1": 0, "weak_EQ": 0, "scaling": 0, "confirmation": 0,
        },
    }
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, "w") as handle:
        json.dump(tables, handle, indent=1, allow_nan=False)

    best = min(representation_rows, key=lambda row: row["oracle_mean"])
    rep_md = table(
        ["status", "concept", "k", "oracle mean", "oracle worst", "predictor mean", "predictor worst", "oracle wall fraction", "selection cond. median / worst", "decision"],
        [[
            "final", row["concept"], row["k"], sci(row["oracle_mean"]), sci(row["oracle_worst"]),
            sci(row["predictor_mean"]), sci(row["predictor_worst"]),
            f"{row['oracle_wall_fraction']:.6f}",
            f"{row['selection_condition_median']:.2f} / {row['selection_condition_worst']:.2f}",
            "kill" if not row["representation_pass"] else "promote",
        ] for row in representation_rows],
    )
    fom_md = table(
        ["status", "N", "outer / inner", "mean / worst error", "median ms", "clustered median 95% CI ms", "Newton / linear work", "first use s", "tight-vs-tighter worst"],
        [[
            "final", row["N"], row["selected"].replace("outer=", "").replace(":inner=", " / "),
            f"{sci(row['mean_error'])} / {sci(row['worst_error'])}", ms(row["median_s"]),
            f"[{ms(row['clustered_ci_s'][0])}, {ms(row['clustered_ci_s'][1])}]",
            f"{row['mean_newton']:.2f} / {row['mean_linear']:.2f}",
            f"{row['first_call_s']:.3f}", sci(row["tighter_difference_worst"]),
        ] for row in fom_rows],
    )
    inherited = d0["inherited"]
    gate_md = table(
        ["status", "gate", "result"],
        [
            ["final fail", "transported representation mean <= 2e-4", sci(best["oracle_mean"])],
            ["final fail", "transported representation worst <= 7e-4", sci(best["oracle_worst"])],
            ["final pass", "reference numerical error <= 1e-4", sci(max(row["tighter_difference_worst"] for row in fom_rows))],
            ["not reached", "decoder / full weak / EQ / all-seed gates", "blocked by representation hard stop"],
            ["not reached", "N=256/512 pure-ROM scaling", "no eligible pure decoder"],
            ["not reached", "N=1024 10x and clustered-LB gates", "no eligible pure decoder"],
            ["not opened", "untouched confirmation", "fixed seed/draw remained untouched"],
        ],
    )

    report = f"""# Burgers 1e-3 / 10x pure NM-ROM search — Phase 1

This report covers the preregistered transported Hermite-Gaussian Phase-1 search and calibrated Burgers FOM. All accepted numbers are **final**; the outcome is a rigorous negative at the preregistered representation hard stop. The untouched model-validation and confirmation draws were never opened.

## Outcome

**[final negative]** Neither transported analytic decoder can represent the locked 64-case selection trajectories closely enough to justify training. The best oracle is {best['concept']} at mean {sci(best['oracle_mean'])} and worst {sci(best['oracle_worst'])}, respectively {best['oracle_mean']/2e-4:.1f}x and {best['oracle_worst']/7e-4:.1f}x above the fixed representation gates. Its authoritative nearest-wall squared-error fraction is {best['oracle_wall_fraction']:.6f}, below the 0.5 license for the conditional wall chart. Therefore HG4 training, HG5 training, HG5S/D1, weak/EQ tuning, ROM scaling, and confirmation were all stopped rather than run on a failed representation.

The inherited seed-0 H160x4/g2 artifact copied into D0 has full-weak mean {sci(inherited['full_weak_trajectory_mean'])} and worst {sci(inherited['full_weak_trajectory_worst'])}; this specific row is still far outside the new 1e-3 target and has no eligible paired 10x point. No claim is made that it supersedes the broader earlier three-seed architecture summary. The new {best['concept']} number is an oracle, not a deployable Pareto point.

## Locked data and integrity

**[final]** D0 used seed {d0['config']['data_seed']}, draw {d0['config']['draw_count']}, training indices {d0['config']['train_indices'][0]}:{d0['config']['train_indices'][1]+1}, and selection indices {d0['config']['selection_indices'][0]}:{d0['config']['selection_indices'][1]+1} at N={d0['config']['N']}. Training and selection reference residual maxima are {sci(d0['reference_health']['train']['independent_max_relative_residual'])} and {sci(d0['reference_health']['selection']['independent_max_relative_residual'])}. Fixed-grid IC recovery has mean/max relative error {sci(d0['deployable_initial_parameter_recovery']['mean_relative'])} / {sci(d0['deployable_initial_parameter_recovery']['max_relative'])}.

The rerunnable audit passes manifests, source hashes against staged commits, GPU/f64/highest provenance, timing-array medians, reference health, and promotion logic. D0 was job {d0['provenance']['slurm_job_id']} on {d0['provenance']['gpu_kind']}; FOM calibration was job {fom['provenance']['slurm_job_id']} on {fom['provenance']['gpu_kind']}.

## Representation and prediction diagnostic

{rep_md}

![Representation floor]({figures['representation']})

The wall fractions above are recomputed from `representation_oracle.trajectory_all`; the staged predictor-error wall fields are non-authoritative. The simple degree-3 ridge predictor was diagnostic only. Because both representation oracles fail by orders of magnitude, improving this predictor cannot repair Phase 1.

## Fastest eligible like-for-like FOM

**[final]** Truth uses cubic history plus exact Helmholtz at outer/inner tolerances {sci(fom['config']['reference_generation']['outer_tolerance'], 0)} / {sci(fom['config']['reference_generation']['inner_tolerance'], 0)} and an independent tighter chain at {sci(fom['config']['reference_generation']['audit_outer_tolerance'], 0)} / {sci(fom['config']['reference_generation']['audit_inner_tolerance'], 0)}. All reference and audit records are finite with zero flags/breakdowns. The fastest accuracy-eligible row at every N is the calibrated outer/inner pair shown below; accuracy, work, residual, and timing come from the same invocation.

{fom_md}

![FOM scaling]({figures['fom_scaling']})

At N=1024 the final median FOM time is {ms(n1024['median_s'])} ms. Thus a 10x online point must be at most {ms(n1024['median_s']/10)} ms, and the 8x clustered-lower-bound threshold corresponds to {ms(n1024['median_s']/8)} ms. No failed representation was timed and promoted against those budgets.

## Gate ledger

{gate_md}

## Exclusions and cell accounting

**[excluded]** Job 2667476 produced provisional N=256/512 rows but failed before N=1024 because its driver used the legacy fixed-eight-Newton training-data rollout as truth. Its checksummed artifact remains in `runs/fom_cal_r2`; none of its numbers support final claims. Jobs 2667361, 2667374, and 2667531 had zero elapsed scientific work/output and are infrastructure/code-preflight records. Local smokes are execution-only and excluded.

**[final accounting]** Three scientific cells were consumed: D0, the excluded partial FOM cell, and corrected FOM calibration. No training, D1, weak/EQ, scaling, or confirmation scientific cell was run. The assigned cluster namespace was empty and no assigned job remained after checksummed pulls; unrelated account jobs were left untouched.

## Artifacts and rerun

- Preregistration: `experiments/burgers-1e3-10x/PRE-REGISTRATION.md`
- D0: `experiments/burgers-1e3-10x/runs/d0_r2/`
- FOM: `experiments/burgers-1e3-10x/runs/fom_cal_r4/`
- Machine tables: `reports/generated/burgers_1e3_10x_tables.json`
- Audit: `reports/generated/burgers_1e3_10x_audit.json`
- Regenerate: `/home/tahmid/Dev/.venv/bin/python reports/audit_2026_08_20_burgers_1e3_10x.py && /home/tahmid/Dev/.venv/bin/python reports/build_2026_08_20_burgers_1e3_10x.py`
"""
    with open(REPORT_PATH, "w") as handle:
        handle.write(report)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
