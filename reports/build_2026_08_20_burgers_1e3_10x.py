"""Generate the final Burgers 1e-3/10x negative-result report from artifacts."""
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
RUNS = os.path.join(EXP, "runs")
D0_PATH = os.path.join(RUNS, "d0_r2", "out", "d0.json")
DECISION_PATH = os.path.join(RUNS, "d0_r2", "d0_decision.json")
FOM_PATH = os.path.join(RUNS, "fom_cal_r4", "out", "fom_cal.json")
S0_PATH = os.path.join(RUNS, "s0_spline_r1", "out", "s0.json")
P3_PATH = os.path.join(RUNS, "p3_d_r1", "out", "phase3_d.json")
AUDIT_PATH = os.path.join(ROOT, "reports", "generated", "burgers_1e3_10x_audit.json")
TABLE_PATH = os.path.join(ROOT, "reports", "generated", "burgers_1e3_10x_tables.json")
REPORT_PATH = os.path.join(ROOT, "reports", "2026-08-20-burgers-1e3-10x.md")
FIG_DIR = os.path.join(ROOT, "reports", "figures")


def load(path):
    with open(path) as handle:
        return json.load(handle)


def sci(value, digits=3):
    return f"{value:.{digits}e}"


def ms(value, digits=3):
    return f"{1e3 * value:.{digits}f}"


def table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def normalize_svg(path):
    with open(path) as handle:
        lines = handle.readlines()
    with open(path, "w") as handle:
        handle.writelines(line.rstrip() + "\n" for line in lines)


def make_figures(decision, fom, p3):
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

    routes = ("K0", "K1", "K2", "K3")
    gates = p3["kernel_diagnostic"]["gates"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    values = [gates[name]["paired_median_speedup"] for name in routes]
    low = [gates[name]["clustered_speedup_ci"][0] for name in routes]
    high = [gates[name]["clustered_speedup_ci"][1] for name in routes]
    x = np.arange(len(routes))
    ax.bar(x, values, color=["#BAB0AC", "#4C78A8", "#72B7B2", "#54A24B"])
    ax.errorbar(x, values, yerr=(np.asarray(values) - low, np.asarray(high) - values),
                fmt="none", ecolor="black", capsize=4)
    ax.axhline(10, color="#E45756", linestyle="--", label="10x point gate")
    ax.axhline(8, color="#F58518", linestyle=":", label="8x lower-bound gate")
    ax.set_xticks(x, routes)
    ax.set_ylabel("paired speedup")
    ax.set_title("Phase-3 exact-kernel bracket, live same-job FOM")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    kernel_base = os.path.join(FIG_DIR, "burgers_1e3_10x_phase3_kernels")
    fig.savefig(kernel_base + ".png", dpi=180)
    fig.savefig(kernel_base + ".svg", metadata={"Date": None})
    normalize_svg(kernel_base + ".svg")
    plt.close(fig)
    return {
        "representation": os.path.relpath(rep_base + ".png", os.path.dirname(REPORT_PATH)),
        "fom_scaling": os.path.relpath(fom_base + ".png", os.path.dirname(REPORT_PATH)),
        "phase3_kernels": os.path.relpath(kernel_base + ".png", os.path.dirname(REPORT_PATH)),
    }


def main():
    d0, decision, fom, s0, p3, audit = map(load, (
        D0_PATH, DECISION_PATH, FOM_PATH, S0_PATH, P3_PATH, AUDIT_PATH
    ))
    assert audit["status"] == "pass"
    assert decision["hard_stop_now"] and s0["decision"]["phase2_hard_stop"]
    assert p3["decision"] == {
        "selected_solver": None, "selected_kernel": "K3",
        "run_P3_F": False, "phase3_hard_stop": True,
    }
    figures = make_figures(decision, fom, p3)

    representation_rows = []
    for name in ("HG4", "HG5"):
        row, raw = decision["concepts"][name], d0["concepts"][name]
        representation_rows.append({
            "concept": name, "k": raw["latent_dimension"],
            "oracle_mean": row["representation_oracle_mean"],
            "oracle_worst": row["representation_oracle_worst"],
            "predictor_mean": row["predictor_decoder_mean"],
            "predictor_worst": row["predictor_decoder_worst"],
            "oracle_wall_fraction": row["nearest_wall_squared_representation_oracle_error_fraction"],
            "representation_pass": row["representation_gate_pass"],
            "wall_license_pass": row["wall_license_pass"],
        })

    fom_rows = []
    for n in (256, 512, 1024):
        mesh = fom["meshes"][str(n)]
        label = mesh["selected_fastest_eligible"]
        row = mesh["rows"][label]
        summary, setup = row["summary"], row["setup"]
        fom_rows.append({
            "N": n, "selected": label,
            "mean_error": summary["trajectory_error_mean"],
            "worst_error": summary["trajectory_error_worst"],
            "median_s": summary["median_elapsed_s"],
            "clustered_ci_s": summary["clustered_median_elapsed_ci_s"],
            "mean_newton": summary["mean_newton_total"],
            "mean_linear": summary["mean_linear_total"],
            "first_call_s": setup["first_call_compile_and_solve_s"],
            "tighter_difference_worst": mesh["independent_tighter_reference_difference"]["worst"],
        })
    n1024 = fom_rows[-1]

    spline_rows = []
    for n in (64, 128, 256):
        row = s0["free_oracle"]["C"]["meshes"][str(n)]
        spline_rows.append({
            "N": n, "mean": row["trajectory_error_mean"], "worst": row["trajectory_error_worst"],
            "healthy_fits": row["healthy_fits"], "fit_count": row["fit_count"],
            "normal_worst": row["relative_normal_worst"],
            "iteration_median": row["lsmr_iterations_median"],
            "iteration_max": row["lsmr_iterations_max"],
            "accuracy_pass": row["trajectory_error_mean"] <= 2e-4 and row["trajectory_error_worst"] <= 7e-4,
            "health_pass": row["zero_unhealthy_fits"],
        })
    s0_gate = s0["cost_panel"]["gates"]["C"]
    s0_summary = s0["cost_panel"]["summaries"]
    s0_cost = {
        "fom_median_s": s0_summary["fom"]["median_elapsed_s"],
        "direct_median_s": s0_summary["C_direct"]["median_elapsed_s"],
        "mandatory_median_s": s0_summary["C_mandatory"]["median_elapsed_s"],
        "maximum_one_median_s": s0_summary["C_maximum_one"]["median_elapsed_s"],
        "speedup": s0_gate["paired_median_speedup_mandatory_lower_bound"],
        "clustered_ci": s0_gate["clustered_speedup_ci"], "pass": s0_gate["pass"],
    }

    solver_rows = []
    for name in ("S1", "S2"):
        row = p3["solver_diagnostic"]["summaries"][name]
        solver_rows.append({"solver": name, **row})
    kernel_rows = []
    for name in ("K0", "K1", "K2", "K3"):
        summary = p3["kernel_diagnostic"]["summaries"][name]
        gate = p3["kernel_diagnostic"]["gates"][name]
        kernel_rows.append({
            "route": name, "median_s": summary["median_elapsed_s"],
            "per_case_median_s": summary["per_case_median_elapsed_s"],
            "outliers": summary["outliers_gt_1p5_within_trajectory_total"],
            "speedup": gate["paired_median_speedup"], "clustered_ci": gate["clustered_speedup_ci"],
            "identity_pass": gate["identity_pass"], "canonical_work_pass": gate["canonical_work_pass"],
            "memory_pass": gate["memory_pass"], "pass": gate["pass"],
        })

    tables = {
        "status": "final_negative_phases_1_to_3",
        "representation_phase1": representation_rows,
        "fom_scaling_phase1": fom_rows,
        "spline_phase2": {"oracle_meshes": spline_rows, "cost": s0_cost},
        "spline_phase3": {
            "solvers": solver_rows, "kernels": kernel_rows,
            "live_fom_accuracy": p3["kernel_diagnostic"]["fom_accuracy"],
            "live_fom_median_s": p3["kernel_diagnostic"]["summaries"]["fom"]["median_elapsed_s"],
            "decision": p3["decision"],
        },
        "inherited_pure_nmrom": d0["inherited"],
        "scientific_cells": {
            "D0": 1, "excluded_partial_FOM": 1, "final_FOM": 1,
            "Phase2_S0": 1, "Phase3_D": 1, "Phase3_F": 0,
            "training": 0, "weak_EQ": 0, "scaling": 0, "confirmation": 0,
        },
    }
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, "w") as handle:
        json.dump(tables, handle, indent=1, allow_nan=False)

    best = min(representation_rows, key=lambda row: row["oracle_mean"])
    rep_md = table(
        ["status", "concept", "k", "oracle mean / worst", "predictor mean / worst", "wall fraction", "decision"],
        [["final", row["concept"], row["k"], f"{sci(row['oracle_mean'])} / {sci(row['oracle_worst'])}",
          f"{sci(row['predictor_mean'])} / {sci(row['predictor_worst'])}", f"{row['oracle_wall_fraction']:.6f}",
          "kill" if not row["representation_pass"] else "promote"] for row in representation_rows],
    )
    fom_md = table(
        ["status", "N", "outer / inner", "mean / worst error", "median ms", "clustered 95% CI ms", "Newton / linear work", "tight-vs-tighter worst"],
        [["final", row["N"], row["selected"].replace("outer=", "").replace(":inner=", " / "),
          f"{sci(row['mean_error'])} / {sci(row['worst_error'])}", ms(row["median_s"]),
          f"[{ms(row['clustered_ci_s'][0])}, {ms(row['clustered_ci_s'][1])}]",
          f"{row['mean_newton']:.2f} / {row['mean_linear']:.2f}", sci(row["tighter_difference_worst"])]
         for row in fom_rows],
    )
    spline_md = table(
        ["status", "N", "oracle mean / worst", "healthy fits", "normal worst", "iterations median / max", "accuracy / health"],
        [["final", row["N"], f"{sci(row['mean'])} / {sci(row['worst'])}",
          f"{row['healthy_fits']}/{row['fit_count']}", sci(row["normal_worst"]),
          f"{row['iteration_median']:.0f} / {row['iteration_max']}",
          f"{'pass' if row['accuracy_pass'] else 'fail'} / {'pass' if row['health_pass'] else 'fail'}"]
         for row in spline_rows],
    )
    solver_md = table(
        ["status", "solver", "healthy fits", "normal worst", "draw-530 error", "S0 error", "no regression", "decision"],
        [["final", row["solver"], f"{row['fit_count']}/{row['fit_count']}", sci(row["worst_relative_normal_residual"]),
          sci(row["target_N128_draw530_trajectory_relative_l2"]), sci(row["target_N128_draw530_s0_trajectory_relative_l2"]),
          "pass" if row["no_snapshot_regression_pass"] else "fail", "promote" if row["pass"] else "kill"]
         for row in solver_rows],
    )
    kernel_md = table(
        ["status", "route", "median ms", "speedup", "clustered 95% CI", "identity / work / memory", "decision"],
        [["final", row["route"], ms(row["median_s"], 5), f"{row['speedup']:.3f}x",
          f"[{row['clustered_ci'][0]:.3f}, {row['clustered_ci'][1]:.3f}]",
          "/".join("pass" if row[key] else "fail" for key in ("identity_pass", "canonical_work_pass", "memory_pass")),
          "pass" if row["pass"] else "fail"] for row in kernel_rows],
    )
    inherited = d0["inherited"]
    live = p3["kernel_diagnostic"]
    selected_solver = min(solver_rows, key=lambda row: row["target_N128_draw530_trajectory_relative_l2"])
    selected_kernel = next(row for row in kernel_rows if row["route"] == p3["decision"]["selected_kernel"])
    miss_ratio = selected_solver["target_N128_draw530_trajectory_relative_l2"] / 7e-4
    improvement = 1 - selected_solver["target_N128_draw530_trajectory_relative_l2"] / selected_solver["target_N128_draw530_s0_trajectory_relative_l2"]
    gate_md = table(["status", "gate", "result"], [
        ["final pass", "reference numerical error <=1e-4", sci(max(row["tighter_difference_worst"] for row in fom_rows))],
        ["final fail", "selected decoder reconstruction mean<=3e-4, worst<=1e-3 on untouched validation", "no trained decoder; validation untouched"],
        ["final fail", "full weak<=7e-4 and EQ<=1e-3, degradation<=1.05", "training/weak/EQ not licensed"],
        ["final pass (structural only)", "N1024 mandatory kernel >=10x, clustered LB>=8x", f"{selected_kernel['speedup']:.3f}x, LB {selected_kernel['clustered_ci'][0]:.3f}"],
        ["final fail", "locked spline representation witness <=7e-4", sci(selected_solver["target_N128_draw530_trajectory_relative_l2"])],
        ["not reached", "N256/N512 learned scaling", "no eligible trained pure model"],
        ["not opened", "untouched confirmation", "fixed data remained untouched"],
    ])

    report = f"""# Burgers 1e-3 / 10x pure NM-ROM search — final Phase 1–3 result

This report covers the finite transported-Hermite search, calibrated Burgers FOM, adaptive transported-spline oracle/cost screen, and exact solver/kernel repair. All accepted numbers are **final**. The result is negative: no pure nonlinear model reached the simultaneous accuracy and speed gates, and model-validation plus untouched-confirmation draws were never opened.

## Outcome

**[final negative]** Phase 1 stopped because the best transported Hermite oracle ({best['concept']}) has mean/worst {sci(best['oracle_mean'])} / {sci(best['oracle_worst'])}. Phase 2's much richer R=48/k=24 spline reaches pooled mean/worst {sci(s0['free_oracle']['C']['summary']['trajectory_error_mean'])} / {sci(s0['free_oracle']['C']['summary']['trajectory_error_worst'])}, but its original coefficient fits are unhealthy and its N=128 worst exceeds the {sci(7e-4)} gate. Its same-job mandatory cost also reaches only {s0_cost['speedup']:.3f}x with clustered interval [{s0_cost['clustered_ci'][0]:.3f}, {s0_cost['clustered_ci'][1]:.3f}].

Phase 3 successfully repairs both implementation defects without changing the spline space. Both algebraic solvers become fully healthy, and exact kernel {p3['decision']['selected_kernel']} reaches {selected_kernel['speedup']:.3f}x with clustered interval [{selected_kernel['clustered_ci'][0]:.3f}, {selected_kernel['clustered_ci'][1]:.3f}]. Yet the best repaired N=128 draw-530 trajectory remains {sci(selected_solver['target_N128_draw530_trajectory_relative_l2'])}: a {100*improvement:.1f}% improvement over {sci(selected_solver['target_N128_draw530_s0_trajectory_relative_l2'])}, but still {miss_ratio:.4f}x the unchanged {sci(7e-4)} representation gate. Therefore `selected_solver=null`, P3-F is not licensed, and no training/weak-EQ/scaling/confirmation result exists.

The inherited seed-0 H160x4/g2 artifact copied into D0 has full-weak mean/worst {sci(inherited['full_weak_trajectory_mean'])} / {sci(inherited['full_weak_trajectory_worst'])}; this specific inherited row is neither a new result nor an eligible Pareto point.

## Phase 1 representation diagnostic

{rep_md}

![Representation floor]({figures['representation']})

The authoritative wall fractions are computed from representation-oracle arrays. Both are below the 0.5 conditional wall-chart license; no failed Hermite concept was trained.

## Calibrated like-for-like FOM

{fom_md}

![FOM scaling]({figures['fom_scaling']})

**[final]** All reference and tighter-audit chains are finite with zero flags/breakdowns. The N=1024 A100 median {ms(n1024['median_s'])} ms is planning evidence only across jobs. Every scientific speed decision below uses a fresh live eligible FOM in the same H200 job.

## Phase 2 adaptive spline screen

{spline_md}

**[final negative]** Arm C's direct/mandatory/maximum-one medians are {ms(s0_cost['direct_median_s'])} / {ms(s0_cost['mandatory_median_s'])} / {ms(s0_cost['maximum_one_median_s'])} ms against its live eligible FOM at {ms(s0_cost['fom_median_s'])} ms. Mandatory speedup is {s0_cost['speedup']:.3f}x with interval [{s0_cost['clustered_ci'][0]:.3f}, {s0_cost['clustered_ci'][1]:.3f}]. No arm is promoted.

## Phase 3 exact solver repair

{solver_md}

**[final negative]** Algebraic health is repaired: all {solver_rows[0]['fit_count']} fixed diagnostic fits pass for each solver, with no snapshot regression. The common remaining {sci(selected_solver['target_N128_draw530_trajectory_relative_l2'])} witness is therefore a representation floor for this locked spline, not a coefficient-solve-health artifact. Both solvers miss the fixed {sci(7e-4)} target and are killed.

## Phase 3 exact kernel repair

{kernel_md}

![Phase-3 kernel speedups]({figures['phase3_kernels']})

**[final]** The live FOM has mean/worst error {sci(live['fom_accuracy']['mean'])} / {sci(live['fom_accuracy']['worst'])}, is healthy/eligible, and has median {ms(live['summaries']['fom']['median_elapsed_s'])} ms. K1, K2, and K3 all pass both speed requirements. For K3, basis-probe maximum absolute weight difference is {sci(live['pallas_basis_identity']['weight_max_abs'])}; all four live-case full/stencil/previous/residual/rho identities, exact boundaries, finite exactly-50 weak-evaluation work records, memory gates, and the 20-repeat cyclic balance pass. All within-trajectory timing outlier counts are zero.

## Gate ledger and stopping condition

{gate_md}

The structural K3 row is not a deployable NM-ROM Pareto point: it uses representative locked states to lower-bound mandatory work and has no eligible trained decoder/solver. Consequently no method can claim the headline N=1024 accuracy/speed result. The preregistered finite Phase-3 stopping condition is met honestly at `selected_solver=null`, `selected_kernel=K3`, `run_P3_F=false`, `phase3_hard_stop=true`.

## Exclusions and cell accounting

**[excluded]** Job 2667476 is a partial FOM attempt that failed before N=1024. Zero-output jobs 2667361, 2667374, and 2667531 are infrastructure/code-preflight records. Every local smoke is execution-only. The two synthetic-only model-validation drafts and the stopped prospective P3-F draft are uncommitted and excluded; none touched model-validation or confirmation data.

**[final accounting]** Five scientific cells were consumed: D0, one excluded partial FOM cell, corrected FOM calibration, Phase-2 S0, and Phase-3 P3-D. P3-F, training, model validation, weak/EQ, scaling, and confirmation consumed zero scientific cells. All completed remote directories were deleted after checksummed pulls, and the assigned namespace is empty.

## Artifacts and rerun

- Preregistrations: `experiments/burgers-1e3-10x/PRE-REGISTRATION.md`, `PHASE-2-PRE-REGISTRATION.md`, `PHASE-3-PRE-REGISTRATION.md`
- Accepted runs: `experiments/burgers-1e3-10x/runs/d0_r2/`, `fom_cal_r4/`, `s0_spline_r1/`, `p3_d_r1/`
- Machine tables: `reports/generated/burgers_1e3_10x_tables.json`
- Audit: `reports/generated/burgers_1e3_10x_audit.json`
- Regenerate: `/home/tahmid/Dev/.venv/bin/python reports/audit_2026_08_20_burgers_1e3_10x.py && /home/tahmid/Dev/.venv/bin/python reports/build_2026_08_20_burgers_1e3_10x.py`
"""
    with open(REPORT_PATH, "w") as handle:
        handle.write(report)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
