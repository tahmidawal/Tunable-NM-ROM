"""Generate the Burgers 1e-3/10x Phase-8 final report from artifacts."""
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
P4_PATH = os.path.join(RUNS, "p4_d_r3", "out", "phase4_d.json")
P4_TRAIN_PATH = os.path.join(RUNS, "p4_h1_s11_r1", "out", "train.json")
P5_PATH = os.path.join(RUNS, "p5_d_r1", "out", "phase5_d.json")
P6_PATH = os.path.join(RUNS, "p6_d_r1", "out", "phase6_d.json")
P7_PATH = os.path.join(RUNS, "p7_g1_s11_r3", "out", "phase7_train.json")
P7_AUDIT_PATH = os.path.join(RUNS, "p7_g1_s11_r3", "out", "AUDIT.json")
P8_PATH = os.path.join(RUNS, "p8_d_r2", "out", "phase8_d.json")
P8_AUDIT_PATH = os.path.join(RUNS, "p8_d_r2", "out", "AUDIT.json")
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
    d0, decision, fom, s0, p3, p4, p4_train, p5, p6, p7, p7_audit, p8, p8_audit, audit = map(load, (
        D0_PATH, DECISION_PATH, FOM_PATH, S0_PATH, P3_PATH, P4_PATH,
        P4_TRAIN_PATH, P5_PATH, P6_PATH, P7_PATH, P7_AUDIT_PATH,
        P8_PATH, P8_AUDIT_PATH, AUDIT_PATH,
    ))
    assert audit["status"] == "pass"
    assert decision["hard_stop_now"] and s0["decision"]["phase2_hard_stop"]
    assert p3["decision"] == {
        "selected_solver": None, "selected_kernel": "K3",
        "run_P3_F": False, "phase3_hard_stop": True,
    }
    assert audit["phase4_d"]["training_seed11_k24_licensed"]
    assert p4["decision"] == {
        "selected_spatial_arm": "H1",
        "training_seed11_k24_licensed": True,
        "correction_classification": "conditional-zero-or-occasional-attempt",
        "phase4_hard_stop": False,
        "scientific_promotion_allowed": True,
    }
    assert audit["phase4_h1_seed11"]["hard_stop"]
    assert audit["phase5_d"]["hard_stop"]
    assert p5["decision"] == {
        "target_integrity_pass": True,
        "arm_training_licenses": {"G1": False, "G2": False},
        "next_seed11_arm": None,
        "phase5_hard_stop": True,
        "scientific_promotion_allowed": True,
    }
    assert audit["phase6_d"]["repair_licensed"]
    assert p6["decision"] == {
        "repair_licensed": True,
        "phase6_hard_stop": False,
        "training_authorized": False,
        "next_action": "separate training proposal/audit",
        "scientific_promotion_allowed": True,
    }
    assert audit["phase7_g1_seed11"]["hard_stop"]
    assert p7_audit["status"] == "pass" and p7_audit["negative_aware"]
    assert p7["gates"] == {
        "representation_oracle_all_N_and_pooled": False,
        "direct_all_N_and_pooled": False,
        "route_identity_all_selection": True,
        "promote_seed": False,
    }
    assert p7["decision"] == {
        "g1_seed11_pass": False,
        "seeds29_47_proposal_licensed": False,
        "phase7_hard_stop": True,
        "training_authorized": False,
        "next_action": "hard stop",
        "scientific_promotion_allowed": True,
    }
    assert audit["phase8_d"]["hard_stop"]
    assert p8_audit["status"] == "pass" and p8_audit["negative_aware"]
    assert p8["gates"] == {"identity": True, "health": False}
    assert p8["decision"] == {
        "p8_d_valid": False,
        "t1_implementation_authorized": False,
        "t1_submission_authorized": False,
        "t2_authorized": False,
        "next_action": "root audit of P8-D",
        "scientific_promotion_allowed": False,
    }
    assert p4_train["gates"] == {
        "representation_oracle_all_N_and_pooled": False,
        "direct_all_N_and_pooled": False,
        "promote_seed": False,
        "local_loss_revision_near_miss_condition": False,
        "loss_revision_licensed": False,
        "loss_revision_license_reason": "joint decision requires completed seed-11/29/47 artifacts; a single-seed cell cannot license revision",
        "k32_retraining_near_miss_condition": False,
        "k32_retraining_licensed": False,
        "phase4_next_decision": "hard stop: learned manifold misses the 2x bracket",
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

    p4_oracle_rows = []
    for arm in ("H1", "H2"):
        oracle = p4["free_oracle"][arm]
        for n in (64, 128, 256):
            row = oracle["meshes"][str(n)]
            p4_oracle_rows.append({
                "arm": arm, "N": n, "mean": row["trajectory_mean"],
                "worst": row["trajectory_worst"], "snapshot_worst": row["snapshot_worst"],
                "healthy": row["healthy_count"], "fits": row["fit_count"],
                "normal_worst": row["normal_worst"],
                "pass": row["zero_unhealthy"] and row["trajectory_mean"] <= 2e-4
                and row["trajectory_worst"] <= 7e-4,
            })
        p4_oracle_rows.append({
            "arm": arm, "N": "pooled", "mean": oracle["summary"]["trajectory_mean"],
            "worst": oracle["summary"]["trajectory_worst"], "snapshot_worst": None,
            "healthy": sum(row["healthy_count"] for row in oracle["meshes"].values()),
            "fits": sum(row["fit_count"] for row in oracle["meshes"].values()),
            "normal_worst": max(row["normal_worst"] for row in oracle["meshes"].values()),
            "pass": oracle["summary"]["pass"],
        })
    p4_cost_rows = []
    for arm in ("H1", "H2"):
        for route, key in (("mandatory", "mandatory"), ("maximum-one", "maximum_one")):
            summary = p4["cost_panel"]["summaries"][f"{arm}_{key}"]
            gate = p4["cost_panel"]["gates"][arm][key]
            p4_cost_rows.append({
                "arm": arm, "route": route, "median_s": summary["median_elapsed_s"],
                "speedup": gate["paired_median_speedup"], "clustered_ci": gate["clustered_speedup_ci"],
                "outliers": summary["outliers_gt_1p5_within_trajectory_total"], "pass": gate["pass"],
            })

    p4_training_rows = []
    oracle_metrics = p4_train["selection_oracle"]["metrics"]
    direct_metrics = p4_train["selection_direct"]["metrics"]
    for n in (64, 128, 256):
        oracle_row, direct_row = oracle_metrics["meshes"][str(n)], direct_metrics["meshes"][str(n)]
        p4_training_rows.append({
            "N": n,
            "oracle_mean": oracle_row["trajectory_error_mean"],
            "oracle_worst": oracle_row["trajectory_error_worst"],
            "direct_mean": direct_row["trajectory_error_mean"],
            "direct_worst": direct_row["trajectory_error_worst"],
            "oracle_pass": oracle_row["trajectory_error_mean"] <= 2e-4
            and oracle_row["trajectory_error_worst"] <= 7e-4,
            "direct_pass": direct_row["trajectory_error_mean"] <= 3e-4
            and direct_row["trajectory_error_worst"] <= 1e-3,
            "finite": oracle_row["all_finite"] and direct_row["all_finite"],
            "exact_boundary": oracle_row["exact_binary_boundary"]
            and direct_row["exact_binary_boundary"],
        })
    p4_training_rows.append({
        "N": "pooled",
        "oracle_mean": oracle_metrics["pooled"]["trajectory_error_mean"],
        "oracle_worst": oracle_metrics["pooled"]["trajectory_error_worst"],
        "direct_mean": direct_metrics["pooled"]["trajectory_error_mean"],
        "direct_worst": direct_metrics["pooled"]["trajectory_error_worst"],
        "oracle_pass": p4_train["selection_oracle"]["gate_pass"],
        "direct_pass": p4_train["selection_direct"]["gate_pass"],
        "finite": oracle_metrics["pooled"]["all_finite"] and direct_metrics["pooled"]["all_finite"],
        "exact_boundary": oracle_metrics["pooled"]["exact_binary_boundary"]
        and direct_metrics["pooled"]["exact_binary_boundary"],
    })

    p5_rows = []
    for arm in ("G1", "G2"):
        geometry = p5["cost_panel"]["geometry"][arm]
        gate = p5["cost_panel"]["gates"][arm]
        identity = p5["cost_panel"]["identity"][arm]
        p5_rows.append({
            "arm": arm,
            "rank": geometry["rank_at_tolerance"],
            "curvature_median": geometry["curvature_median"],
            "mandatory_median_s": p5["cost_panel"]["summaries"][f"{arm}_mandatory"]["median_elapsed_s"],
            "mandatory_speedup": gate["mandatory"]["paired_median_speedup"],
            "mandatory_ci": gate["mandatory"]["clustered_speedup_ci"],
            "maximum_one_median_s": p5["cost_panel"]["summaries"][f"{arm}_maximum_one"]["median_elapsed_s"],
            "maximum_one_speedup": gate["maximum_one"]["paired_median_speedup"],
            "maximum_one_ci": gate["maximum_one"]["clustered_speedup_ci"],
            "full_field_identity_worst": max(row["relative_l2"]["full_fields"] for row in identity),
            "weak_identity_worst": max(row["relative_l2"]["weak_residual"] for row in identity),
            "identity_pass": gate["identity_pass"],
            "training_license": gate["training_cost_license"],
        })
    p5_targets = p5["train_targets"]
    p5_target_normal_worst = max(row["normal_worst"] for row in p5_targets["chunks"])
    p6_rows = []
    for route in ("R0_polynomial_weak", "R1_cox_weak_k3_full"):
        summary = p6["cost_panel"]["summaries"][route]
        identities = p6["cost_panel"]["identity"][route]
        p6_rows.append({
            "route": route,
            "median_s": summary["median_elapsed_s"],
            "outliers": summary["outliers_gt_1p5_within_trajectory_total"],
            "speedup": (
                p6["cost_panel"]["summaries"]["fom"]["median_elapsed_s"]
                / summary["median_elapsed_s"]
            ),
            "clustered_ci": (
                p6["cost_panel"]["gate"]["clustered_speedup_ci"]
                if route == "R1_cox_weak_k3_full" else None
            ),
            "identity_worst": max(row["max_relative_l2"] for row in identities),
            "identity_pass": all(row["pass"] for row in identities),
            "decision": "repair licensed" if route == "R1_cox_weak_k3_full" else "control fails",
        })
    p6_gate = p6["cost_panel"]["gate"]
    p7_rows = []
    p7_oracle = p7["selection_oracle"]["metrics"]
    p7_direct = p7["selection_direct"]["metrics"]
    for n in (64, 128, 256, "pooled"):
        key = str(n)
        oracle_row = p7_oracle["pooled"] if n == "pooled" else p7_oracle["meshes"][key]
        direct_row = p7_direct["pooled"] if n == "pooled" else p7_direct["meshes"][key]
        p7_rows.append({
            "N": n,
            "oracle_mean": oracle_row["trajectory_error_mean"],
            "oracle_worst": oracle_row["trajectory_error_worst"],
            "direct_mean": direct_row["trajectory_error_mean"],
            "direct_worst": direct_row["trajectory_error_worst"],
            "direct_over_oracle": p7["selection_direct"]
            ["degradation_direct_over_oracle"][key],
            "oracle_pass": oracle_row["trajectory_error_mean"] <= 2e-4
            and oracle_row["trajectory_error_worst"] <= 7e-4,
            "direct_pass": direct_row["trajectory_error_mean"] <= 3e-4
            and direct_row["trajectory_error_worst"] <= 1e-3,
            "finite": oracle_row["all_finite"] and direct_row["all_finite"],
            "exact_boundary": oracle_row["exact_binary_boundary"]
            and direct_row["exact_binary_boundary"],
        })

    p8_labels = (
        ("train_free_h1", "train free H1"),
        ("train_encoder_handoff", "train encoder handoff"),
        ("train_final_autolatent", "train final autolatent"),
        ("selection_free_h1", "selection free H1"),
        ("selection_free_target_encoder", "selection free-target encoder"),
        ("selection_predictor_q_exact_affine", "selection predictor q (exact affine)"),
        ("selection_deployed_predictor", "selection deployed predictor"),
        ("selection_locked_p7_oracle", "selection locked P7 oracle"),
        ("selection_trust_starts", "selection trust predictor start"),
        ("selection_trust_starts", "selection trust free-encoder start"),
        ("selection_trust_two_start", "selection trust two-start lower"),
    )
    p8_rows = []
    for index, (key, label) in enumerate(p8_labels):
        control = p8["controls"][key]
        if key == "selection_trust_starts":
            start = "predictor_q" if index == 8 else "free_target_encoder_q"
            control = control[start]
        row = control["pooled"]
        p8_rows.append({
            "control": label,
            "mean": row["trajectory_error_mean"],
            "worst": row["trajectory_error_worst"],
            "identity_worst": row["k3_cox_identity_worst"],
            "finite": row["all_finite"],
            "exact_boundary": row["exact_binary_boundary"],
            "boundary_violations": row["boundary_violation_count"],
        })
    p8_mesh_rows = []
    for n in (64, 128, 256, "pooled"):
        row = (
            p8["controls"]["selection_trust_two_start"]["pooled"]
            if n == "pooled" else
            p8["controls"]["selection_trust_two_start"]["meshes"][str(n)]
        )
        p8_mesh_rows.append({
            "N": n,
            "mean": row["trajectory_error_mean"],
            "worst": row["trajectory_error_worst"],
            "identity_worst": row["k3_cox_identity_worst"],
            "finite": row["all_finite"],
            "exact_boundary": row["exact_binary_boundary"],
            "boundary_violations": row["boundary_violation_count"],
        })

    tables = {
        "status": "phase8_d_final_health_hard_stop",
        "representation_phase1": representation_rows,
        "fom_scaling_phase1": fom_rows,
        "spline_phase2": {"oracle_meshes": spline_rows, "cost": s0_cost},
        "spline_phase3": {
            "solvers": solver_rows, "kernels": kernel_rows,
            "live_fom_accuracy": p3["kernel_diagnostic"]["fom_accuracy"],
            "live_fom_median_s": p3["kernel_diagnostic"]["summaries"]["fom"]["median_elapsed_s"],
            "decision": p3["decision"],
        },
        "hierarchical_phase4": {
            "oracle": p4_oracle_rows, "cost": p4_cost_rows,
            "live_fom_accuracy": p4["cost_panel"]["fom_accuracy"],
            "live_fom_median_s": p4["cost_panel"]["summaries"]["fom"]["median_elapsed_s"],
            "decision": p4["decision"],
            "seed11_training": {
                "rows": p4_training_rows,
                "chosen_oracle_start": p4_train["selection_oracle"]["chosen_start"],
                "chosen_oracle_loss": p4_train["selection_oracle"]["chosen_start_selection_loss"],
                "manifold_final_sampled_loss": p4_train["training"]["manifold_history"][-1]["relative_field_loss"],
                "predictor_final_relative_field_loss": p4_train["training"]["predictor_history"][-1]["relative_field_loss"],
                "gates": p4_train["gates"],
            },
        },
        "nonlinear_generator_phase5": {
            "rows": p5_rows,
            "target_snapshot_count": p5_targets["snapshot_count"],
            "target_chunk_count": p5_targets["chunk_count"],
            "target_normal_worst": p5_target_normal_worst,
            "target_integrity_pass": p5_targets["integrity_pass"],
            "head_rms": p5_targets["normalization"]["head_rms"],
            "live_fom_accuracy": p5["cost_panel"]["fom_accuracy"],
            "live_fom_median_s": p5["cost_panel"]["summaries"]["fom"]["median_elapsed_s"],
            "decision": p5["decision"],
        },
        "identity_repair_phase6": {
            "rows": p6_rows,
            "actual_route_identity_worst": max(
                row["max_relative_l2"]
                for row in p6["cost_panel"]["actual_route_consistency"]
            ),
            "live_fom_accuracy": p6["cost_panel"]["fom_accuracy"],
            "live_fom_median_s": p6["cost_panel"]["summaries"]["fom"]["median_elapsed_s"],
            "gate": p6_gate,
            "decision": p6["decision"],
        },
        "g1_training_phase7": {
            "rows": p7_rows,
            "chosen_oracle_start": p7["selection_oracle"]["chosen_start"],
            "chosen_oracle_loss": p7["selection_oracle"]
            ["chosen_start_selection_loss"],
            "training_binding": p7["data"]["training_binding"],
            "local_normalization_recompute": p7_audit[
                "target_normalization_local_recompute"
            ],
            "gates": p7["gates"],
            "decision": p7["decision"],
        },
        "full_grid_diagnostic_phase8": {
            "pooled_controls": p8_rows,
            "two_start_by_mesh": p8_mesh_rows,
            "objective": p8["config"]["objective"],
            "scientific_full_grid_route": p8["config"]["scientific_full_grid_route"],
            "trust_checkpoints": p8["trust_checkpoints"],
            "trust_health": p8["trust_health"],
            "nonpositive_prediction_sentinel_rows": audit["phase8_d"]
            ["nonpositive_prediction_sentinel_rows"],
            "gates": p8["gates"],
            "decision": p8["decision"],
        },
        "inherited_pure_nmrom": d0["inherited"],
        "scientific_cells": {
            "D0": 1, "excluded_partial_FOM": 1, "final_FOM": 1,
            "Phase2_S0": 1, "Phase3_D": 1, "Phase3_F": 0, "Phase4_D": 1,
            "Phase5_D": 1, "Phase4_training": 1, "Phase7_training": 1,
            "weak_EQ": 0, "scaling": 0, "confirmation": 0, "Phase6_D": 1,
            "Phase8_D": 1,
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
    p4_oracle_md = table(
        ["status", "arm", "N", "oracle mean / worst", "snapshot worst", "healthy fits", "normal worst", "decision"],
        [["final", row["arm"], row["N"], f"{sci(row['mean'])} / {sci(row['worst'])}",
          "—" if row["snapshot_worst"] is None else sci(row["snapshot_worst"]),
          f"{row['healthy']}/{row['fits']}", sci(row["normal_worst"]),
          "pass" if row["pass"] else "fail"] for row in p4_oracle_rows],
    )
    p4_cost_md = table(
        ["status", "arm", "route", "median ms", "speedup", "clustered 95% CI", "outliers", "decision"],
        [["final", row["arm"], row["route"], ms(row["median_s"]), f"{row['speedup']:.3f}x",
          f"[{row['clustered_ci'][0]:.3f}, {row['clustered_ci'][1]:.3f}]", row["outliers"],
          "pass" if row["pass"] else "fail"] for row in p4_cost_rows],
    )
    p4_training_md = table(
        ["status", "N", "learned oracle mean / worst", "direct mean / worst", "health", "decision"],
        [["final", row["N"], f"{sci(row['oracle_mean'])} / {sci(row['oracle_worst'])}",
          f"{sci(row['direct_mean'])} / {sci(row['direct_worst'])}",
          "finite / exact boundary" if row["finite"] and row["exact_boundary"] else "fail",
          f"{'pass' if row['oracle_pass'] else 'fail'} / {'pass' if row['direct_pass'] else 'fail'}"]
         for row in p4_training_rows],
    )
    p5_md = table(
        ["status", "arm", "rank / curvature", "mandatory ms / speedup / 95% CI",
         "maximum-one ms / speedup / 95% CI", "full / weak identity worst", "decision"],
        [["final", row["arm"], f"{row['rank']} / {sci(row['curvature_median'])}",
          f"{ms(row['mandatory_median_s'])} / {row['mandatory_speedup']:.3f}x / "
          f"[{row['mandatory_ci'][0]:.3f}, {row['mandatory_ci'][1]:.3f}]",
          f"{ms(row['maximum_one_median_s'])} / {row['maximum_one_speedup']:.3f}x / "
          f"[{row['maximum_one_ci'][0]:.3f}, {row['maximum_one_ci'][1]:.3f}]",
          f"{sci(row['full_field_identity_worst'])} / {sci(row['weak_identity_worst'])}",
          "train" if row["training_license"] else "kill"] for row in p5_rows],
    )
    p6_md = table(
        ["status", "route", "median ms / speedup", "clustered 95% CI",
         "identity worst", "outliers", "decision"],
        [["final", row["route"], f"{ms(row['median_s'], 5)} / {row['speedup']:.3f}x",
          "—" if row["clustered_ci"] is None else
          f"[{row['clustered_ci'][0]:.3f}, {row['clustered_ci'][1]:.3f}]",
          sci(row["identity_worst"]), row["outliers"], row["decision"]]
         for row in p6_rows],
    )
    p7_md = table(
        ["status", "N", "learned oracle mean / worst", "direct mean / worst",
         "direct/oracle mean", "health", "decision"],
        [["final", row["N"],
          f"{sci(row['oracle_mean'])} / {sci(row['oracle_worst'])}",
          f"{sci(row['direct_mean'])} / {sci(row['direct_worst'])}",
          f"{row['direct_over_oracle']:.6f}",
          "finite / exact boundary" if row["finite"] and row["exact_boundary"] else "fail",
          f"{'pass' if row['oracle_pass'] else 'fail'} / "
          f"{'pass' if row['direct_pass'] else 'fail'}"]
         for row in p7_rows],
    )
    p8_md = table(
        ["status", "full-grid control", "trajectory mean / worst",
         "identity worst", "health"],
        [["final", row["control"], f"{sci(row['mean'])} / {sci(row['worst'])}",
          sci(row["identity_worst"]),
          "finite / exact boundary" if row["finite"] and row["exact_boundary"]
          and row["boundary_violations"] == 0 else "fail"]
         for row in p8_rows],
    )
    p8_mesh_md = table(
        ["status", "N", "two-start trajectory mean / worst",
         "identity worst", "health"],
        [["final", row["N"], f"{sci(row['mean'])} / {sci(row['worst'])}",
          sci(row["identity_worst"]),
          "finite / exact boundary" if row["finite"] and row["exact_boundary"]
          and row["boundary_violations"] == 0 else "fail"]
         for row in p8_mesh_rows],
    )
    inherited = d0["inherited"]
    live = p3["kernel_diagnostic"]
    selected_solver = min(solver_rows, key=lambda row: row["target_N128_draw530_trajectory_relative_l2"])
    selected_kernel = next(row for row in kernel_rows if row["route"] == p3["decision"]["selected_kernel"])
    miss_ratio = selected_solver["target_N128_draw530_trajectory_relative_l2"] / 7e-4
    improvement = 1 - selected_solver["target_N128_draw530_trajectory_relative_l2"] / selected_solver["target_N128_draw530_s0_trajectory_relative_l2"]
    h1 = p4["free_oracle"]["H1"]
    h1_cost = p4["cost_panel"]["gates"]["H1"]
    trained_oracle = p4_train["selection_oracle"]["metrics"]["pooled"]
    trained_direct = p4_train["selection_direct"]["metrics"]["pooled"]
    p7_trained_oracle = p7["selection_oracle"]["metrics"]["pooled"]
    p7_trained_direct = p7["selection_direct"]["metrics"]["pooled"]
    p8_train_free = p8["controls"]["train_free_h1"]["pooled"]
    p8_selection_free = p8["controls"]["selection_free_h1"]["pooled"]
    p8_train_autolatent = p8["controls"]["train_final_autolatent"]["pooled"]
    p8_trust_predictor = p8["controls"]["selection_trust_starts"]["predictor_q"]["pooled"]
    p8_trust_encoder = p8["controls"]["selection_trust_starts"]["free_target_encoder_q"]["pooled"]
    p8_trust_two = p8["controls"]["selection_trust_two_start"]["pooled"]
    p8_floor_ratio = p8_trust_two["trajectory_error_mean"] / p8_selection_free["trajectory_error_mean"]
    p5_g1 = p5_rows[0]
    gate_md = table(["status", "gate", "result"], [
        ["final pass", "reference numerical error <=1e-4", sci(max(row["tighter_difference_worst"] for row in fom_rows))],
        ["final fail on exposed selection", "Phase-7 G1 seed-11 oracle mean<=2e-4, worst<=7e-4", f"{sci(p7_trained_oracle['trajectory_error_mean'])} / {sci(p7_trained_oracle['trajectory_error_worst'])}"],
        ["final fail on exposed selection", "Phase-7 G1 seed-11 direct mean<=3e-4, worst<=1e-3", f"{sci(p7_trained_direct['trajectory_error_mean'])} / {sci(p7_trained_direct['trajectory_error_worst'])}"],
        ["final fail on exposed selection", "Phase-4 dense seed-11 oracle mean<=2e-4, worst<=7e-4", f"{sci(trained_oracle['trajectory_error_mean'])} / {sci(trained_oracle['trajectory_error_worst'])}"],
        ["final fail on exposed selection", "Phase-4 dense seed-11 direct mean<=3e-4, worst<=1e-3", f"{sci(trained_direct['trajectory_error_mean'])} / {sci(trained_direct['trajectory_error_worst'])}"],
        ["not opened", "selected decoder reconstruction mean<=3e-4, worst<=1e-3 on untouched validation", "no eligible seed-11 model; validation untouched"],
        ["not reached", "full weak<=7e-4 and EQ<=1e-3, degradation<=1.05", "training hard stop prevents weak/EQ"],
        ["final pass (structural only)", "N1024 H1 mandatory >=10x, clustered LB>=8x", f"{h1_cost['mandatory']['paired_median_speedup']:.3f}x, LB {h1_cost['mandatory']['clustered_speedup_ci'][0]:.3f}"],
        ["final fail (worst-case route)", "N1024 H1 maximum-one >=10x, clustered LB>=8x", f"{h1_cost['maximum_one']['paired_median_speedup']:.3f}x, LB {h1_cost['maximum_one']['clustered_speedup_ci'][0]:.3f}"],
        ["final pass", "H1 free-oracle mean<=2e-4, worst<=7e-4", f"{sci(h1['summary']['trajectory_mean'])} / {sci(h1['summary']['trajectory_worst'])}"],
        ["final pass", "Phase-8 full-grid selection free-H1 mean<=2e-4, worst<=7e-4", f"{sci(p8_selection_free['trajectory_error_mean'])} / {sci(p8_selection_free['trajectory_error_worst'])}"],
        ["final fail", "Phase-8 exact two-start trust recovery mean<=2e-4, worst<=7e-4", f"{sci(p8_trust_two['trajectory_error_mean'])} / {sci(p8_trust_two['trajectory_error_worst'])}"],
        ["final fail", "Phase-8 preregistered trust health", f"all_attempt_values_finite={str(p8['trust_health']['all_attempt_values_finite']).lower()}; identity={str(p8['gates']['identity']).lower()}"],
        ["not reached", "N256/N512 learned scaling", "no eligible trained pure model"],
        ["not opened", "untouched confirmation", "fixed data remained untouched"],
        ["final repaired", "G1 actual Cox-weak/K3-full identity <=2e-14",
         sci(max(row["max_relative_l2"] for row in p6["cost_panel"]["identity"]["R1_cox_weak_k3_full"]))],
        ["final pass (structural only)", "Phase-6 actual route >=10x, clustered LB>=8x",
         f"{p6_gate['paired_median_speedup']:.3f}x, LB {p6_gate['clustered_speedup_ci'][0]:.3f}"],
    ])

    report = f"""# Burgers 1e-3 / 10x pure NM-ROM search — Phase 8 diagnostic hard stop

This report covers the finite transported-Hermite search, calibrated Burgers FOM, adaptive transported-spline screens, exact solver/kernel repair, hierarchical-spline diagnostic, H1/k24 seed-11 training, the nonlinear-generator and Cox-weak/K3-full route diagnostics, final Phase-7 G1 seed-11 training, and the Phase-8 exact full-grid inference decomposition. All accepted numbers are **final** through Phase 8. Phase 8 preserves its preregistered health hard stop and licenses no training; model-validation and untouched-confirmation draws remain unopened.

## Outcome

**[final negative]** Phase 1 stopped because the best transported Hermite oracle ({best['concept']}) has mean/worst {sci(best['oracle_mean'])} / {sci(best['oracle_worst'])}. Phase 2's much richer R=48/k=24 spline reaches pooled mean/worst {sci(s0['free_oracle']['C']['summary']['trajectory_error_mean'])} / {sci(s0['free_oracle']['C']['summary']['trajectory_error_worst'])}, but its original coefficient fits are unhealthy and its N=128 worst exceeds the {sci(7e-4)} gate. Its same-job mandatory cost also reaches only {s0_cost['speedup']:.3f}x with clustered interval [{s0_cost['clustered_ci'][0]:.3f}, {s0_cost['clustered_ci'][1]:.3f}].

Phase 3 successfully repairs both implementation defects without changing the spline space. Both algebraic solvers become fully healthy, and exact kernel {p3['decision']['selected_kernel']} reaches {selected_kernel['speedup']:.3f}x with clustered interval [{selected_kernel['clustered_ci'][0]:.3f}, {selected_kernel['clustered_ci'][1]:.3f}]. Yet the best repaired N=128 draw-530 trajectory remains {sci(selected_solver['target_N128_draw530_trajectory_relative_l2'])}: a {100*improvement:.1f}% improvement over {sci(selected_solver['target_N128_draw530_s0_trajectory_relative_l2'])}, but still {miss_ratio:.4f}x the unchanged {sci(7e-4)} representation gate. Therefore `selected_solver=null`, P3-F is not licensed, and no training/weak-EQ/scaling/confirmation result exists.

**[final Phase-4 diagnostic]** The genuinely hierarchical H1 chart passes every exposed free-oracle gate with pooled mean/worst {sci(h1['summary']['trajectory_mean'])} / {sci(h1['summary']['trajectory_worst'])} and is the preregistered smallest passing arm. Its same-H200 mandatory path reaches {h1_cost['mandatory']['paired_median_speedup']:.3f}x with clustered interval [{h1_cost['mandatory']['clustered_speedup_ci'][0]:.3f}, {h1_cost['mandatory']['clustered_speedup_ci'][1]:.3f}], which licensed the now-completed seed-11 k24 training cell. The worst-case maximum-one route reaches only {h1_cost['maximum_one']['paired_median_speedup']:.3f}x with interval [{h1_cost['maximum_one']['clustered_speedup_ci'][0]:.3f}, {h1_cost['maximum_one']['clustered_speedup_ci'][1]:.3f}], so H1 was only conditionally eligible for a zero/occasional-attempt policy; the training hard stop prevents any rollout claim.

**[final Phase-4 training hard stop]** The complete H1/k24 seed-11 hyperdecoder/autolatent fit fails its exposed learned-manifold oracle by orders of magnitude: pooled mean/worst {sci(trained_oracle['trajectory_error_mean'])} / {sci(trained_oracle['trajectory_error_worst'])}. Its direct predictor is better but still fails at {sci(trained_direct['trajectory_error_mean'])} / {sci(trained_direct['trajectory_error_worst'])}. Both are finite and preserve the exact binary boundary. The learned oracle also misses the locked k32 near-miss bracket, so seeds 29/47, k32, model validation, weak/EQ, scaling, and confirmation are not licensed.

**[final Phase-5 diagnostic hard stop; implementation floor repaired in Phase 6]** The fixed nonlinear generator bracket clears target integrity, non-collapse, memory, canonical-work, and mandatory-speed screens, but neither arm clears the unchanged K3/Cox identity tolerance {sci(p5['config']['identity_tolerance'])}. G1 is the faster arm at {p5_g1['mandatory_speedup']:.3f}x with clustered interval [{p5_g1['mandatory_ci'][0]:.3f}, {p5_g1['mandatory_ci'][1]:.3f}], yet its worst weak-residual identity is {sci(p5_g1['weak_identity_worst'])}. The full-field discrepancy is only {sci(p5_g1['full_field_identity_worst'])}. Phase 5 therefore correctly stopped before training; Phase 6 prospectively tested the single arithmetic repair below.

**[final Phase-6 positive diagnostic]** The actual G1 route uses Cox--de Boor arithmetic only for the 50 weak evaluations and retains K3/Pallas for the charged 51-field full decode. Its worst Cox-control identity is {sci(max(row['max_relative_l2'] for row in p6['cost_panel']['identity']['R1_cox_weak_k3_full']))}, and the independently executed timed route agrees with the identity route to {sci(max(row['max_relative_l2'] for row in p6['cost_panel']['actual_route_consistency']))}. Against the same-job eligible FOM at {ms(p6['cost_panel']['summaries']['fom']['median_elapsed_s'])} ms, the repaired route takes {ms(p6['cost_panel']['summaries']['R1_cox_weak_k3_full']['median_elapsed_s'], 5)} ms for {p6_gate['paired_median_speedup']:.3f}x with clustered interval [{p6_gate['clustered_speedup_ci'][0]:.3f}, {p6_gate['clustered_speedup_ci'][1]:.3f}]. Exact-50/51 work, zero failures, memory, boundary, basis/support, and live-FOM gates all pass. This structural result licensed the separately preregistered Phase-7 training cell below.

**[final Phase-7 hard stop]** The complete G1 seed-11 learned oracle fails every N and pooled accuracy gate; pooled mean/worst are {sci(p7_trained_oracle['trajectory_error_mean'])} / {sci(p7_trained_oracle['trajectory_error_worst'])}. The direct predictor is substantially better than the learned oracle but also fails every accuracy gate at pooled {sci(p7_trained_direct['trajectory_error_mean'])} / {sci(p7_trained_direct['trajectory_error_worst'])}. Both routes are finite, preserve the exact boundary, and pass Cox/K3 identity. Therefore `g1_seed11_pass=false`, seeds 29/47 are not licensed, and no G2, weak/EQ, model validation, scaling, or confirmation cell is opened.

**[final Phase-8 diagnostic hard stop]** Exact full-grid Cox verification separates the free-space and learned-generator floors. The train/selection free-H1 controls pass at pooled mean/worst {sci(p8_train_free['trajectory_error_mean'])} / {sci(p8_train_free['trajectory_error_worst'])} and {sci(p8_selection_free['trajectory_error_mean'])} / {sci(p8_selection_free['trajectory_error_worst'])}. The locked learned generator's train final autolatent is {sci(p8_train_autolatent['trajectory_error_mean'])} / {sci(p8_train_autolatent['trajectory_error_worst'])}. Forty matrix-free trust attempts reduce the selection predictor start to {sci(p8_trust_predictor['trajectory_error_mean'])} / {sci(p8_trust_predictor['trajectory_error_worst'])}; the independently initialized diagnostic start reaches {sci(p8_trust_encoder['trajectory_error_mean'])} / {sci(p8_trust_encoder['trajectory_error_worst'])}, and their per-trajectory lower envelope remains {sci(p8_trust_two['trajectory_error_mean'])} / {sci(p8_trust_two['trajectory_error_worst'])}. That mean is {p8_floor_ratio:.1f}x the free-H1 mean, so the dominant floor is the learned generator/training rather than the spline space or deployable inference start. This is a diagnostic conclusion only: {audit['phase8_d']['nonpositive_prediction_sentinel_rows']} attempted rows used the preregistered nonpositive-prediction sentinel, making `health=false`; the independent negative-aware audit accepts the immutable trace but `p8_d_valid=false`, so T1/T2 remain unauthorized.

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

## Phase 4 hierarchical spline diagnostic

{p4_oracle_md}

{p4_cost_md}

**[final positive selection]** Both H1 and H2 pass the exposed oracle and mandatory structural-cost gates; the preregistered smallest passing arm H1 is selected. Its full exposed oracle contains {sum(row['fit_count'] for row in h1['meshes'].values())} healthy fits with no S0 or fixed-P3-subset regression. The paired live FOM has mean/worst error {sci(p4['cost_panel']['fom_accuracy']['mean'])} / {sci(p4['cost_panel']['fom_accuracy']['worst'])} and median {ms(p4['cost_panel']['summaries']['fom']['median_elapsed_s'])} ms. Identity is at most {sci(max(row['max_relative_l2'] for row in p4['cost_panel']['identity']['H1']))}; exact boundaries, support, finite exactly-50 mandatory work, memory, balanced 20-repeat ordering, and zero timing outliers pass. The maximum-one row is a measured limitation, not a headline result or a reason to weaken the final speed gate.

## Phase 4 H1/k24 seed-11 training

{p4_training_md}

**[final negative]** The selected oracle start is {p4_train['selection_oracle']['chosen_start']} by pooled mean snapshot-relative-L2-squared {sci(p4_train['selection_oracle']['chosen_start_selection_loss'])}. The manifold and predictor schedules finish exactly at {p4_train['training']['manifold_history'][-1]['step']} and {p4_train['training']['predictor_history'][-1]['step']} steps. The learned-manifold failure is the active Phase-4 floor: the free H1 coefficient projection and mandatory K3 path already pass, while the fixed dense coefficient generator cannot reproduce that exposed free-oracle solution set. No deployed rollout was timed in this training cell, so the structural speed row is not a learned-model speed claim.

## Phase 5 nonlinear spatial-generator diagnostic

{p5_md}

**[final Phase-5 negative; repaired route measured separately below]** All {p5_targets['snapshot_count']} S2/H1 training coefficient targets across {p5_targets['chunk_count']} locked chunks are finite and healthy; the worst normal residual is {sci(p5_target_normal_worst)}. The train-only per-head coefficient RMS values are {sci(p5_targets['normalization']['head_rms'][0])} and {sci(p5_targets['normalization']['head_rms'][1])}. Both prospective generators exhibit numerical rank {p5_rows[0]['rank']} and nonzero curvature, so the diagnostic does not show the Phase-4 fixed-affine-output-subspace collapse. The live FOM is eligible at mean/worst {sci(p5['cost_panel']['fom_accuracy']['mean'])} / {sci(p5['cost_panel']['fom_accuracy']['worst'])} with median {ms(p5['cost_panel']['summaries']['fom']['median_elapsed_s'])} ms. Mandatory structural speed passes for both arms, but the Phase-5 weak-residual identity component fails for every live case while fields and stencils agree near machine precision. The locked Phase-5 stop remains valid for that invocation.

## Phase 6 G1 actual-route identity repair

{p6_md}

**[final positive diagnostic]** The R0 row reproduces the exposed Phase-5 arithmetic failure. R1 changes no basis, state, learned parameter, weak form, or final full-grid decoder: it uses the canonical Cox evaluator for weak queries and K3/Pallas for the full decode. Its live FOM mean/worst error is {sci(p6['cost_panel']['fom_accuracy']['mean'])} / {sci(p6['cost_panel']['fom_accuracy']['worst'])}; the tight/tighter reference difference is at most {sci(p6['cost_panel']['reference_health']['cross_chain_worst'])}. All four R1 records have exactly 50 weak evaluations and 51 coefficient-grid evaluations with zero failures. Compiled eligibility memory is {p6_gate['compiled_device_bytes'] / 1e9:.3f} GB. The result licensed the separate preregistered generator-training cell now reported below; it was never itself a trained or headline NM-ROM claim.

## Phase 7 G1 seed-11 training

{p7_md}

**[final negative hard stop]** The selected oracle start is {p7['selection_oracle']['chosen_start']} by pooled mean snapshot-relative-L2-squared {sci(p7['selection_oracle']['chosen_start_selection_loss'])}. The fixed warmup, joint, predictor, and oracle protocols completed, and the independent negative-aware audit passed. The regenerated training affine matched the immutable normalized target exactly in the cluster artifact (recorded maximum {sci(p7['data']['training_binding']['normalized_affine_max_abs'])}); an independent local scalar exp/log recomputation differs by at most {sci(p7_audit['target_normalization_local_recompute']['max_abs'])}, within its prospectively fixed absolute-only {sci(p7_audit['target_normalization_local_recompute']['absolute_tolerance'])} portability ceiling. Physical targets, features, checkpoint physical targets, and the artifact affine/state mapping remain bitwise exact. Accuracy, not integrity or route identity, causes the stop.

## Phase 8 exact full-grid inference decomposition

{p8_md}

{p8_mesh_md}

**[final diagnostic negative]** Every reported field is finite, has zero boundary violations, and agrees with the K3 identity control within {sci(2e-14)}. The exact Cox full-grid objective is not a random-point or coefficient surrogate. Predictor-start pooled mean snapshot-relative-L2-squared falls from {sci(p8['trust_checkpoints']['predictor_q'][0]['pooled_mean_snapshot_relative_l2_squared'])} at attempt 0 to {sci(p8['trust_checkpoints']['predictor_q'][-1]['pooled_mean_snapshot_relative_l2_squared'])} at attempt 40, while its worst snapshot falls from {sci(p8['trust_checkpoints']['predictor_q'][0]['pooled_worst_snapshot_relative_l2'])} to {sci(p8['trust_checkpoints']['predictor_q'][-1]['pooled_worst_snapshot_relative_l2'])}. The second, explicitly nondeployable oracle-only start converges to the same order. Work was persisted for {p8['trust_health']['attempted_total']} attempts, {p8['trust_health']['accepted_total']} acceptances, {p8['trust_health']['terminated_total']} terminations, and {p8['trust_health']['jvp_total']} JVP plus {p8['trust_health']['vjp_total']} VJP calls, with zero CG breakdown and zero unhealthy exhaustion. Because the locked health expression requires all attempted trace values finite and the sanctioned nonpositive-prediction sentinel is nonfinite, the health gate fails exactly as preregistered. This result must not be weakened into a Phase-8 training license.

## Gate ledger and stopping condition

{gate_md}

No row is a deployable NM-ROM Pareto point. H1 proves that the spline space and structural mandatory kernel can clear their exposed gates, and Phase 6 proves that G1's actual Cox-weak/K3-full route clears structural identity and speed. Phase 8 localizes the fixed Phase-7 failure to the learned generator/training floor even after exact full-grid inference globalization, but its unchanged health gate fails. Consequently Phase 8 licenses no T1/T2, no method claims the headline N=1024 result, and weak/EQ, model validation, scaling, and confirmation remain unrun.

## Exclusions and cell accounting

**[excluded]** Job 2667476 is a partial FOM attempt that failed before N=1024. Zero-output jobs 2667361, 2667374, 2667531, Phase-4 job 2668601, Phase-7 jobs 2669861/2669975, and Phase-8 job 2683739 are infrastructure/code-preflight records. The Phase-7 failures occurred before output or a training update: r1 exposed a physical-vs-normalized schema check, and r2 exposed a nonportable bitwise regeneration comparison. Phase-8 r1 failed in two seconds because the exact staged-set verifier did not allow the two Slurm-created log files; no driver, backend, data, or scientific output ran. Phase-4 job 2668613 is excluded without scientific inspection because its root manifest omitted the nested P3 manifest. The terminated overlong local trust diagnostic is excluded and was never reused. Every compliant local smoke is execution-only. The two synthetic-only model-validation drafts are uncommitted and excluded; none touched model-validation or confirmation data.

**[final through Phase-8 accounting]** Eleven scientific cells were consumed: D0, one excluded partial FOM cell, corrected FOM calibration, Phase-2 S0, Phase-3 P3-D, Phase-4 P4-D, H1/k24 seed-11 training, Phase-5 P5-D, Phase-6 P6-D, final Phase-7 G1 seed-11 training, and Phase-8 P8-D. The Phase-4 provenance failures, Phase-7 pre-update failures, and Phase-8 zero-science lifecycle failure did not alter the scientific method/cap and are excluded. Model validation, weak/EQ, scaling, and confirmation consumed zero cells. All completed remote directories were deleted after checksummed pulls, and the assigned namespace is empty.

## Artifacts and rerun

- Preregistrations: `experiments/burgers-1e3-10x/PRE-REGISTRATION.md`, `PHASE-2-PRE-REGISTRATION.md`, `PHASE-3-PRE-REGISTRATION.md`, `PHASE-4-PRE-REGISTRATION.md`, `PHASE-5-PRE-REGISTRATION.md`, `PHASE-6-PRE-REGISTRATION.md`, `PHASE-7-PRE-REGISTRATION.md`, `PHASE-8-PRE-REGISTRATION.md`
- Accepted runs: `experiments/burgers-1e3-10x/runs/d0_r2/`, `fom_cal_r4/`, `s0_spline_r1/`, `p3_d_r1/`, `p4_d_r3/`, `p4_h1_s11_r1/`, `p5_d_r1/`, `p6_d_r1/`, `p7_g1_s11_r3/`, `p8_d_r2/`
- Machine tables: `reports/generated/burgers_1e3_10x_tables.json`
- Audit: `reports/generated/burgers_1e3_10x_audit.json`
- Regenerate: `/home/tahmid/Dev/.venv/bin/python reports/audit_2026_08_20_burgers_1e3_10x.py && /home/tahmid/Dev/.venv/bin/python reports/build_2026_08_20_burgers_1e3_10x.py`
"""
    with open(REPORT_PATH, "w") as handle:
        handle.write(report)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
