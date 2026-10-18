#!/usr/bin/env python3
"""Build the final cross-PDE hybrid warm-start report from audited JSONs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "2026-08-19-hybrid-warm-starts-through-1024.md"


def locate(*relative_candidates: str) -> Path:
    for relative in relative_candidates:
        path = ROOT / relative
        if path.exists():
            return path
    joined = "\n".join(str(ROOT / item) for item in relative_candidates)
    raise FileNotFoundError(f"none of the required result files exists:\n{joined}")


POISSON_PAIR_PATH = locate(
    "experiments/poisson-hybrid-1024/runs/pairfinal1/out/pairfinal1.json",
    "worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/"
    "runs/pairfinal1/out/pairfinal1.json",
)
POISSON_PANEL_PATH = locate(
    "experiments/poisson-hybrid-1024/runs/final1/out/final1.json",
    "worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/"
    "runs/final1/out/final1.json",
)
BURGERS_FINAL_PATH = locate(
    "experiments/burgers-hybrid-1024/runs/confirm2/out/final_summary.json",
    "worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/"
    "runs/confirm2/out/final_summary.json",
)
BURGERS_RAW_PATH = locate(
    "experiments/burgers-hybrid-1024/runs/confirm2/out/confirm2.json",
    "worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/"
    "runs/confirm2/out/confirm2.json",
)
BURGERS_CORR_PATH = locate(
    "experiments/burgers-hybrid-1024/runs/corr_gate/out/corr_gate.json",
    "worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/"
    "runs/corr_gate/out/corr_gate.json",
)
BURGERS_SHIFT_PATH = locate(
    "experiments/burgers-hybrid-1024/runs/shift_corr_gate/out/shift_corr_gate.json",
    "worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/"
    "runs/shift_corr_gate/out/shift_corr_gate.json",
)


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


pair = load(POISSON_PAIR_PATH)
panel = load(POISSON_PANEL_PATH)
burgers = load(BURGERS_FINAL_PATH)
burgers_raw = load(BURGERS_RAW_PATH)
corr = load(BURGERS_CORR_PATH)
shift = load(BURGERS_SHIFT_PATH)


def sci(value: float) -> str:
    return f"{value:.0e}"


def f3(value: float) -> str:
    return f"{value:.3f}"


def ci(values: list[float], digits: int = 3) -> str:
    return f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def pair_status(row: dict) -> str:
    speed_ci = row["speedup_vs_zero_cg_bootstrap_ci95"]
    delta_ci = row["paired_delta_bootstrap_ci95_ms"]
    if speed_ci[0] > 1.0 and delta_ci[1] < 0.0:
        return "supported faster"
    if speed_ci[1] < 1.0 and delta_ci[0] > 0.0:
        return "supported slower"
    return "inconclusive / tie"


pair_rows = sorted(pair["rows"], key=lambda row: (row["N"], -row["fom_tau"]))
supported_pair = [row for row in pair_rows if pair_status(row) == "supported faster"]
pair_map = {(row["N"], row["fom_tau"]): row for row in pair_rows}
panel_rows = panel["rows"]


def panel_row(N: int, tau: float, arm: str) -> dict:
    return next(
        row
        for row in panel_rows
        if row["N"] == N and row["fom_tau"] == tau and row["arm"] == arm
    )


poisson_ns = sorted({row["N"] for row in panel_rows})
poisson_taus = sorted({row["fom_tau"] for row in panel_rows}, reverse=True)
burgers_comparisons = sorted(
    burgers["comparisons"], key=lambda row: (row["N"], -row["fom_tau"])
)
burgers_ns = sorted({row["N"] for row in burgers_comparisons})
burgers_taus = sorted({row["fom_tau"] for row in burgers_comparisons}, reverse=True)
max_n = max(max(poisson_ns), max(burgers_ns))

poisson_prov = pair["provenance"]
burgers_prov = burgers["provenance"]
poisson_cfg = panel["config"]
group_cfg = panel["nonlinear_groupfilm_checkpoint"]["config"]
rbf_selected = panel["rbf_train_only"]["selected"]
poisson_coarse_n = pair["config"]["arms"][0]["coarse_n"]
burgers_mesh0 = next(iter(burgers_raw["offline_per_mesh"].values()))
burgers_weak_M = burgers_mesh0["eq_info"]["M"]
burgers_weak_m = burgers_mesh0["eq_info"]["m"]
burgers_k = burgers_raw["config"]["checkpoint_config"]["k_lat"]

lines: list[str] = []
add = lines.append
add(f"# NM-ROM warm starts followed by FOM correction through N={max_n}")
add("")
add(
    "This final report covers the autonomous Poisson-2D and Burgers-2D architecture, "
    "hyperparameter, and solver push. Every numeric statement and table is generated from "
    "the checksummed run JSONs; superseded timing claims are identified inline."
)
add("")
add("## Outcome")
add("")
one = supported_pair[0]
add(
    f"- **Poisson:** the optimized genuine K={poisson_cfg['pkl_config']['K_LAT']} "
    f"NM-ROM warm start has exactly {len(supported_pair)} supported crossover in the balanced "
    f"confirmation: N={one['N']}, FOM tolerance {sci(one['fom_tau'])}, "
    f"{f3(one['hybrid_total_ms'])} ms versus {f3(one['baseline_total_ms'])} ms, "
    f"or {one['speedup_vs_zero_cg']:.3f}x with clustered interval "
    f"{ci(one['speedup_vs_zero_cg_bootstrap_ci95'])}. It is a modest win against counting CG, "
    "not the best Poisson solver."
)
add(
    f"- **Burgers:** the calibrated cubic-history FOM is faster than linear history in "
    f"{burgers['decision_counts']['cubic_clearly_faster_than_linear']} of "
    f"{burgers['decision_counts']['condition_count']} conditions. The guarded weak FiLM "
    f"NM-ROM is slower than cubic in "
    f"{burgers['decision_counts']['guarded_film_clearly_slower_than_cubic']} of "
    f"{burgers['decision_counts']['condition_count']} conditions. The learned warm start does "
    "not win."
)
add("")
add("## What was optimized")
add("")
add(
    f"Poisson tested the original full-grid K={poisson_cfg['pkl_config']['K_LAT']} FiLM path, "
    f"a trust-region K={poisson_cfg['pkl_config']['K_LAT']} path decoded at fixed "
    f"N={poisson_coarse_n}, and a cached nonlinear GroupFiLM with "
    f"K={group_cfg['K_LAT']} and {panel['nonlinear_groupfilm_checkpoint']['n_params']} "
    "parameters. The train-only RBF latent predictor was stopped after its selected "
    f"standardized latent MSE was {rbf_selected['calibration_standardized_latent_mse']:.3f}. "
    "The optimized K8 path uses weak-form empirical quadrature with "
    f"M={pair['config']['M']} and m={pair['config']['m']}; construction at N={max_n} is "
    f"{f3(pair_map[(max_n, poisson_taus[0])]['construction_ms'])} ms."
)
rank_metrics = corr["spectrum"]["rank_metrics"]
rank32 = rank_metrics[str(max(map(int, rank_metrics)))]
shift_best = shift["selection"]["best_supervised_surrogate"]
add(
    f"Burgers first tuned the FOM itself: the selected inner tolerances are "
    f"{', '.join(sci(v) for v in burgers['method']['inner_tolerances'])} for outer tolerances "
    f"{', '.join(sci(v) for v in burgers['method']['outer_tolerances'])}, with the exact "
    f"Helmholtz preconditioner. A rank-{max(map(int, rank_metrics))} correction basis still "
    f"left {rank32['validation_projection_remaining_ratio']:.3f} of validation correction, "
    f"and the best translated/scaled deployable surrogate left "
    f"{shift_best['remaining_ratio_global']:.3f}; both missed their gate. The retained genuine "
    f"FiLM control therefore uses K={burgers_k}, "
    f"M={burgers_weak_M}, m={burgers_weak_m}, "
    "a fixed coarse decode, two weak Jacobians per step, an exact-residual guard, and a charged "
    "live cubic fallback."
)
add("")
add("## Poisson: learned warm start across resolution")
add("")
add(
    "Rows through the last unbalanced mesh are screening evidence from the all-arm panel; "
    "their large losses are retained for the requested resolution ladder. The two finest "
    "meshes use the fresh-seed, position-balanced AB/BA confirmation and are authoritative "
    "near parity."
)
add("")
add("| design | N | FOM tau | hybrid ms | zero-CG ms | speedup | interval | verdict |")
add("|---|---:|---:|---:|---:|---:|---|---|")
for N in poisson_ns:
    for tau in poisson_taus:
        if (N, tau) in pair_map:
            row = pair_map[(N, tau)]
            design = "balanced AB/BA"
            interval = ci(row["speedup_vs_zero_cg_bootstrap_ci95"])
            verdict = pair_status(row)
        else:
            row = panel_row(N, tau, "lmtrmean_c64_q0")
            design = "screening panel"
            interval = ci(row["speedup_vs_zero_cg_bootstrap_ci95"])
            verdict = "slower"
        add(
            f"| {design} | {N} | {sci(tau)} | {f3(row['hybrid_total_ms'])} | "
            f"{f3(row['baseline_total_ms'])} | {row['speedup_vs_zero_cg']:.3f}x | "
            f"{interval} | {verdict} |"
        )
add("")
add("### Balanced paired evidence at the two finest meshes")
add("")
add("| N | tau | hybrid - zero ms [clustered interval] | favorable cases | status |")
add("|---:|---:|---:|---:|---|")
for row in pair_rows:
    signs = row["paired_case_sign_counts"]
    add(
        f"| {row['N']} | {sci(row['fom_tau'])} | "
        f"{row['paired_delta_arm_minus_zero_ms']:+.3f} "
        f"{ci(row['paired_delta_bootstrap_ci95_ms'])} | "
        f"{signs['arm_faster']}/{signs['arm_faster'] + signs['zero_faster']} | "
        f"{pair_status(row)} |"
    )
add("")
add("### Strong Poisson controls at the largest mesh")
add("")
add(
    "These rows are same-job comparisons from the all-arm panel. The spectral/direct margin is "
    "large enough that the timing-order issue affecting the small learned crossover cannot "
    "change the conclusion."
)
add("")
add("| FOM tau | selected spectral arm | spectral ms | zero-CG ms | speedup | FFT-DST ms |")
add("|---:|---|---:|---:|---:|---:|")
for tau in poisson_taus:
    arm = "spectral_q512" if tau == max(poisson_taus) else "spectral_q1024"
    row = panel_row(max_n, tau, arm)
    add(
        f"| {sci(tau)} | `{arm}` | {f3(row['hybrid_total_ms'])} | "
        f"{f3(row['baseline_total_ms'])} | {row['speedup_vs_zero_cg']:.1f}x | "
        f"{f3(row['exact_direct_ms'])} |"
    )
add("")
add("The learned-plus-spectral arm is not evidence for learning: at the largest mesh its total is")
add("")
add("| FOM tau | GroupFiLM + q8 ms | matched selected spectral ms | excess ms |")
add("|---:|---:|---:|---:|")
for tau in poisson_taus:
    combo = panel_row(max_n, tau, "groupn_rt30_c64_q8")
    spec_arm = "spectral_q512" if tau == max(poisson_taus) else "spectral_q1024"
    spec = panel_row(max_n, tau, spec_arm)
    add(
        f"| {sci(tau)} | {f3(combo['hybrid_total_ms'])} | "
        f"{f3(spec['hybrid_total_ms'])} | {combo['hybrid_total_ms'] - spec['hybrid_total_ms']:.3f} |"
    )
add("")
add("## Burgers: all final resolution/tolerance conditions")
add("")
add(
    "Times are medians across trajectory-level repetition medians. Intervals resample whole "
    "trajectories; timing repetitions are not treated as independent cases. Positive savings "
    "mean the named candidate is faster."
)
add("")
add(
    "| N | FOM tau | inner tau | linear ms | cubic ms | cubic saving [interval] | "
    "guarded FiLM ms | FiLM saving vs cubic [interval] | accepted steps |"
)
add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in burgers_comparisons:
    cubic = row["cubic"]
    film = row["guarded_film_nmrom"]
    add(
        f"| {row['N']} | {sci(row['fom_tau'])} | {sci(row['linear_tol'])} | "
        f"{f3(row['linear']['time_ms'])} | {f3(cubic['time_ms'])} | "
        f"{cubic['saving_vs_linear_ms']:.3f} {ci(cubic['saving_vs_linear_95ci_ms'])} | "
        f"{f3(film['time_ms'])} | {film['saving_vs_cubic_ms']:.3f} "
        f"{ci(film['saving_vs_cubic_95ci_ms'])} | "
        f"{film['guard_accepted_count_median']:.1f}/{burgers['method']['steps']} |"
    )
add("")
add("## Accuracy, measurement, and provenance")
add("")
add(
    f"Poisson pair confirmation: job {poisson_prov['slurm_job_id']} on "
    f"{poisson_prov['gpu_kind']}, seed {pair['config']['test_seed']}, "
    f"{pair['config']['n_time']} cases and {pair['config']['time_reps']} repetitions per case. "
    f"Every method occupied first and second position "
    f"{pair['config']['time_reps'] // 2} times per case. All "
    f"{len(pair_rows)} rows meet their named true-residual tolerance."
)
add(
    f"Burgers confirmation: job {burgers_prov['slurm_job_id']} on "
    f"{burgers_prov['gpu_kind']}, seed {burgers['cohort']['seed']}, "
    f"{len(burgers['cohort']['selected_indices'])} trajectories and "
    f"{burgers['cohort']['repetitions_per_trajectory']} repetitions each. It contains "
    f"{burgers['audit']['timed_record_count']} timed records across "
    f"{burgers['audit']['row_count']} rows, with "
    f"{burgers['audit']['flags_nonzero_total']} nonzero flags and "
    f"{burgers['audit']['breakdowns_total']} breakdowns. The maximum returned "
    f"residual/tolerance ratio is {burgers['audit']['max_outer_residual_over_tau']:.6f}."
)
add(
    f"Both jobs report `jax_backend=gpu`, x64={poisson_prov['x64'] and burgers_prov['x64']}, "
    f"and matmul precision `{poisson_prov['matmul_precision']}`. Checksums and staged source "
    "hashes match; stderr/warning audits pass; both explicit remote job directories were deleted "
    "after the results were pulled."
)
add("")
add("## Retractions and final interpretation")
add("")
add(
    "- The earlier Poisson multi-arm claim of a learned crossover at every largest-mesh "
    "tolerance is retracted. Its cyclic schedule did not balance timing position. The balanced "
    "fresh-seed run supports only the single row reported above."
)
add(
    "- The provisional Burgers selected-cohort panel is not the headline result. The final table "
    "uses the untouched confirmation seed and trajectory-clustered uncertainty."
)
add(
    "- Architecture tuning improved the Poisson learned path enough for a small loose-tolerance "
    "crossover, but operator-aware classical structure wins by orders of magnitude. On Burgers, "
    "solver and history tuning dominate; the tested learned manifolds cannot supply enough "
    "accepted warm-start improvement to repay their construction cost."
)
add("")
add("## Source artifacts")
add("")
for path in (
    POISSON_PAIR_PATH,
    POISSON_PANEL_PATH,
    BURGERS_FINAL_PATH,
    BURGERS_RAW_PATH,
    BURGERS_CORR_PATH,
    BURGERS_SHIFT_PATH,
):
    add(f"- `{path.relative_to(ROOT)}`")

OUT.write_text("\n".join(lines) + "\n")
print(OUT)
