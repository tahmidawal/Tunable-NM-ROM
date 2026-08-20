#!/usr/bin/env python3
"""Build the final cross-PDE N=2048 report from independently audited JSONs.

A table is more informative than a chart here: each experiment has only three tolerance
conditions, while confidence intervals, support status, and method classification are essential.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "2026-08-20-hybrid-warm-starts-at-2048.md"


def locate(*candidates: str) -> Path:
    for candidate in candidates:
        path = ROOT / candidate
        if path.exists():
            return path
    raise FileNotFoundError("missing input:\n" + "\n".join(candidates))


POISSON = locate(
    "worktrees/2026-08-20-poisson-hybrid-2048/experiments/poisson-hybrid-1024/"
    "runs/n2048final1/out/n2048final1.json",
)
POISSON_AUDIT = locate(
    "worktrees/2026-08-20-poisson-hybrid-2048/experiments/poisson-hybrid-1024/"
    "runs/n2048final1/INDEPENDENT-AUDIT.json",
)
BURGERS_PRIMARY = locate(
    "worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/"
    "runs/final3/out/final.json",
)
BURGERS_PRIMARY_AUDIT = locate(
    "worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/"
    "runs/final3/INDEPENDENT-AUDIT.json",
)
BURGERS_LEARNED = locate(
    "worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/"
    "runs/learned1/out/learned.json",
)
BURGERS_LEARNED_AUDIT = locate(
    "worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/"
    "runs/learned1/LEARNED-AUDIT.json",
)


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


poisson = load(POISSON)
poisson_audit = load(POISSON_AUDIT)
burgers = load(BURGERS_PRIMARY)
burgers_audit = load(BURGERS_PRIMARY_AUDIT)
learned = load(BURGERS_LEARNED)
learned_audit = load(BURGERS_LEARNED_AUDIT)


def fmt_ci(values: list[float], digits: int = 3) -> str:
    return f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def learned_poisson_verdict(row: dict) -> str:
    ratio_ci = row["speedup_vs_zero_cg_bootstrap_ci95"]
    delta_ci = row["paired_delta_bootstrap_ci95_ms"]
    if ratio_ci[0] > 1.0 and delta_ci[1] < 0.0:
        return "supported faster"
    if ratio_ci[1] < 1.0 and delta_ci[0] > 0.0:
        return "supported slower"
    return "unsupported / inconclusive"


def validate() -> None:
    assert poisson["complete"] and poisson_audit["audit_pass"]
    assert sha256(POISSON) == poisson_audit["source_json_sha256"]
    assert poisson["config"]["ns"] == [2048]
    assert poisson["config"]["test_seed"] == 20260826
    assert poisson["config"]["fom_taus"] == [1e-6, 1e-8, 1e-10]
    assert poisson["provenance"]["jax_backend"] == "gpu"
    assert poisson["provenance"]["x64"]
    assert poisson["provenance"]["matmul_precision"] == "highest"
    assert poisson_audit["record_counts"] == {
        "learned": 576,
        "production": 1200,
        "outliers": 0,
    }
    assert len(poisson_audit["learned_pair"]) == 3
    assert sum(row["supported"] for row in poisson_audit["learned_pair"]) == 1
    assert all(
        row["learned_true_residual_max"] <= row["tau"]
        and row["zero_true_residual_max"] <= row["tau"]
        for row in poisson_audit["learned_pair"]
    )

    assert burgers["complete"] and burgers_audit["status"] == "pass"
    assert sha256(BURGERS_PRIMARY) == burgers_audit["source_sha256"]
    assert burgers["config"]["ns"] == [2048]
    assert burgers["config"]["test_seed"] == 20260830
    assert burgers["config"]["classification"]["dynamic"].endswith(
        "not learned and not NM-ROM"
    )
    assert burgers["provenance"]["jax_backend"] == "gpu"
    assert burgers["provenance"]["x64"]
    assert burgers["provenance"]["matmul_precision"] == "highest"
    assert burgers_audit["timing_record_count"] == 288
    assert burgers_audit["burn_record_count"] == 144
    assert burgers_audit["reference"]["all_pass"]
    assert len(burgers_audit["rows"]) == 3
    assert burgers_audit["supported_speedup_cells"] == 1
    assert all(
        row["max_returned_residual"] <= row["fom_tau"]
        for row in burgers_audit["rows"]
    )

    assert learned["complete"] and learned_audit["status"] == "complete"
    assert sha256(BURGERS_LEARNED) == learned_audit["source_sha256"]
    assert learned["config"]["N"] == 2048
    assert learned["config"]["test_seed"] == 20260828
    assert learned["config"]["classification"]["film_nmrom"].startswith(
        "genuine weak FiLM NM-ROM"
    )
    assert learned["provenance"]["jax_backend"] == "gpu"
    assert learned["provenance"]["x64"]
    assert learned["provenance"]["matmul_precision"] == "highest"
    assert sum(len(row["records"]) for row in learned["rows"]) == 576
    assert sum(len(row["burn_records"]) for row in learned["rows"]) == 288
    assert learned_audit["film_supported_vs_cubic_cells"] == 0
    assert learned_audit["film_supported_vs_dynamic_cells"] == 0
    assert all(
        not summary["supported_film_speedup"]
        for row in learned["rows"]
        for summary in row["summaries"].values()
    )


validate()

poisson_rows = sorted(poisson_audit["learned_pair"], key=lambda row: -row["tau"])
poisson_controls = [
    row
    for row in poisson_audit["production_controls"]
    if row["method"] == "spectral_q1024"
]
poisson_controls.sort(key=lambda row: -row["tau"])
burgers_rows = sorted(burgers_audit["rows"], key=lambda row: -row["fom_tau"])
learned_rows = sorted(learned["rows"], key=lambda row: -row["fom_tau"])

poisson_supported = next(row for row in poisson_rows if row["supported"])
burgers_supported = next(row for row in burgers_rows if row["supported_speedup"])

lines: list[str] = []
add = lines.append
add("# Hybrid warm starts at N=2048")
add("")
add(
    "This final report extends the audited Poisson-2D and Burgers-2D hybrid studies to "
    "N=2048. It separates genuine learned NM-ROM results from classical solver improvements; "
    "every number below is generated from checksummed, independently audited run JSONs."
)
add("")
add("## Technical summary")
add("")
add(
    f"The genuine Poisson K=8 NM-ROM warm start has one supported crossover: at tolerance "
    f"{poisson_supported['tau']:.0e}, it takes {poisson_supported['learned_median_ms']:.3f} ms "
    f"versus {poisson_supported['zero_median_ms']:.3f} ms for counting CG, "
    f"{poisson_supported['speedup']:.3f}x with ratio interval "
    f"{fmt_ci(poisson_supported['speedup_ci95'])}. The two tighter tolerances are unsupported."
)
add("")
add(
    f"The genuine Burgers FiLM NM-ROM wins none of its three same-job comparisons against "
    f"either optimized cubic history or the classical correction. The practical Burgers result "
    f"is instead nonlearned: at tolerance {burgers_supported['fom_tau']:.0e}, the charged "
    f"residual-plus-Helmholtz correction reduces {burgers_supported['cubic_median_ms']:.3f} ms "
    f"to {burgers_supported['dynamic_median_ms']:.3f} ms, "
    f"{burgers_supported['speedup_cubic_over_dynamic']:.3f}x, with paired-saving interval "
    f"{fmt_ci(burgers_supported['paired_saving_trajectory_cluster_95ci_ms'])} ms. Its tighter "
    "rows are unsupported."
)
add("")
add(
    "Operationally, Poisson's learned crossover is not competitive with the structure-aware "
    "solver: the eligible q=1024 spectral warm start is hundreds of times faster than zero-start "
    "CG at all three tolerances."
)
add("")
add("## Poisson: a learned crossover only at loose tolerance")
add("")
add(
    "The learned arm is the fixed-N=64 K=8 trust-region NM-ROM decode followed by the same "
    "true-residual counting CG as the zero-start baseline. Each row uses eight fresh cases, "
    "12 repetitions per method and case, exact AB/BA position balance, a fresh burn before both "
    "orders, and whole-case bootstrap inference."
)
add("")
add("| FOM tolerance | K8 NM-ROM+CG ms | zero-start CG ms | speedup [95% CI] | paired saving ms [95% CI] | case signs K8/zero | verdict |")
add("|---:|---:|---:|---|---|---:|---|")
for row in poisson_rows:
    delta = -row["paired_delta_ms"]
    delta_ci = [-row["paired_delta_ci95_ms"][1], -row["paired_delta_ci95_ms"][0]]
    signs = row["case_signs"]
    add(
        f"| {row['tau']:.0e} | {row['learned_median_ms']:.3f} | "
        f"{row['zero_median_ms']:.3f} | {row['speedup']:.3f}x "
        f"{fmt_ci(row['speedup_ci95'])} | {delta:.3f} {fmt_ci(delta_ci)} | "
        f"{signs['learned_faster']}/{signs['zero_faster']} | "
        f"{learned_poisson_verdict({'speedup_vs_zero_cg_bootstrap_ci95': row['speedup_ci95'], 'paired_delta_bootstrap_ci95_ms': row['paired_delta_ci95_ms']})} |"
    )
add("")
add(
    f"All {poisson_audit['record_counts']['learned']} learned/zero timing records meet their "
    f"true-residual and boundary gates. The diagnostic outlier count is "
    f"{poisson_audit['record_counts']['outliers']}; no sample was removed."
)
add("")
add("## Poisson: the production solver remains classical")
add("")
add(
    "The separate balanced five-method block compares each method at every clock position and "
    "uses the same cases and GPU within the block. The fastest eligible method at every tolerance "
    "is the partial q=1024 sine-mode solve followed, when needed, by counting CG."
)
add("")
add("| FOM tolerance | spectral q=1024 ms | same-block zero CG ms | speedup vs zero | mean finishing CG iterations | eligible |")
add("|---:|---:|---:|---:|---:|:---:|")
for row in poisson_controls:
    add(
        f"| {row['tau']:.0e} | {row['median_ms']:.3f} | "
        f"{row['median_ms'] * row['speedup_vs_zero']:.3f} | "
        f"{row['speedup_vs_zero']:.1f}x | {row['iterations_mean']:.3f} | "
        f"{'yes' if row['eligible'] else 'no'} |"
    )
add("")
add(
    "Dense and FFT DST direct solves also pass the 1e-6 and 1e-8 gates, but their measured true "
    "residuals miss 1e-10. The q=1024 arm remains eligible there. This rectangular, separable "
    "Poisson family is therefore a strong case for exploiting known operator structure rather "
    "than adding a learned warm start."
)
add("")
add("## Burgers: the genuine FiLM NM-ROM remains slower")
add("")
add(
    "The learned sensitivity uses a genuine K=8 weak FiLM NM-ROM with M=64, m=256, at most two "
    "latent Jacobians per step, fixed-N=64 decoding and prolongation, and a charged exact-residual "
    "guard. The table contains only comparisons made inside its own H200 job and seed; absolute "
    "times are not mixed with the separate classical final job. A positive control-minus-FiLM "
    "saving would favor FiLM."
)
add("")
add("| FOM tolerance | control | control ms | FiLM ms | control/FiLM | paired control-minus-FiLM ms [95% CI] | FiLM supported faster |")
add("|---:|---|---:|---:|---:|---|:---:|")
for row in learned_rows:
    for control in ("cubic", "dynamic"):
        summary = row["summaries"][control]
        add(
            f"| {row['fom_tau']:.0e} | {control} | "
            f"{summary['control_median_ms']:.3f} | {summary['film_nmrom_median_ms']:.3f} | "
            f"{summary['speedup_control_over_film_nmrom']:.3f}x | "
            f"{summary['paired_saving_control_minus_film_median_ms']:.3f} "
            f"{fmt_ci(summary['paired_saving_trajectory_cluster_95ci_ms'])} | "
            f"{'yes' if summary['supported_film_speedup'] else 'no'} |"
        )
add("")
add(
    f"The residual guard accepts a median of {learned_rows[0]['summaries']['cubic']['film_guard_accepted_count_median']:.0f} "
    f"of 50 steps while the arm performs "
    f"{learned_rows[0]['summaries']['cubic']['reduced_jacobians_total_median']:.0f} reduced "
    "Jacobians per trajectory. That construction does not repay itself at N=2048."
)
add("")
add("## Burgers: a classical loose-tolerance speedup")
add("")
add(
    "The practical arm is not learned and is not an NM-ROM. Before every fine FOM step it "
    "evaluates one target-grid exact-upwind residual and applies one exact Helmholtz inverse to "
    "the live cubic prediction. Its measured wall time includes all 50 residual evaluations and "
    "50 inverses; the reported finishing Newton/BiCGStab counters exclude those extra operations."
)
add("")
add("| FOM tolerance | cubic ms | corrected ms | speedup | paired saving ms [95% trajectory CI] | finish Newton cubic/corrected | finish BiCG cubic/corrected | verdict |")
add("|---:|---:|---:|---:|---|---:|---:|---|")
for row in burgers_rows:
    verdict = "supported faster" if row["supported_speedup"] else "unsupported / inconclusive"
    add(
        f"| {row['fom_tau']:.0e} | {row['cubic_median_ms']:.3f} | "
        f"{row['dynamic_median_ms']:.3f} | {row['speedup_cubic_over_dynamic']:.3f}x | "
        f"{row['paired_saving_median_ms']:.3f} "
        f"{fmt_ci(row['paired_saving_trajectory_cluster_95ci_ms'])} | "
        f"{row['finish_newton_median']['cubic']:.1f}/{row['finish_newton_median']['dynamic']:.1f} | "
        f"{row['finish_bicgstab_median']['cubic']:.1f}/{row['finish_bicgstab_median']['dynamic']:.1f} | "
        f"{verdict} |"
    )
add("")
add(
    f"All {burgers_audit['timing_record_count']} timed records and "
    f"{burgers_audit['burn_record_count']} burns pass the timing-grid, order, solver-health, "
    "accuracy, and same-invocation checks. The one supported row is 1e-6; the point estimates at "
    "1e-8 and 1e-10 do not establish a benefit."
)
add("")
add("## Reference and failure audit")
add("")
reference = burgers_audit["reference"]
add(
    "The Burgers final timing was allowed only after two separately compiled exact-Helmholtz "
    "reference routes passed on all four fresh trajectories. Their worst actual outer residual "
    f"is {reference['max_actual_outer_relative_residual']:.3e}; maximum step and trajectory "
    f"field disagreements are {reference['max_step_relative_field_difference']:.3e} and "
    f"{reference['max_trajectory_relative_field_difference']:.3e}. Both routes have zero flags "
    "and breakdowns."
)
add("")
add(
    "Two earlier Burgers N=2048 timing attempts are excluded pre-science because the inherited "
    "public JAX reference froze on one development trajectory. A batch-shape diagnostic then "
    "reproduced nonfinite public updates and did not license a replacement. The final reference "
    "design was frozen prospectively and evaluated on a new seed before any final timing. An "
    "earlier diagnostic also exposed a shell `producer && checker` false-success pattern; that "
    "diagnostic is excluded, and the authoritative final uses separate fail-closed producer and "
    "checker commands."
)
add("")
add("## Scope and limitations")
add("")
add(
    f"Poisson used seed {poisson['config']['test_seed']} on an "
    f"{poisson['provenance']['gpu_kind']} (job {poisson['provenance']['slurm_job_id']}); the "
    f"learned Burgers sensitivity used seed {learned['config']['test_seed']} on an "
    f"{learned['provenance']['gpu_kind']} (job {learned['provenance']['slurm_job_id']}); the "
    f"classical Burgers final used seed {burgers['config']['test_seed']} on an "
    f"{burgers['provenance']['gpu_kind']} (job {burgers['provenance']['slurm_job_id']}). "
    "No wall-clock number is compared across jobs or GPU types."
)
add("")
add(
    "All timings are warmed, compiled steady-state online costs. Checkpoint loading, compilation, "
    "data/reference generation, and first-query latency are excluded. Poisson inference uses eight "
    "timed cases; Burgers uses four trajectories, so its trajectory-cluster intervals—especially "
    "at tighter tolerances—remain sample-limited. These results establish N=2048 behavior for the "
    "tested families, not a universal ranking on irregular domains or different operators."
)
add("")
add("## Recommended operating policy")
add("")
add(
    "1. On this separable Poisson rectangle, use the q=1024 spectral warm start rather than the "
    "learned hybrid. Retain the K8 result as evidence that learned construction can cross counting "
    "CG at large N, not as the production winner."
)
add(
    "2. For this Burgers family at tolerance 1e-6, use the charged residual-plus-Helmholtz "
    "correction. Keep cubic history at 1e-8 and 1e-10 because the N=2048 correction rows are "
    "unsupported."
)
add(
    "3. Stop tuning the current Burgers FiLM warm start. A future learned route needs a materially "
    "better transported trajectory manifold and much lower online construction, and should first "
    "be tested on nonseparable settings where exact structured solvers are unavailable."
)
add("")
add("## Generated inputs")
add("")
for label, path in (
    ("Poisson final JSON", POISSON),
    ("Poisson independent audit", POISSON_AUDIT),
    ("Burgers classical final JSON", BURGERS_PRIMARY),
    ("Burgers classical independent audit", BURGERS_PRIMARY_AUDIT),
    ("Burgers learned JSON", BURGERS_LEARNED),
    ("Burgers learned audit", BURGERS_LEARNED_AUDIT),
):
    add(f"- {label}: `{path.relative_to(ROOT)}`")
add("")

OUT.write_text("\n".join(lines))
print(OUT)
