#!/usr/bin/env python3
"""Build the final 2026-08-20 hybrid speed-push report from audited JSONs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "2026-08-20-hybrid-speed-push.md"


def locate(*candidates: str) -> Path:
    for candidate in candidates:
        path = ROOT / candidate
        if path.exists():
            return path
    raise FileNotFoundError("missing input:\n" + "\n".join(candidates))


POISSON_LM = locate(
    "experiments/poisson-hybrid-1024/runs/paramlmgate1/out/paramlmgate1.json",
    "worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/"
    "runs/paramlmgate1/out/paramlmgate1.json",
)
POISSON_RITZ = locate(
    "experiments/poisson-hybrid-1024/runs/paramritzg1/out/paramritzg1.json",
    "worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/"
    "runs/paramritzg1/out/paramritzg1.json",
)
POISSON_PAIR = locate(
    "experiments/poisson-hybrid-1024/runs/pairfinal1/out/pairfinal1.json",
    "worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/"
    "runs/pairfinal1/out/pairfinal1.json",
)
POISSON_PANEL = locate(
    "experiments/poisson-hybrid-1024/runs/final1/out/final1.json",
    "worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/"
    "runs/final1/out/final1.json",
)
BURGERS_TRAJECTORY = locate(
    "experiments/burgers-hybrid-1024/runs/trajectory_representation/out/trajectory_quality.json",
    "worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/"
    "runs/trajectory_representation/out/trajectory_quality.json",
)
BURGERS_DYNAMIC = locate(
    "experiments/burgers-hybrid-1024/runs/dynamic_final/out/dynamic_final_summary.json",
    "worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/"
    "runs/dynamic_final/out/dynamic_final_summary.json",
)
BURGERS_RAW = locate(
    "experiments/burgers-hybrid-1024/runs/dynamic_final/out/dynamic_final.json",
    "worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/"
    "runs/dynamic_final/out/dynamic_final.json",
)


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


lm = load(POISSON_LM)
ritz = load(POISSON_RITZ)
pair = load(POISSON_PAIR)
panel = load(POISSON_PANEL)
trajectory = load(BURGERS_TRAJECTORY)
dynamic = load(BURGERS_DYNAMIC)
dynamic_raw = load(BURGERS_RAW)


def row_map(data: dict) -> dict[tuple[int, float, str], dict]:
    return {(row["N"], row["fom_tau"], row["arm"]): row for row in data["rows"]}


def fmt_ci(values: list[float]) -> str:
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def poisson_verdict(row: dict) -> str:
    speed = row["speedup_vs_zero_cg_bootstrap_ci95"]
    delta = row["paired_delta_bootstrap_ci95_ms"]
    if speed[0] > 1.0 and delta[1] < 0.0:
        return "supported faster"
    if speed[1] < 1.0 and delta[0] > 0.0:
        return "supported slower"
    return "inconclusive / tie"


def validate() -> None:
    for data, seed in ((lm, 0), (ritz, 20260821), (pair, 20260820)):
        assert data["complete"]
        assert data["config"]["test_seed"] == seed
        provenance = data["provenance"]
        assert provenance["jax_backend"] == "gpu"
        assert provenance["x64"] and provenance["matmul_precision"] == "highest"
        assert not provenance.get("dirty")
    assert panel["complete"]
    assert trajectory["complete"]
    assert dynamic["status"] == "complete"
    assert dynamic["total_cells"] == 18
    assert len(dynamic["rows"]) == 18
    assert dynamic["classification"] == "classical FOM warm start; not learned and not NM-ROM"
    assert dynamic_raw["complete"]
    timed_records = sum(
        len(records)
        for row in dynamic_raw["rows"]
        for records in row["records"].values()
    )
    burn_records = sum(len(row["burn_records"]) for row in dynamic_raw["rows"])
    assert timed_records == 1728
    assert burn_records == 864
    assert all(row["max_returned_residual"] <= row["fom_tau"] for row in dynamic["rows"])
    assert dynamic["supported_speedup_cells"] == sum(
        bool(row["supported_speedup"]) for row in dynamic["rows"]
    )


validate()

lm_rows = row_map(lm)
ritz_rows = row_map(ritz)
pair_rows = sorted(pair["rows"], key=lambda row: (row["N"], -row["fom_tau"]))
pair_supported = [row for row in pair_rows if poisson_verdict(row) == "supported faster"]
assert len(pair_supported) == 1
poisson_headline = pair_supported[0]

trajectory_n256_q64 = next(
    row for row in trajectory["rows"] if row["N"] == 256 and row["coarse_n"] == 64
)

dynamic_rows = sorted(dynamic["rows"], key=lambda row: (row["N"], -row["fom_tau"]))
dynamic_n1024 = {row["fom_tau"]: row for row in dynamic_rows if row["N"] == 1024}
tau6_rows = [row for row in dynamic_rows if row["fom_tau"] == 1e-6]
tau8_rows = [row for row in dynamic_rows if row["fom_tau"] == 1e-8]
tau10_rows = [row for row in dynamic_rows if row["fom_tau"] == 1e-10]
assert all(row["supported_speedup"] for row in tau6_rows)
assert not any(row["supported_speedup"] for row in tau8_rows)

panel_rows = panel["rows"]


def panel_row(tau: float, arm: str) -> dict:
    return next(
        row for row in panel_rows
        if row["N"] == 1024 and row["fom_tau"] == tau and row["arm"] == arm
    )


lines: list[str] = []
add = lines.append
add("# Hybrid speed push through N=1024")
add("")
add(
    "This final report covers the bounded follow-on architecture, objective, and solver search "
    "for Poisson-2D and Burgers-2D. All numbers and tables are generated from the audited run "
    "JSONs; the learned and classical outcomes are kept explicitly separate."
)
add("")
add("## Final outcome")
add("")
add(
    f"- **Poisson learned hybrid:** no new one-update candidate passed. The strongest genuine "
    f"NM-ROM result remains the earlier N={poisson_headline['N']}, tolerance "
    f"{poisson_headline['fom_tau']:.0e} row: {poisson_headline['hybrid_total_ms']:.3f} ms "
    f"versus {poisson_headline['baseline_total_ms']:.3f} ms for counting CG, "
    f"{poisson_headline['speedup_vs_zero_cg']:.3f}x "
    f"{fmt_ci(poisson_headline['speedup_vs_zero_cg_bootstrap_ci95'])}. It is not the fastest "
    "Poisson solver."
)
head = dynamic_n1024[1e-6]
add(
    f"- **Burgers practical winner:** the learned trajectory NM-ROM failed its scaling gate. A "
    f"classical full-grid residual-plus-Helmholtz warm start is supported in "
    f"{dynamic['supported_speedup_cells']}/{dynamic['total_cells']} cells. At N={head['N']} and "
    f"tolerance {head['fom_tau']:.0e}, it takes {head['dynamic_median_ms']:.3f} ms versus "
    f"{head['cubic_median_ms']:.3f} ms for optimized cubic-history FOM, "
    f"{head['speedup_cubic_over_dynamic']:.3f}x, with paired saving "
    f"{head['paired_saving_median_ms']:.3f} ms "
    f"{fmt_ci(head['paired_saving_trajectory_cluster_95ci_ms'])}."
)
add("")
add("## Why the learned routes stopped")
add("")
add(
    "The Poisson conditional candidates started from a direct physical-parameter surrogate and "
    "performed one online weak Gauss--Newton update. Alpha=1 minimized a truncated field-error "
    "objective; the independent follow-on fixed alpha=0.5 to target energy/A-error. Every update "
    "was accepted and reduced its projected objective, but both global A-error and CG work became "
    "worse. The table shows the final solver-aligned round."
)
add("")
add("| N | direct construct ms | direct A-error | direct CG | direct total ms | NM-ROM construct ms | NM-ROM A-error | NM-ROM CG | NM-ROM total ms |")
add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for n in (64, 256):
    base = ritz_rows[(n, 1e-6, "param1_c64_q0")]
    candidate = ritz_rows[(n, 1e-6, "paramritz1_m24_c64_q0")]
    add(
        f"| {n} | {base['construction_ms']:.3f} | {base['guess_a_norm_ratio_mean']:.3e} | "
        f"{base['iters_hybrid_mean']:.2f} | {base['hybrid_total_ms']:.3f} | "
        f"{candidate['construction_ms']:.3f} | {candidate['guess_a_norm_ratio_mean']:.3e} | "
        f"{candidate['iters_hybrid_mean']:.2f} | {candidate['hybrid_total_ms']:.3f} |"
    )
add("")
add(
    f"For Burgers, the one-shot trajectory representation looked compact at N=64 but left "
    f"{trajectory_n256_q64['remaining_ratio_global']:.3f} of the correction at N=256, above "
    f"the frozen 0.25 gate. It therefore never advanced to a weak online wrapper or final timing."
)
add("")
add("## Poisson: strongest learned result and production controls")
add("")
add("| N | tolerance | NM-ROM+FOM ms | counting CG ms | speedup [95% CI] | verdict |")
add("|---:|---:|---:|---:|---|---|")
for row in pair_rows:
    add(
        f"| {row['N']} | {row['fom_tau']:.0e} | {row['hybrid_total_ms']:.3f} | "
        f"{row['baseline_total_ms']:.3f} | {row['speedup_vs_zero_cg']:.3f}x "
        f"{fmt_ci(row['speedup_vs_zero_cg_bootstrap_ci95'])} | {poisson_verdict(row)} |"
    )
add("")
add("At N=1024 the fastest eligible structured controls are:")
add("")
add("| tolerance | control | total ms | counting-CG ms | speedup vs counting CG |")
add("|---:|---|---:|---:|---:|")
for tau in (1e-6, 1e-8, 1e-10):
    arm = "spectral_q512" if tau == 1e-6 else "spectral_q1024"
    spec = panel_row(tau, arm)
    if tau in (1e-6, 1e-8):
        method = "dense DST direct"
        total = spec["dense_dst_direct_ms"]
    else:
        method = arm
        total = spec["hybrid_total_ms"]
    add(
        f"| {tau:.0e} | `{method}` | {total:.3f} | {spec['baseline_total_ms']:.3f} | "
        f"{spec['baseline_total_ms'] / total:.1f}x |"
    )
add("")
add("## Burgers: fresh-seed full-resolution panel")
add("")
add(
    "The successful candidate is **classical and nonlearned**. Before each fine FOM step it "
    "performs one full-grid residual evaluation and one exact Helmholtz inverse. The warmed "
    "online wall time includes all 50 residual evaluations and all 50 inverses. Finish Newton "
    "and BiCGStab counters do not include those extra operations, so wall time is the authoritative "
    "speed metric. Compilation, module loading, data generation, and reference generation are "
    "excluded."
)
add("")
add("| N | FOM tol | cubic ms | corrected ms | speedup | paired saving ms [95% trajectory CI] | supported |")
add("|---:|---:|---:|---:|---:|---|:---:|")
for row in dynamic_rows:
    add(
        f"| {row['N']} | {row['fom_tau']:.0e} | {row['cubic_median_ms']:.3f} | "
        f"{row['dynamic_median_ms']:.3f} | {row['speedup_cubic_over_dynamic']:.3f}x | "
        f"{row['paired_saving_median_ms']:.3f} "
        f"{fmt_ci(row['paired_saving_trajectory_cluster_95ci_ms'])} | "
        f"{'yes' if row['supported_speedup'] else 'no'} |"
    )
add("")
add(
    f"All {len(tau6_rows)} tolerance-1e-6 rows are supported wins; none of the "
    f"{len(tau8_rows)} tolerance-1e-8 rows is supported. At tolerance 1e-10, "
    f"{sum(row['supported_speedup'] for row in tau10_rows)}/{len(tau10_rows)} rows are "
    "supported. The N=1024 tight-tolerance interval is positive but very wide because the final "
    "population contains four trajectories."
)
add("")
add("## Measurement and audit")
add("")
prov = dynamic_raw["provenance"]
timed_record_count = sum(
    len(records)
    for row in dynamic_raw["rows"]
    for records in row["records"].values()
)
burn_record_count = sum(len(row["burn_records"]) for row in dynamic_raw["rows"])
add(
    f"Burgers final job {prov['slurm_job_id']} ran on {prov['gpu_kind']} from clean commit "
    f"`{prov['commit'][:12]}` with GPU backend, f64/x64, and highest matmul precision. It "
    f"contains {timed_record_count:,} timed records and "
    f"{burn_record_count:,} burns in exact AB/BA order. All solver flags and "
    "breakdowns are zero; all named residual tolerances pass. Tukey outliers are counted and "
    "retained, never removed."
)
add("")
add(
    "All referenced cluster job directories were checksum-verified after pull and explicitly "
    "deleted. "
    "The Poisson branch closes at `57329c0`; the Burgers branch closes at `559583c`. Neither "
    "branch was merged automatically."
)
add("")
add("## Input artifacts")
add("")
for path in (
    POISSON_LM, POISSON_RITZ, POISSON_PAIR, POISSON_PANEL,
    BURGERS_TRAJECTORY, BURGERS_DYNAMIC, BURGERS_RAW,
):
    add(f"- `{path.relative_to(ROOT)}`")

OUT.write_text("\n".join(lines) + "\n")
