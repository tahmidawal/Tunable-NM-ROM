"""Generate the final Poisson speed-push audit from immutable run JSONs.

Usage:
  /absolute/python summarize_speed_push.py OUTPUT \
      PARAM_LM_JSON PARAM_RITZ_JSON FINAL_JSON PAIR_FINAL_JSON
"""
from __future__ import annotations

import json
import os
import statistics
import sys


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    if not data.get("complete"):
        raise SystemExit(f"incomplete run: {path}")
    return data


def ms(value):
    return f"{value:.3f}"


def sci(value):
    return f"{value:.3e}"


def pct(value):
    return f"{100.0 * value:.1f}%"


def row_map(data):
    return {(row["N"], row["fom_tau"], row["arm"]): row for row in data["rows"]}


def residual_max(block, method):
    grades = [grade for case in block["timed_telemetry"][method] for grade in case]
    return max(grade["recomputed_true_rel_residual"] for grade in grades)


def verdict(row):
    speed_ci = row["speedup_vs_zero_cg_bootstrap_ci95"]
    delta_ci = row["paired_delta_bootstrap_ci95_ms"]
    if speed_ci[0] > 1.0 and delta_ci[1] < 0.0:
        return "supported faster than counting CG"
    if speed_ci[1] < 1.0 and delta_ci[0] > 0.0:
        return "supported slower than counting CG"
    return "inconclusive/tie versus counting CG"


def validate_runs(param_lm, param_ritz, final, pair):
    expected = [
        (param_lm, 0, {64, 256}),
        (param_ritz, 20260821, {64, 256}),
        (final, 20260819, {32, 64, 128, 256, 512, 1024}),
        (pair, 20260820, {512, 1024}),
    ]
    for data, seed, meshes in expected:
        if data["config"]["test_seed"] != seed:
            raise SystemExit(f"unexpected test seed: {data['config']['test_seed']} != {seed}")
        if set(data["config"]["ns"]) != meshes:
            raise SystemExit(f"unexpected mesh set for seed {seed}: {data['config']['ns']}")
        prov = data["provenance"]
        if prov["jax_backend"] != "gpu" or not prov["x64"]:
            raise SystemExit(f"invalid GPU/f64 provenance for seed {seed}")
        if prov["matmul_precision"] != "highest" or prov.get("dirty"):
            raise SystemExit(f"invalid precision/dirty provenance for seed {seed}")
        if data["config"]["dtype"] != "f64":
            raise SystemExit(f"invalid dtype for seed {seed}: {data['config']['dtype']}")
        for row in data["rows"]:
            if row["final_true_rel_residual_max"] > row["fom_tau"]:
                raise SystemExit(
                    f"same-invocation residual gate failed: seed={seed}, "
                    f"N={row['N']}, arm={row['arm']}"
                )
    for data in (param_lm, param_ritz):
        for mesh in data["mesh_checks"]:
            for block in mesh["joint_timing"].values():
                if any(len(case) != 6 for cases in block["all_s"].values() for case in cases):
                    raise SystemExit("one-update run is missing raw timing repetitions")
                if any(len(cases) != 8 for cases in block["all_s"].values()):
                    raise SystemExit("one-update run is missing timed source cases")


def main():
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    output, lm_path, ritz_path, final_path, pair_path = sys.argv[1:]
    param_lm = load(lm_path)
    param_ritz = load(ritz_path)
    final = load(final_path)
    pair = load(pair_path)
    validate_runs(param_lm, param_ritz, final, pair)

    lines = [
        "# Poisson hybrid speed push (generated)",
        "",
        "This is the generated closing audit for the bounded NM-ROM warm-start speed search. "
        "Its numeric claims come directly from the checksummed run JSONs listed below.",
        "",
        "## Outcome",
        "",
        "The final solver-aligned one-update candidate fails every attribution gate at both "
        "development meshes. It therefore does not advance to the frozen N=1024 confirmation, "
        "and the Poisson nonlinear-warm-start search is exhausted under the pre-registered "
        "construction and production-control budgets.",
        "",
        "The strongest defensible genuine NM-ROM result remains the earlier K8 conditional "
        "nonlinear decoder followed by counting CG: one small balanced win against zero-start "
        "counting CG at N=1024 and tolerance 1e-6. It is not a production speed win: the eligible "
        "dense or spectral/direct controls remain hundreds of times faster.",
        "",
        "## One-update mechanism gates",
        "",
        "`param1_c64_q0` is a direct source-parameter surrogate, not an NM-ROM. The update arms "
        "are conditional nonlinear NM-ROMs because their latent variable is optimized online; "
        "they additionally require the known source parameters for initialization.",
        "",
        "A candidate passes a mesh only when construction is below 0.6 ms and A-error, mean "
        "counting-CG iterations, and same-job total are all strictly lower than the matched "
        "direct surrogate. Total is the stored mean of case medians from the common rotated "
        "timing block.",
        "The alpha=1 seed-0 cohort is the canonical held-out slice 512:528, after the 512 "
        "checkpoint-training cases; the alpha=.5 round uses the wholly new seed shown below.",
        "",
        "| round | seed | N | candidate | construct ms | A-error | CG iters | total ms | "
        "direct construct ms | direct A-error | direct CG iters | direct total ms | gate | "
        "accepted | objective drop |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    mechanism_passes = []
    for label, data in (("alpha=1 field", param_lm), ("alpha=.5 energy", param_ritz)):
        rows = row_map(data)
        candidates = sorted({row["arm"] for row in data["rows"]
                             if row["arm"].startswith(("paramlm1_", "paramritz1_"))})
        for n in data["config"]["ns"]:
            base = rows[(n, 1e-6, "param1_c64_q0")]
            for arm in candidates:
                row = rows[(n, 1e-6, arm)]
                diagnostics = row["lm_diagnostics_per_case"]
                accepted = sum(item["accepted"] for item in diagnostics)
                reduction = 1.0 - statistics.mean(
                    item["objective_reduction"] for item in diagnostics
                )
                passed = (
                    row["construction_ms"] < 0.6
                    and row["guess_a_norm_ratio_mean"] < base["guess_a_norm_ratio_mean"]
                    and row["iters_hybrid_mean"] < base["iters_hybrid_mean"]
                    and row["hybrid_total_ms"] < base["hybrid_total_ms"]
                )
                mechanism_passes.append(passed)
                lines.append(
                    f"| {label} | {data['config']['test_seed']} | {n} | `{arm}` | "
                    f"{ms(row['construction_ms'])} | {sci(row['guess_a_norm_ratio_mean'])} | "
                    f"{row['iters_hybrid_mean']:.2f} | {ms(row['hybrid_total_ms'])} | "
                    f"{ms(base['construction_ms'])} | {sci(base['guess_a_norm_ratio_mean'])} | "
                    f"{base['iters_hybrid_mean']:.2f} | {ms(base['hybrid_total_ms'])} | "
                    f"{'pass' if passed else 'fail'} | {accepted}/{len(diagnostics)} | "
                    f"{pct(reduction)} |"
                )
    if any(mechanism_passes):
        raise SystemExit("generated conclusion disagrees with mechanism gate")

    lines.extend([
        "",
        "Both objectives successfully optimize their truncated weak targets, but neither "
        "improves the global FOM-relevant energy error or CG work. The final alpha=.5 result "
        "therefore isolates a truncated-objective/global-A mismatch rather than a rejected or "
        "failed Gauss--Newton step.",
        "",
        "## Strongest genuine NM-ROM versus matched counting CG",
        "",
        "These rows are the fresh-seed, fully balanced AB/BA audit. A crossover is supported "
        "only when both the case-clustered speed interval is above one and the paired "
        "learned-minus-zero interval is below zero.",
        "",
        "| N | tolerance | construct ms | NM-ROM+FOM ms | zero CG ms | speedup | "
        "speedup 95% CI | paired delta 95% CI ms | verdict |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for row in pair["rows"]:
        speed_ci = row["speedup_vs_zero_cg_bootstrap_ci95"]
        delta_ci = row["paired_delta_bootstrap_ci95_ms"]
        lines.append(
            f"| {row['N']} | {row['fom_tau']:.0e} | {ms(row['construction_ms'])} | "
            f"{ms(row['hybrid_total_ms'])} | {ms(row['baseline_total_ms'])} | "
            f"{row['speedup_vs_zero_cg']:.3f} | [{speed_ci[0]:.3f}, {speed_ci[1]:.3f}] | "
            f"[{delta_ci[0]:.3f}, {delta_ci[1]:.3f}] | {verdict(row)} |"
        )

    lines.extend([
        "",
        "## N=1024 production controls",
        "",
        "The controls below were measured in the same fresh-seed rotated job. Eligibility uses "
        "the maximum true residual recomputed from each timed invocation. The fastest eligible "
        "row at each tolerance is the production comparator.",
        "",
        "| tolerance | method | total ms | max true residual | eligible | fastest eligible |",
        "|---:|---|---:|---:|---|---|",
    ])
    mesh = next(item for item in final["mesh_checks"] if item["N"] == 1024)
    production_methods = ("dense_dst_direct", "fft_dst_direct", "spectral_q1024")
    for tau_text, block in sorted(
        mesh["joint_timing"].items(), key=lambda item: float(item[0]), reverse=True
    ):
        tau = float(tau_text)
        records = []
        for method in production_methods:
            total = block["mean_of_case_medians_ms"][method]
            residual = residual_max(block, method)
            records.append((method, total, residual, residual <= tau))
        fastest = min((rec for rec in records if rec[3]), key=lambda rec: rec[1])[0]
        for method, total, residual, eligible in records:
            lines.append(
                f"| {tau:.0e} | `{method}` | {ms(total)} | {sci(residual)} | "
                f"{'yes' if eligible else 'no'} | {'yes' if method == fastest else 'no'} |"
            )

    lines.extend([
        "",
        "## Audit trail",
        "",
        "| run | seed | job | GPU | commit | source hash |",
        "|---|---:|---:|---|---|---|",
    ])
    for path, data in (
        (lm_path, param_lm), (ritz_path, param_ritz),
        (final_path, final), (pair_path, pair),
    ):
        provenance = data["provenance"]
        lines.append(
            f"| `{os.path.basename(path)}` | {data['config']['test_seed']} | "
            f"{provenance['slurm_job_id']} | {provenance['gpu_kind']} | "
            f"{provenance['commit_short']} | "
            f"{provenance['source_sha256']['feasibility.py']} |"
        )
    lines.extend([
        "",
        "All four inputs are complete GPU/f64/highest-precision runs. The two one-update runs "
        "persist six raw timing repetitions for each of eight timed cases, retain direct and "
        "spectral controls, and pass their same-invocation true-residual gates. Pull checksums "
        "are stored beside each run; the cluster job directories were deleted after verified "
        "pulls.",
        "",
        "## Input files",
        "",
    ])
    lines.extend(f"- `{path}`" for path in (lm_path, ritz_path, final_path, pair_path))
    with open(output, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
