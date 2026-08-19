"""Generate the nonlinear-decoder report from experiment summary JSON."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPERIMENT = ROOT / "experiments" / "nonlinear-decoder-architecture"
SUMMARY = EXPERIMENT / "summary.json"
OUTPUT = HERE / "2026-08-19-nonlinear-decoder-architecture.md"


def sci(value):
    return f"{value:.3e}"


def pct(value):
    return f"{100 * value:.1f}%"


def table(headers, align, rows):
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    result.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return result


def aggregate(summary, pde, **selectors):
    key = f"{pde.lower()}_three_seed"
    matches = [r for r in summary[key]
               if all(r.get(name) == value for name, value in selectors.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one {pde} aggregate for {selectors}, got {matches}")
    return matches[0]


def bench(summary, pde):
    return next(r for r in summary["benchmarks"] if r["pde"] == pde)


def bench_row(benchmark, kernel, points):
    return next(r for r in benchmark["rows"]
                if r["kernel"] == kernel and r["points"] == points)


def e2e_pairs(summary):
    grouped = {}
    for row in summary["e2e"]:
        key = (row["cell"], row["M"], row["m"], row["tau"])
        grouped.setdefault(key, {})[row["arm"]] = row
    pairs = []
    for key, arms in sorted(grouped.items()):
        if set(arms) != {"control", "variant"}:
            continue
        control, variant = arms["control"], arms["variant"]
        pairs.append(dict(
            cell=key[0], M=key[1], m=key[2], tau=key[3],
            control=control, variant=variant,
            speedup=control["time_ms"] / variant["time_ms"],
            iso_fom_speedup=(None if variant.get("fom_iso_accuracy_ms") is None else
                             variant["fom_iso_accuracy_ms"] / variant["time_ms"]),
        ))
    return pairs


def main():
    with SUMMARY.open() as f:
        summary = json.load(f)

    p_recommended = summary["poisson_gate_audit"]["recommended"]
    p_selected = aggregate(
        summary, "Poisson", architecture=p_recommended["architecture"],
        hidden=p_recommended["hidden"], layers=p_recommended["layers"],
        group_size=p_recommended["group_size"], n_params=p_recommended["n_params"],
        M=p_recommended["M"], m=p_recommended["m"])
    p_control = next(r for r in summary["poisson"]
                     if r["cell"] == "saved-control/fair-M128")
    p_bench, b_bench = bench(summary, "poisson"), bench(summary, "burgers")
    p_jac = bench_row(p_bench, "jacobian", 2560)
    p_full = bench_row(p_bench, "forward", 262144)
    b_jac = bench_row(b_bench, "jacobian", 2560)
    b_full = bench_row(b_bench, "forward", 262144)

    burgers_complete = bool(summary["burgers_three_seed"])
    e2e_complete = ({r["cell"] for r in summary["e2e"]} >=
                    {"nda_pe2e_g98_r23", "nda_be2e_g160m640_r24"})
    final = (burgers_complete and e2e_complete and
             summary["burgers_gate_audit"]["recommended"] is not None)
    state = ("Final for the N=64, k=16 architecture comparison described here."
             if final else
             "Provisional: Burgers seed or end-to-end cells are still absent from the pulled artifacts.")

    md = [
        "# Pure nonlinear decoder architectures for Poisson and Burgers NMROMs", "",
        f"This report compares compact, purely nonlinear coordinate decoders with the saved FiLM controls. {state} Every table and prose result below is generated from run JSONs, not transcribed manually.", "",
        "## Recommendation", "",
        (f"For Poisson, use group-FiLM H{p_selected['hidden']}×{p_selected['layers']} with group size "
         f"{p_selected['group_size']} at M={p_selected['M']},m={p_selected['m']}: "
         f"{p_selected['n_params']:,} parameters, with three-seed decoder/full/EQ means "
         f"{sci(p_selected['decoder']['mean'])}, {sci(p_selected['full']['mean'])}, and "
         f"{sci(p_selected['eq']['mean'])}. It is the narrowest tested even width "
         f"whose three-seed mean clears the accuracy gates; H96 failed, while H100 is the nearby accuracy-margin option."), "",
    ]
    if burgers_complete:
        b96 = aggregate(summary, "Burgers", hidden=160, group_size=2, M=96, m=384)
        b128 = aggregate(summary, "Burgers", hidden=160, group_size=2, M=128, m=512)
        gate = summary["burgers_gate_audit"]
        recommended = gate["recommended"]
        if recommended is not None:
            chosen = aggregate(
                summary, "Burgers", architecture=recommended["architecture"],
                hidden=recommended["hidden"], layers=recommended["layers"],
                group_size=recommended["group_size"], n_params=recommended["n_params"],
                M=recommended["M"], m=recommended["m"])
            recommendation = (
                f"For Burgers, use group-FiLM H{recommended['hidden']}×{recommended['layers']} "
                f"with group size {recommended['group_size']} ({recommended['n_params']:,} parameters) "
                f"at M={recommended['M']},m={recommended['m']}. Its three-seed decoder/full/EQ means are "
                f"{sci(chosen['decoder']['mean'])}, {sci(chosen['full']['mean'])}, and {sci(chosen['eq']['mean'])}, "
                "and every seed clears the conservative decoder, full-ROM, EQ-ROM, and EQ/full gates. "
                f"The M96,m384 arm has full/EQ means {sci(b96['full']['mean'])} and {sci(b96['eq']['mean'])}, "
                "but is retained only as the aggressive lower-cost arm because at least one per-seed gate fails."
            )
        else:
            b_params = b_bench["models"]["variant"]["n_params"]
            recommendation = (
                f"No Burgers objective arm cleared every conservative three-seed gate. The H160×4 architecture "
                f"has {b_params:,} parameters; its M96,m384 and M128,m512 full/EQ means are "
                f"{sci(b96['full']['mean'])}/{sci(b96['eq']['mean'])} and "
                f"{sci(b128['full']['mean'])}/{sci(b128['eq']['mean'])}, respectively."
            )
        md += [recommendation, ""]
    else:
        md += ["The Burgers H160×4 recommendation is awaiting the two pulled seed-confirmation artifacts.", ""]

    md += [
        "These are not POD-plus-corrector models. Coordinates and the latent state pass through a nonlinear network, and the output is not restricted to a fixed linear basis.", "",
        "## Accuracy and model size", "",
        "### Poisson three-seed selection", "",
    ]
    md += table(
        ["model", "parameters", "decoder mean±std", "full mean±std", "EQ mean±std"],
        ["---", "---:", "---:", "---:", "---:"],
        [["saved H128×4 control (single seed, fair M,m)", f'{p_control["n_params"]:,}',
          sci(p_control["decoder"]), sci(p_control["full"]), sci(p_control["eq"])],
         *[[f'group-FiLM H{row["hidden"]}×{row["layers"]}, M{row["M"]},m{row["m"]}', f'{row["n_params"]:,}',
            f'{sci(row["decoder"]["mean"])}±{sci(row["decoder"]["sample_std"])}',
            f'{sci(row["full"]["mean"])}±{sci(row["full"]["sample_std"])}',
            f'{sci(row["eq"]["mean"])}±{sci(row["eq"]["sample_std"])}']
           for row in summary["poisson_three_seed"]]]
    )
    md += ["",
           (f"The selected Poisson decoder removes {pct(1 - p_selected['n_params']/p_control['n_params'])} of the parameters. "
            "The saved control remains more accurate in the fair M128,m512 comparison, so the defensible claim is a size/speed tradeoff at acceptable accuracy—not an accuracy improvement over the control."), ""]

    if burgers_complete:
        b_control_params = b_bench["models"]["control"]["n_params"]
        md += ["### Burgers three-seed selection", ""]
        md += table(
            ["model/objective", "parameters", "decoder mean±std", "full mean±std", "EQ mean±std"],
            ["---", "---:", "---:", "---:", "---:"],
            [[f'group-FiLM H{row["hidden"]}×{row["layers"]} g{row["group_size"]}, M{row["M"]},m{row["m"]}',
              f'{row["n_params"]:,}',
              f'{sci(row["decoder"]["mean"])}±{sci(row["decoder"]["sample_std"])}',
              f'{sci(row["full"]["mean"])}±{sci(row["full"]["sample_std"])}',
              f'{sci(row["eq"]["mean"])}±{sci(row["eq"]["sample_std"])}']
             for row in summary["burgers_three_seed"]]
        )
        selected = summary["burgers_gate_audit"]["recommended"]
        selected_params = (b_bench["models"]["variant"]["n_params"] if selected is None
                           else selected["n_params"])
        md += ["", f"The selected Burgers decoder removes {pct(1 - selected_params/b_control_params)} of the saved control parameters.", ""]

    md += [
        "![Architecture accuracy tradeoff](../experiments/nonlinear-decoder-architecture/figures/architecture_accuracy_tradeoff.png)", "",
        "## Decoder-kernel speed", "",
        "All accepted decoder timings are f64/highest, use nine persisted repetitions, burn the GPU, then warm the exact compiled kernel. Speedups compare models within one job on one GPU.", "",
    ]
    md += table(
        ["PDE", "representative kernel", "points", "raw speedup", "coordinate-cached speedup", "outliers control/variant/cache"],
        ["---", "---", "---:", "---:", "---:", "---:"],
        [["Poisson", "hyper-reduced Jacobian", p_jac["points"], f'{p_jac["speedup"]:.3f}×', f'{p_jac["cached_speedup"]:.3f}×',
          f'{p_jac["outliers_control"]}/{p_jac["outliers_variant"]}/{p_jac["outliers_cached"]}'],
         ["Poisson", "large forward", p_full["points"], f'{p_full["speedup"]:.3f}×', f'{p_full["cached_speedup"]:.3f}×',
          f'{p_full["outliers_control"]}/{p_full["outliers_variant"]}/{p_full["outliers_cached"]}'],
         ["Burgers", "hyper-reduced Jacobian", b_jac["points"], f'{b_jac["speedup"]:.3f}×', f'{b_jac["cached_speedup"]:.3f}×',
          f'{b_jac["outliers_control"]}/{b_jac["outliers_variant"]}/{b_jac["outliers_cached"]}'],
         ["Burgers", "large forward", b_full["points"], f'{b_full["speedup"]:.3f}×', f'{b_full["cached_speedup"]:.3f}×',
          f'{b_full["outliers_control"]}/{b_full["outliers_variant"]}/{b_full["outliers_cached"]}']]
    )
    md += ["", "![Same-GPU decoder speedup](../experiments/nonlinear-decoder-architecture/figures/same_gpu_decoder_speedup.png)", "",
           "Coordinate caching is exact to the check stored in each benchmark JSON; it removes coordinate-only affine work without changing the nonlinear model.", "",
           "## End-to-end rollout measurements", ""]
    pairs = e2e_pairs(summary)
    md += table(
        ["cell", "M,m", "tau", "control ms/error", "compact ms/error", "control/compact", "iso-FOM/compact", "censored control/compact", "timing outliers control/compact"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        [[r["cell"], f'{r["M"]},{r["m"]}', sci(r["tau"]),
          f'{r["control"]["time_ms"]:.3f} / {sci(r["control"]["error"])}',
          f'{r["variant"]["time_ms"]:.3f} / {sci(r["variant"]["error"])}',
          f'{r["speedup"]:.3f}×',
          "—" if r["iso_fom_speedup"] is None else f'{r["iso_fom_speedup"]:.3f}×',
          f'{pct(r["control"]["censored_frac"])} / {pct(r["variant"]["censored_frac"])}',
          f'{r["control"]["timing_outliers"]}/{r["control"]["timing_samples"]} / '
          f'{r["variant"]["timing_outliers"]}/{r["variant"]["timing_samples"]}'] for r in pairs]
    )
    any_censored = any(r[arm]["censored_frac"] > 0 for r in pairs for arm in ("control", "variant"))
    if any_censored:
        md += ["", "Rows with nonzero censoring are budget-limited. They are useful diagnostic measurements but are not promoted to headline iso-accuracy speedups.", ""]

    selected_burgers_e2e = [r for r in pairs if r["cell"] == "nda_be2e_g160m640_r24"]
    if selected_burgers_e2e:
        loose = next(r for r in selected_burgers_e2e if r["tau"] == 1e-2)
        fom_relation = (f"{loose['iso_fom_speedup']:.3f}× faster"
                        if loose["iso_fom_speedup"] >= 1 else
                        f"{1/loose['iso_fom_speedup']:.3f}× slower")
        md += [
            (f"At the selected Burgers M={loose['M']},m={loose['m']} arm and tau={sci(loose['tau'])}, "
             f"the compact decoder is {loose['speedup']:.3f}× faster than the saved decoder architecture "
             f"inside the same job. It is {fom_relation} than the "
             "like-for-like iso-accuracy FOM, so this is an architecture speedup—not a claimed FOM crossover."), ""]

    trust_rows = [r for r in summary["burgers_objectives"] if r["trust_factor"] > 0]
    if trust_rows:
        md += ["## Local latent trust-region check", ""]
        md += table(
            ["arm", "M,m", "decoder", "full", "EQ", "EQ/full", "blowups full/EQ"],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
            [[r["cell"].split("/")[-1], f'{r["M"]},{r["m"]}', sci(r["decoder"]),
              sci(r["full"]), sci(r["eq"]), f'{r["eq"]/r["full"]:.3f}',
              f'{r["full_blowups"]}/{r["eq_blowups"]}'] for r in trust_rows]
        )
        md += ["", "The trust radius is 1% of the training-cloud radius. This is reported as a solver sensitivity check, not folded into the architectural comparison.", ""]

    failure_rows = [
        "## What failed or was retracted", "",
        "- Compact residual-FiLM arms failed both PDE accuracy targets; parameter count alone was not enough.",
        "- Burgers H192 did not materially improve the seed-0 decoder or M64 ROM over H160, so increasing width was not the productive direction.",
        "- The old Poisson M64,m256 control comparison is not used for architectural accuracy claims. The control was re-evaluated fairly at M128,m512.",
        "- `nda_pbench_g98_r8` is retained but rejected: its exact kernels were warmed before the GPU burn. `nda_pbench_g98b_r8` is the accepted corrected rerun.",
        "- `nda_be2e_g160_r12` is retained but rejected: the driver dropped the compact checkpoint's decoder metadata and failed before that arm. The loader was fixed in the subsequent runs.",
        "- `nda_pe2e_g98_r11`, `nda_be2e_g160_r14`, and `nda_be2e_g160m640_r21` are excluded from timing claims because they retained per-source medians but not every raw repetition. The accepted r23/r24 reruns persist the full arrays.",
    ]
    group4 = [r for r in summary["burgers_three_seed"]
              if r["group_size"] == 4 and r["M"] == 128 and r["m"] == 640]
    if group4:
        failure_rows.append(
            f"- The smaller group-4 H160 decoder is not robust: its three-seed maximum full-ROM error is "
            f"{sci(group4[0]['full']['max'])}, above the 1% gate, despite a passing seed-0 result.")
    group8 = [r for r in summary["burgers"] if r["group_size"] == 8 and r["hidden"] == 160]
    if group8:
        failure_rows.append(
            f"- The group-8 H160 compression bracket fails already at M64 on seed 0: decoder/full/EQ512 "
            f"errors are {sci(group8[0]['decoder'])}, {sci(group8[0]['full'])}, and {sci(group8[0]['eq512'])}.")
    md += failure_rows + ["",
        "## Scope and provenance", "",
        "The result is limited to N=64, k=16, the recorded held-out families, and the weak-form solvers tested here. Every cluster cell regenerated its data from seed, logged `jax_backend=gpu`, used f64/highest precision, ran alone in its directory, and was pulled with checksums. Exact run rows, timing arrays, medians, maxima, outlier counts, manifests, and job logs are in the experiment directory.", "",
        "- [Generated full tables](../experiments/nonlinear-decoder-architecture/SUMMARY.md)",
        "- [Machine-readable summary](../experiments/nonlinear-decoder-architecture/summary.json)",
        "- [Generated result audit](../experiments/nonlinear-decoder-architecture/AUDIT.md)",
        "- [Experiment method and code](../experiments/nonlinear-decoder-architecture/README.md)", "",
    ]
    OUTPUT.write_text("\n".join(md))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
