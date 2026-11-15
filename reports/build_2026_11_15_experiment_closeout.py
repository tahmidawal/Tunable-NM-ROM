"""Generate operator and sparse-solver measurements from frozen lane JSONs."""
import json
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(path.read_text())


def operator_section(trees):
    root = trees / "2026-11-12-operator-scaling/experiments/operator-scaling"
    freeze = load(root / "closeout/selection.json")
    audit = load(root / "closeout/audit.json")
    threshold = audit["target_median_percent"]
    rows = {}
    for path in sorted((root / "inputs/test").glob("*.json")):
        data = load(path)
        item = data["L256"]
        errors = [max(frames) * 100 for frames in item["per_case_per_time"]]
        timings = [duration for case in item["timing_repetitions_ms"] for duration in case]
        assert len(errors) == audit["test_cases"] and len(timings) == 5 * len(errors)
        rows[data["key"]] = median(errors), max(errors), median(timings)
    assert set(rows) == set(freeze["runs"])
    lines = [
        "## Held-out neural operators",
        "",
        f"Job `{audit['job']}` evaluated {len(rows)} frozen checkpoints on {audit['test_cases']} Burgers cases. "
        f"{sum(row[0] <= threshold for row in rows.values())} met the {threshold:g}% test-median target. "
        "Networks ran in the registered float32 precision with TF32 disabled. Errors were scored in float64.",
        "",
        "| Validation-selected operator | Median error | Worst error | Median query time |",
        "|---|---:|---:|---:|",
    ]
    for family, decision in freeze["selection"].items():
        assert decision["selected"] == min(decision["candidates"], key=decision["candidates"].get)
        med, worst, ms = rows[decision["selected"]]
        lines.append(f"| {family} | {med:.5f}% | {worst:.5f}% | {ms:.3f} ms |")
    lines += [
        "",
        "Errors are per-case maxima over evolved output times. Query time is the median across cases and repetitions "
        "on one A100 80 GB. The scored output came from the final timed invocation for each case. Reporting includes "
        "surviving best checkpoints from interrupted training. Training budgets differ between the selected models. "
        "The historical ROM ran in a different allocation.",
        "",
        "[All checkpoints and training budgets](https://github.com/tahmidawal/Tunable-NM-ROM/blob/"
        "eb699ee396b23fdf3ba44f101221f14bfe033c83/experiments/operator-scaling/reports/tables.md) · "
        "[Numerical audit](https://github.com/tahmidawal/Tunable-NM-ROM/blob/"
        "eb699ee396b23fdf3ba44f101221f14bfe033c83/experiments/operator-scaling/closeout/audit.json)",
    ]
    return lines


def sparse_section(trees):
    root = trees / "2026-11-13-sparse-solvers/experiments/sparse-solvers"
    paper = load(root / "ours_burgers.json")
    results = root / "results"
    rom = load(results / "ours/same_gpu.json")
    lines = [
        "## Sparse GPU solver and ROM measurements",
        "",
        "AmgX cases are timed full Burgers trajectories on one H200. Each row uses the fastest converged, "
        "finite setting whose observed worst error passes the historical accurate ROM threshold. Selection "
        "uses all repetitions. The paired time below uses the first timed repetition of each case, the invocation "
        "that supplied the scored field. The thresholds come from a larger cohort.",
        "",
        "| Mesh | Cases | Worst-error threshold | Median error | Worst error | Paired median time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mesh in (256, 512, 1024):
        run = load(results / f"step/h200/amgx_b2d_L{mesh}.json")
        threshold = paper["burgers2d"][str(mesh)]["accurate"]["worst_percent"]
        eligible = [s for s in run["settings"].values() if s["status"] == "ok" and s["finite"] and
                    s["unconverged_steps_total"] == 0 and 100 * s["worst_error"] <= threshold]
        chosen = min(eligible, key=lambda s: s["time_s"]["median"])
        eset = [100 * case["error"] for case in chosen["per_case"]]
        ms = median(1000 * case["times"][0] for case in chosen["per_case"])
        lines.append(f"| {mesh}² | {len(eset)} | {threshold:.4f}% | {median(eset):.4f}% | {max(eset):.4f}% | {ms:.2f} ms |")
    lines += [
        "",
        "The ROM panels ran in separate allocations. Each admitted timing below passed its lane’s timing controls "
        "and reproduced the paper’s errors. The smaller Poisson panel failed the timing-order check.",
        "",
        "| Problem | Mesh | Cases | Accurate / fast median time | Accurate / fast worst error |",
        "|---|---|---:|---:|---:|",
    ]
    for problem, mesh in [("poisson2d", 1024), ("poisson2d", 2048), ("burgers2d", 2048)]:
        arms = rom[problem][str(mesh)]
        a, f = arms["accurate"], arms["fast"]
        assert a["timing_usable"] and f["timing_usable"]
        name = {"poisson2d":"Poisson", "burgers2d":"Burgers"}[problem]
        lines.append(f"| {name} | {mesh}² | {a['cases']} | {a['ms']:.3f} / {f['ms']:.3f} ms | "
                     f"{a['worst_percent']:.4f}% / {f['worst_percent']:.4f}% |")
    lines += [
        "",
        "[Sparse collection and archive verification](https://github.com/tahmidawal/Tunable-NM-ROM/tree/"
        "4b3a989e9b2926a7e5ff341a5cdb06f3b3f06465/experiments/sparse-solvers/ledger/closeout-2026-11-15) · "
        "[Complete AmgX settings](https://github.com/tahmidawal/Tunable-NM-ROM/blob/"
        "4b3a989e9b2926a7e5ff341a5cdb06f3b3f06465/experiments/sparse-solvers/RESULTS-burgers-stepper.md)",
    ]
    return lines


def main():
    root = Path(__file__).resolve().parents[1]
    trees = root / "worktrees"
    report = [
        "# Closing the Burgers quadrature, sparse-solver and neural-operator experiments",
        "",
        "The sparse-solver and held-out operator measurements are final. "
        "The separate sealed Burgers quadrature confirmation is pending.",
        "",
        *operator_section(trees),
        "",
        *sparse_section(trees),
        "",
        "## Glossary",
        "",
        "- **Mesh** gives the spatial resolution in one direction. A square denotes a two-dimensional grid.",
        "- **FNO and U-Net** are neural operators. An FNO uses Fourier transforms to mix spatial features. "
        "A U-Net uses an encoder and a decoder with connections between corresponding resolutions.",
        "- **Checkpoint** means saved trained model weights. The validation cases selected one checkpoint "
        "per family before the held-out test cases were scored.",
        "- **ROM** means reduced-order model. Accurate and fast denote the previously selected solver configurations.",
        "- **AmgX** is the NVIDIA sparse linear-solver library used inside the implicit Burgers solver. "
        "The trajectory time includes all time steps.",
        "- **Median and worst error** summarize the middle and largest case error. The error of an individual "
        "case is the maximum relative spatial error over its evolved output times. The threshold is an "
        "existing accuracy target. A worst-error threshold applies to the largest case error in that panel.",
        "- **Median query time** measures elapsed time for one solve. A paired time uses the call that produced "
        "the scored solution. The case count gives the number of physical inputs in each comparison.",
        "- **Float32, float64 and TF32** describe floating-point precision. Disabling TF32 keeps the operator's "
        "float32 matrix products from using reduced-precision TF32 inputs.",
        "",
    ]
    (root/"reports/2026-11-15-experiment-closeout.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
