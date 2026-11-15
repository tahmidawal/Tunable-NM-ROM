"""Generate operator and sparse-solver measurements from frozen lane JSONs."""
import hashlib
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


def sealed_section(trees):
    root = trees / "2026-11-15-jcp-sealed-closeout/experiments/jcp-sealed-closeout"
    setup = load(root/"F1.json")
    approved = load(root/"F1-REVIEW.json")
    decisions = load(root/"results/four-rule-decisions.json")
    dims = {2: (256, 512, 1024, 2048, 4096), 3: (65, 129, 257)}
    pretty = {"fib987":"Fibonacci lattice", "gauss45":"Tensor Gauss", "lat4096":"CBC lattice", "gl20":"Tensor Gauss"}
    point_counts = {}
    verification = load(root/"results/verification.json")
    assert verification["decisions"] == decisions
    selections = load(root/"selection.json")
    for d in dims:
        previous = next(iter(selections[str(d)].values()))
        path = trees/"2026-11-05-burgers-quadrature-plateau"/previous["old_panel"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == previous["old_result_sha256"]
        for row in json.loads(data)["rows"]:
            if row["arm"] in pretty:
                assert point_counts.setdefault(row["arm"],row["m"]) == row["m"]
    results = root/"results"
    panels = {d:[load(results/f"sealed{d}d{m}-decision.json") for m in meshes] for d,meshes in dims.items()}
    comp = {d:[load(results/f"sealed{d}d{m}-comparators.json") for m in meshes] for d,meshes in dims.items()}
    intro = [
        "## Sealed Burgers quadrature confirmation",
        "",
        f"The freeze committed the configurations for {setup['case_counts']['2']} two-dimensional and "
        f"{setup['case_counts']['3']} three-dimensional in-family cases before a single authorized opening. "
        f"The Git identifier of the accepted freeze is `{approved['freeze_commit'][:12]}`. All runs used "
        "float64 and the highest matrix-product precision.",
        "",
        "The primary reference refines both space and time (ST). The secondary reference refines space (S). "
        "A pass requires the selected rule and every larger tested rule in its family to keep both median and "
        "worst trajectory error within the registered uncertainty-adjusted margin of the dense reference rule "
        "on every mesh. The registered margin is five percent. A failure on one qualified mesh falsifies sufficiency.",
        "",
        "| Dimension | Frozen quadrature | Points | Primary ST across meshes | Secondary S across meshes | Eligible ST median error across meshes | Eligible ST worst error across meshes |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    def rate_range(values):
        return f"{min(values)*100:.4f}–{max(values)*100:.4f}%" if values else "unavailable"
    for rule in ("fib987","gauss45","lat4096","gl20"):
        d = 2 if rule in ("fib987","gauss45") else 3
        stats = [p["kinds"]["ST"].get("arm_stats",{}).get(rule,{}) for p in panels[d]]
        def rates(key):
            return rate_range([s[key] for s in stats if s.get(key) is not None])
        v = decisions[rule]
        def decision_cell(kind):
            checks = v[kind]['by_mesh']
            n_pass = sum(state=='PASS' for state in checks.values())
            return f"{v[kind]['decision']} ({n_pass}/{len(checks)} pass)"
        intro.append(f"| {d}D | {pretty[rule]} | {point_counts[rule]:,} | {decision_cell('ST')} | "
                     f"{decision_cell('S')} | {rates('T')} | {rates('W')} |")
    intro += ["", "| Dimension | Physical cases | Eligible ST / S | Meshes passing both lattice timing checks |",
              "|---|---:|---:|---:|"]
    for d in dims:
        pp = panels[d]
        total = setup['case_counts'][str(d)]
        counts = [[len(p['kinds'][k]['eligible_cases']) for k in ('ST','S')] for p in pp if 'eligible_cases' in p['kinds']['ST']]
        assert not counts or len(set(tuple(c) for c in counts))==1
        eligible = " / ".join(str(count) for count in counts[0]) if counts else "unavailable"
        key = "rom_lat|" + ("fib987" if d == 2 else "lat4096")
        q = sum(p['timing']['dual_block_subject_qualified'].get(key) is True for p in comp[d])
        intro.append(f"| {d}D | {total} | {eligible} | {q} / {len(dims[d])} |")
    intro += ["", "Native POD controls used their previously selected native meshes. Median and worst errors "
               "below use the eligible ST cases with finite fields. The valid cases column counts numerical "
               "terminations over the entire physical cohort.",
               "", "| Control | Valid cases | Eligible ST median error | Eligible ST worst error |",
               "|---|---:|---:|---:|"]
    for d in dims:
        native = load(results/f"sealed{d}d{dims[d][0]}-pod-and-comparators.json")
        pod = native['pod']
        err = pod['ST']['eligible']
        intro.append(f"| {d}D POD | {pod['valid_cases']} / {pod['meta']['n']} | {100*err['median']:.4f}% | "
                     f"{100*err['maximum']:.4f}% |")
    ranks = {d:load(results/f"native-{d}d-paired-ranks.json")["subjects"]["bank"] for d in dims}
    for d, rules in ((2, ("fib987", "gauss45")), (3, ("lat4096", "gl20"))):
        primary = [ranks[d][rule]["ST"]["both_valid_case_flags"] for rule in rules]
        if primary[0] == primary[1]:
            counts = primary[0]
            intro += ["", f"On the native {d}D mesh, POD had lower ST error than each selected quadrature rule "
                           f"on {counts.get('pod_lower',0)} cases, and each rule had lower error on "
                           f"{counts.get('other_lower',0)}."]
    intro += ["", "The paired comparisons require differences to exceed twice the case's reference uncertainty."]
    intro += ["", "The full-order pseudospectral solver with $N=32$ appears in both native panels. Each table "
               "entry uses the previously declared solver setting. The numerical convergence count covers all "
               "physical cases.",
               "", "| Native panel | Converged full-order solves | Eligible ST median error | Eligible ST worst error |",
               "|---|---:|---:|---:|"]
    for d in dims:
        ps = comp[d][0]['foms']['PS_N32']
        n = setup['case_counts'][str(d)]
        err = ps['ST']['eligible']
        intro.append(f"| {d}D | {n-ps['unconverged_count']} / {n} | {100*err['median']:.4f}% | "
                     f"{100*err['maximum']:.4f}% |")
    checks = verification["panels"].values()
    comparisons = sum(panel["witness_count"] for panel in checks)
    discrepancy = max(panel["max_witness_discrepancy"] for panel in checks)
    intro += ["", f"Independent scoring reproduced {comparisons:,} saved case–reference comparisons across the "
               f"mesh panels. The maximum relative discrepancy was {discrepancy:.2e}. "
               "Checks used saved model and full-solver fields on selected 2D cases and decoded "
               "all 3D model coefficients. The historical 3D protocol saved no full-solver fields for independent "
               "reconstruction. POD and quadrature ran in separate allocations. Custody used one shared research "
               "account. The out-of-family populations remain sealed."]
    intro += ["", "[Committed freeze](https://github.com/tahmidawal/Tunable-NM-ROM/blob/"
                   "138cc10a8f50caeca3791f0353a3ba7b60bb8b8e/experiments/jcp-sealed-closeout/F1.json) · "
                   "[Mesh-by-mesh decisions and controls](https://github.com/tahmidawal/Tunable-NM-ROM/tree/"
                   "61b52f6de54a37f21f9514eb69d97ea42a62f649/experiments/jcp-sealed-closeout/results)"]
    return intro


def main():
    root = Path(__file__).resolve().parents[1]
    trees = root / "worktrees"
    report = [
        "# Closing the Burgers quadrature, sparse-solver and neural-operator experiments",
        "",
        "All three experiments are final. The tables report the frozen Burgers quadrature decisions and their "
        "classical controls, the held-out operator errors, and the recovered sparse-solver measurements.",
        "",
        *sealed_section(trees),
        "",
        *operator_section(trees),
        "",
        *sparse_section(trees),
        "",
        "## Glossary",
        "",
        "- **Quadrature** approximates a spatial integral using weighted points. Fibonacci and component-by-component "
        "(CBC) lattice rules distribute points using structured sequences. Tensor Gauss combines one-dimensional "
        "Gauss rules across coordinates. The dense rule uses many points on the same reduced model.",
        "- **Burgers, POD and full-order solvers.** Burgers combines nonlinear transport and diffusion. Proper orthogonal "
        "decomposition (POD) builds reduced coordinates from snapshots. A full-order pseudospectral solver approximates "
        "the PDE in spectral modes without a trained reduced basis. $N$ sets its spatial resolution.",
        "- **ST and S** denote the space-and-time and space-only reference procedures. Each case and reference "
        "kind has an original numerical quality check and uncertainty estimate. Only individually eligible cases enter "
        "formal accuracy rankings. A mesh passes the lattice timing checks when timed outputs agree with the scored "
        "outputs and the registered timing stability tests pass.",
        "- **Mesh** gives the spatial resolution in one direction. A square denotes a two-dimensional grid.",
        "- **FNO and U-Net** are neural operators. An FNO uses Fourier transforms to mix spatial features. "
        "A U-Net uses an encoder and a decoder with connections between corresponding resolutions.",
        "- **Checkpoint** means saved trained model weights. The validation cases selected one checkpoint "
        "per family before the held-out test cases were scored.",
        "- **ROM** means reduced-order model. Accurate and fast denote the previously selected solver configurations.",
        "- **AmgX** is the NVIDIA sparse linear-solver library used inside the implicit Burgers solver. "
        "The trajectory time includes all time steps.",
        "- **Median and worst error** summarize the middle and largest case error. For a Burgers trajectory, the case "
        "error takes the maximum over the evolved output times. A stationary Poisson solve yields one spatial error. The threshold is an "
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
