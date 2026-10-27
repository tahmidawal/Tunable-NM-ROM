"""Build the same-sample Burgers reconstruction report from immutable exports.

This consumes completed validation exports only. It never selects settings, fits
models, or reads sealed cases. Casewise failures remain in every denominator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median


GATE = {"median_ratio_max": .8, "resolved_win_fraction_min": .75}
COHORT = {2: ("val32", 32, {32, 64, 128}, 256), 3: ("val64", 64, {16, 32}, 64)}
CONFIG_COUNTS = {2: {32: 77, 64: 77, 128: 145}, 3: {16: 93, 32: 93}}
LABEL = {
    "ordinary": "Development-selected interpolation",
    "bank_base": "Fine-trained decoder",
    "bank_coarse": "Coarse-trained decoder",
    "pod_base": "Fine-data POD",
    "pod_coarse": "Coarse-data POD",
    "bank": "Learned decoder",
    "pod_matched": "POD, matched observations",
    "pod_stronger": "POD, extra fine samples",
}
CLASSICAL = {"linear": "Linear", "cubic_notaknot": "Cubic", "sine_full": "Full sine"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def normalize(path, expected_dimension):
    data = json.loads(Path(path).read_text())
    assert data["phase"] == "val", "Report accepts validation exports only"
    dim = data["dimension"]
    assert dim == expected_dimension
    cohort, count, meshes, fine_mesh = COHORT[dim]
    rows = []
    for r in data["rows"] if dim == 2 else data["cases"]:
        if dim == 2:
            times = r["timings"]["host_query_seconds"]
            row = dict(dimension=dim, Lc=r["Lc"], Lf=r["Lf"], category=r["group"],
                       error=r["max_error"], eps=r["epsilon_reference"],
                       amplification=r["amplification"], strict=r["strict_coarse_control"],
                       seconds=median(times) if times else None,
                       fit_seconds=median(r["timings"]["fit_seconds"]) if times else None,
                       decode_seconds=median(r["timings"]["decode_seconds"]) if times else None)
        else:
            row = dict(dimension=dim, Lc=r["coarse_intervals"], Lf=r["fine_intervals"],
                       category=r["descriptor"]["category"], error=r["error"],
                       eps=r["reference_eps"], amplification=r["physical_amplification"],
                       strict=False, seconds=(r["timing_median"] or {}).get("host_query_seconds"),
                       fit_seconds=(r["timing_median"] or {}).get("fit_seconds"),
                       decode_seconds=(r["timing_median"] or {}).get("decode_seconds"))
        row.update({k: r[k] for k in ("source", "cohort", "case", "config", "status", "selected", "input_eligible")})
        row["finite"] = row["status"] == "complete" and finite(row["error"])
        if row["status"] == "complete":
            error_times = r["error_t"] if dim == 2 else r["error_times"][1:]
            assert len(error_times) == 5 and all(finite(e) and e >= 0 for e in error_times)
            assert row["finite"] and row["error"] >= 0
            assert math.isclose(row["error"], max(error_times), rel_tol=2e-14, abs_tol=1e-16)
        assert finite(row["eps"]) and row["eps"] >= 0
        assert type(row["case"]) is int
        assert row["source"] in ("A", "B") and row["cohort"] == cohort
        assert row["Lc"] in meshes and row["Lf"] == fine_mesh
        rows.append(row)
    assert rows and all(r["dimension"] == dim for r in rows)
    keys = [(r["Lc"], r["source"], r["case"], r["config"]) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicate case/configuration entries"
    cohorts = {r["cohort"] for r in rows}
    assert len(cohorts) == 1
    universe = set(range(count))
    assert {r["case"] for r in rows} == universe
    assert {(r["Lc"], r["source"]) for r in rows} == {(lc, s) for lc in meshes for s in ("A", "B")}
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["Lc"], row["source"], row["config"]].append(row)
    for rr in by_config.values():
        assert {r["case"] for r in rr} == universe, "Incomplete declared cohort coverage"
        assert len({r["selected"] for r in rr}) == 1
        assert len({r["category"] for r in rr}) == 1
    for lc in meshes:
        assert len({r["config"] for r in rows if r["Lc"] == lc and r["source"] == "A"}) == CONFIG_COUNTS[dim][lc]
        assert {r["config"] for r in rows if r["Lc"] == lc and r["source"] == "A"} == {
            r["config"] for r in rows if r["Lc"] == lc and r["source"] == "B"}
    return data, rows


def summarize(rows):
    valid = [r for r in rows if r["finite"]]
    errors = [r["error"] for r in valid]
    timings = {key: [r[key] for r in valid if finite(r[key])]
               for key in ("seconds", "fit_seconds", "decode_seconds")}
    return dict(cases=len(rows), finite_cases=len(valid), failures=len(rows)-len(valid),
                input_admitted=sum(r["input_eligible"] for r in rows),
                median=median(errors) if errors else None, worst=max(errors) if errors else None,
                above_one_percent=sum(e > .01 for e in errors),
                **{key: median(values) if len(values) == len(rows) else None
                   for key, values in timings.items()},
                eps_max=max(r["eps"] for r in rows),
                amplification=max((r["amplification"] for r in valid
                                   if finite(r["amplification"])), default=None))


def compare(method, baseline):
    a = {r["case"]: r for r in method}
    b = {r["case"]: r for r in baseline}
    assert a.keys() == b.keys() and a
    sources = {r["source"] for r in method+baseline}
    assert len(sources) == 1
    source = next(iter(sources))
    pairs = []
    for case in sorted(a):
        x, y = a[case], b[case]
        assert x["eps"] == y["eps"], "Methods must share reference qualification"
        ok = x["finite"] and y["finite"]
        gap = y["error"]-x["error"] if ok else None
        width = 1e-14 if source == "A" else 2*x["eps"]
        pairs.append(dict(case=case, finite=ok, win=bool(ok and gap > width),
                          loss=bool(ok and gap < -width), difference=gap,
                          input_admitted=x["input_eligible"] and y["input_eligible"]))
    full = all(p["finite"] for p in pairs)
    sm, sb = summarize(method), summarize(baseline)
    ratio = sm["median"]/sb["median"] if full and sb["median"] > 0 else None
    wins = sum(p["win"] for p in pairs)
    losses = sum(p["loss"] for p in pairs)
    method_worst_lower = max(max(0., r["error"]-r["eps"]) for r in method) if full else None
    baseline_worst_upper = max(r["error"]+r["eps"] for r in baseline) if full else None
    worst_ok = bool(full and method_worst_lower <= baseline_worst_upper)
    return dict(method=method[0]["config"], baseline=baseline[0]["config"],
                source=source, cases=len(pairs), complete_pairs=sum(p["finite"] for p in pairs),
                reference_eps_max=sm['eps_max'],
                wins=wins, losses=losses, ties=sum(p["finite"] for p in pairs)-wins-losses,
                failed_pairs=sum(not p["finite"] for p in pairs), ratio_of_medians=ratio,
                no_resolved_worst_deterioration=worst_ok if source == "B" else None,
                method_worst_lower=method_worst_lower, baseline_worst_upper=baseline_worst_upper,
                accuracy_gate=bool(source == "B" and ratio is not None
                                  and ratio <= GATE["median_ratio_max"]
                                  and wins/len(pairs) >= GATE["resolved_win_fraction_min"]
                                  and worst_ok),
                input_admitted=sum(p["input_admitted"] for p in pairs), pairs=pairs)


def fmt(value, scale=1):
    return "—" if value is None else f"{value*scale:.4g}"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "| " + " | ".join("---" for _ in header) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def diagnostic_evidence(roots, chosen):
    """Read immutable setup/oracle records without recomputing or refitting fields."""
    consumed, cache = {}, {}
    def read(path):
        if path not in cache:
            cache[path] = json.loads(path.read_text())
            consumed[str(path.resolve())] = sha(path)
        return cache[path]
    conditioning, setup_rows, oracle_rows = [], [], []
    for dim, root in roots.items():
        cohort, count, meshes, _ = COHORT[dim]
        plans_by_mesh = {}
        for lc in sorted(meshes):
            path = root/f'plans/L{lc}/plans.json' if dim == 2 else root/f'L{lc}/plans/setup.json'
            setup = read(path)
            plans_by_mesh[lc] = ({r['name']:r for r in setup['plans']} if dim == 2 else setup['plans'])
            setup_rows.append([dim, lc, fmt(setup['prepare_seconds'] if dim == 2 else setup['total_seconds'])])
        for (rdim, lc, source, category), rows in sorted(chosen.items()):
            if rdim != dim or source != 'B' or category == 'ordinary':
                continue
            p = plans_by_mesh[lc][rows[0]['config']]
            spectrum = p['singular_values']
            retained = p['sampled_effective_rank']
            condition = spectrum[0]/spectrum[retained-1] if retained and spectrum[retained-1] > 0 else None
            conditioning.append([dim, lc, LABEL[category], p['original_rank'], p['physical_rank'], retained,
                                 fmt(condition), fmt(p['physical_amplification'])])
        # Primary-grid limits and the extra coarse-training-control grid. Other
        # meshes share the fine representation and remain in the raw archive.
        for lc in ([32, 128] if dim == 2 else [16]):
            per_label = defaultdict(list)
            for case in range(count):
                path = (root/f'oracle/L{lc}/{cohort}_c{case:03d}.json' if dim == 2
                        else root/f'L{lc}/{cohort}/c{case:03d}/oracle.json')
                for label, row in read(path).items():
                    values = row['error_t'] if dim == 2 else row['e_t'][1:]
                    assert len(values) == 5 and all(finite(e) and e >= 0 for e in values)
                    per_label[label].append(max(values))
            for label, values in sorted(per_label.items()):
                assert len(values) == count
                oracle_rows.append([dim, lc, f'`{label}`', fmt(median(values),100), fmt(max(values),100), count])
    return dict(conditioning=conditioning, setup=setup_rows, oracle=oracle_rows, consumed=consumed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--two-dimensional", type=Path, required=True)
    p.add_argument("--three-dimensional", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    inputs = [args.two_dimensional, args.three_dimensional]
    allrows, provenance = [], []
    roots = {}
    for dim, path in zip((2, 3), inputs):
        data, rows = normalize(path, dim)
        roots[dim] = path.parent
        allrows.extend(rows)
        provenance.append(dict(path=str(path.resolve()), sha256=sha(path),
                               dimension=data["dimension"], phase=data["phase"],
                               runtime=data.get("runtime", data.get("execution"))))
    groups = defaultdict(list)
    for r in allrows:
        groups[r["dimension"], r["Lc"], r["source"], r["config"]].append(r)
    chosen = {}
    for (dim, lc, source, config), rr in groups.items():
        if rr[0]["selected"]:
            key = dim, lc, source, rr[0]["category"]
            assert key not in chosen, "Multiple selected settings in one category"
            chosen[key] = rr
    comparisons = []
    for (dim, lc, source, category), rr in sorted(chosen.items()):
        ordinary = chosen.get((dim, lc, source, "ordinary"))
        if category != "ordinary" and ordinary:
            comparisons.append(dict(dimension=dim, Lc=lc, category=category,
                                    **compare(rr, ordinary)))
    primary = [c for c in comparisons if c["source"] == "B"
               and c["Lc"] == {2: 32, 3: 16}[c["dimension"]]
               and c["category"] in ("bank_base", "bank")]
    primary_by_dim = {c["dimension"]: c for c in primary}
    evidence = diagnostic_evidence(roots, chosen)
    sentences = []
    for dim in sorted({r["dimension"] for r in allrows}):
        c = primary_by_dim.get(dim)
        if c is None:
            sentences.append(f"The {dim}D primary comparison has no available frozen selection.")
        else:
            verdict = "passes" if c["accuracy_gate"] else "does not pass"
            sentences.append(f"The {dim}D decoder {verdict} the prespecified accuracy gate "
                             f"against development-selected interpolation.")
    lines = ["# Burgers reconstruction from the same coarse samples", "",
             "This report compares interpolation and learned-basis reconstruction using identical "
             "coarse field samples. Numerical tables are generated from the completed validation exports; "
             "scientific acceptance also requires the accompanying independent result audit.", "",
             " ".join(sentences), "", "## What is being compared", "",
             "Source A supplies exact samples of the stored fine-reference array. Source B supplies "
             "an actual coarse numerical solution. Every method receives the same field at the same "
             "physical time, coordinates and zero boundary values. There is no rollout or subsequent "
             "PDE correction. Sealed and out-of-distribution cohorts remain closed.", "",
             "The decoder and POD have training-derived information about plausible fields. "
             "Interpolation has grid geometry and boundary information. Finer decoder evaluation "
             "does not add observations: any improvement comes from the learned representation "
             "and its regularization. Direct fitting to a fine reference is an oracle diagnostic, "
             "excluded from the reconstruction comparison.", "",
             "Errors use the volume-weighted fine-grid norm, divided by the fixed initial-reference "
             "norm. Each case is scored by its maximum error across the evolved observation times:", "",
             r"$$e_i=\max_{t>0}\frac{\|\widehat u_i(t)-u_{i,\mathrm{ref}}(t)\|_{h_f}}{\|u_{i,\mathrm{ref}}(0)\|_{h_f}}.$$", "",
             "## Actual coarse solutions: selected settings", ""]
    score_rows = []
    for (dim, lc, source, category), rr in sorted(chosen.items()):
        if source != "B":
            continue
        s = summarize(rr)
        score_rows.append([dim, f"{lc} → {rr[0]['Lf']}", LABEL[category],
                           f"`{rr[0]['config']}`", fmt(s["median"],100), fmt(s["worst"],100),
                           s["above_one_percent"], f"{s['failures']}/{s['cases']}",
                           f"{s['input_admitted']}/{s['cases']}", fmt(s["fit_seconds"],1000),
                           fmt(s["decode_seconds"],1000), fmt(s["seconds"],1000)])
    lines += [table(["Dimension", "Intervals per axis", "Method family", "Frozen configuration",
                     "Median error (%)", "Worst error (%)", "Above 1%", "Failed / all", "Input admitted / all",
                     "Fit (ms)", "Decode (ms)", "Warm query (ms)"], score_rows), "",
              "When any fit fails, displayed error summaries describe finite cases only; the failure "
              "column preserves the full denominator, and the full-cohort improvement ratio is unavailable. "
              "An unadmitted coarse input still permits a same-input accuracy comparison, but its "
              "error cannot be attributed cleanly to spatial discretization alone.", "",
              "## Paired comparisons with selected interpolation", ""]
    comparison_rows = []
    for c in comparisons:
        comparison_rows.append([c["dimension"], c["Lc"], c["source"], LABEL[c["category"]],
                                fmt(c["ratio_of_medians"]), fmt(c['reference_eps_max'],100), c["wins"], c["ties"], c["losses"],
                                f"{c['failed_pairs']}/{c['cases']}"])
    lines += [table(["Dimension", "Coarse intervals", "Source", "Method family", "Median error ratio",
                     "Max reference ε (%)", "Wins", "Ties", "Losses", "Failed pairs / all"], comparison_rows), "",
              "The ratio is method median error divided by interpolation median error; smaller is better. "
              "It is not the median of paired ratios. Source A wins describe completion of fixed numerical "
              "arrays, with a numerical tie tolerance. Source B wins exceed twice the stored reference "
              "uncertainty estimate. That estimate is a conditional numerical resolution rule, not a "
              "certified physical-error bound or a statistical confidence interval.", "",
              "## Fixed ordinary methods and coarse-training control", ""]
    controls = []
    for (dim, lc, source, config), rr in sorted(groups.items()):
        if config not in CLASSICAL and not rr[0]["strict"]:
            continue
        s = summarize(rr)
        label = CLASSICAL.get(config, LABEL[rr[0]["category"]] + " (fixed control)")
        controls.append([dim, lc, source, label, f"`{config}`", fmt(s["median"],100),
                         fmt(s["worst"],100), f"{s['failures']}/{s['cases']}"])
    lines += [table(["Dimension", "Coarse intervals", "Source", "Method", "Configuration",
                     "Median error (%)", "Worst error (%)", "Failed / all"], controls), "",
              "The coarse-training control uses its prespecified rank and no fitted ridge penalty. "
              "Other selected configurations use exposed development references; their selection "
              "therefore has fine-reference information. Both development and validation are exposed "
              "research cohorts, rather than sealed generalization tests.", "",
              "## Do the samples determine the representation?", "",
              table(["Dimension", "Coarse intervals", "Selected B family", "Requested directions",
                     "Retained fine directions", "Sampled rank", "Retained-spectrum condition", "Amplification"],
                    evidence['conditioning']), "",
              "The fine-grid Gram cutoff can remove nearly dependent directions before fitting. "
              "A sampled rank below the retained fine rank leaves directions unidentified by the observations. "
              "Full sampled rank alone does not imply stability: the condition and the regularized map's "
              "physical amplification quantify sensitivity at different stages.", "",
              "## Fine-reference oracle limits — diagnostics only", "",
              table(["Dimension", "Coarse-grid plan", "Fine representation", "Oracle median error (%)",
                     "Oracle worst error (%)", "Cases"], evidence['oracle']), "",
              "These projections use the full fine target and are inadmissible as reconstruction methods. "
              "They diagnose whether a representation can express the target at all. A large oracle floor "
              "limits any coefficient fit; a small floor with poor coarse reconstruction points toward "
              "identification, regularization or input error. They do not measure prediction skill.", "",
              "## Qualifications and cost", "",
              "POD uses recipe-reproduced training data and an approximate randomized factorization. "
              "Recipe, parameter-table and observation-coordinate agreement do not establish byte identity "
              "with archived training arrays. The matched-observation spatial extension and the stronger "
              "comparator with extra fine values are distinct. These comparisons do not by themselves "
              "isolate a causal advantage of the network architecture.", "",
              "Warm query time is the median of per-case medians for independently fitted single fields, "
              "including common host transfers. It excludes reusable basis evaluation, factorization and "
              "compilation. Accuracy and timing come from the same calls, with raw repetitions retained. "
              "Compare costs only within the same job and GPU; cross-dimensional timings are not speedup comparisons.", "",
              table(["Dimension", "Coarse intervals", "Recorded library setup (s)"], evidence['setup']), "",
              "Library setup covers the entire tested configuration sweep in that validation job, rather "
              "than the cost of deploying one selected method. Fine-coordinate basis evaluation was cached "
              "during preparation. These measurements support steady repeated-query comparisons, not a "
              "claim about total cost for a single uncached reconstruction. Separate fit/decode medians "
              "need not sum to the median end-to-end time.", "",
              "No derivative-accuracy claim is made. Reference-level perturbations, coarse-time-step "
              "perturbations, coefficient observability, amplification and POD convergence diagnostics "
              "remain part of the archived evidence and independent acceptance audit.", "",
              "## Prespecified larger-campaign gate", "",
              f"The primary actual-coarse comparison requires a median-error ratio no larger than "
              f"{GATE['median_ratio_max']:g}, resolved wins in at least "
              f"{GATE['resolved_win_fraction_min']*100:g}% of all declared cases, and no reference-resolved "
              "deterioration of the cohort's worst error. Ties and failed pairs do not count as wins. "
              "This is an accuracy-screen gate; training-information, conditioning and input qualifications "
              "still constrain the scientific claim. Worst-error deterioration is resolved only when "
              "the method's lower uncertainty envelope exceeds interpolation's upper envelope:", "",
              r"$$\max_i\max(0,e_{i,\mathrm{method}}-\epsilon_i)>\max_i(e_{i,\mathrm{interp}}+\epsilon_i).$$", "",
              table(["Dimension", "Primary coarse intervals", "Accuracy gate", "Input admitted / all"],
                    [[dim, c["Lc"], "Pass" if c["accuracy_gate"] else "Does not pass",
                      f"{c['input_admitted']}/{c['cases']}"] for dim,c in sorted(primary_by_dim.items())]),
              "", "## Provenance", ""]
    for item in provenance:
        lines.append(f"- {item['dimension']}D validation export: `{item['path']}`; SHA-256 `{item['sha256']}`.")
    lines += ["", "The companion JSON contains normalized case records, full comparison pairs, "
              "runtime metadata and the gate constants used to generate these tables.", "",
              f"The [complete configuration table]({args.out.stem}.configurations.csv) retains every "
              "tested setting, including failed and unselected configurations. Error summaries in that "
              "table describe finite cases and must be read beside the failure count; blank timings "
              "mean that a complete timing measurement is unavailable.", "",
              "## Plain-language glossary", "",
              "- **Dimension / D:** number of spatial directions.",
              "- **Intervals per axis / coarse intervals:** number of mesh cells along each direction; "
              "the full grid includes both boundary nodes.",
              "- **Source A / B:** sampled fine-reference array / actual coarse numerical solution.",
              "- **Decoder / learned basis:** coordinate-dependent functions learned from training fields; "
              "only their coefficients are fitted to the supplied coarse field.",
              "- **POD:** proper orthogonal decomposition, a linear basis extracted from training snapshots.",
              "- **Frozen configuration:** rank, penalty and method chosen before validation was run.",
              "- **Rank / ridge:** number of retained basis directions / penalty that discourages unstable coefficients.",
              "- **Requested directions / retained fine directions / sampled rank:** initial basis size / "
              "directions kept after removing numerical dependence / directions distinguishable on the coarse grid.",
              "- **Retained-spectrum condition:** largest divided by smallest retained singular value of the "
              "weighted observation matrix; larger values indicate a more sensitive coefficient fit.",
              "- **Gram cutoff / library setup:** numerical rule for removing dependent basis directions / "
              "preparing all candidate reconstruction maps before querying them.",
              "- **Median error / worst error / above 1%:** middle case error / largest finite case error / "
              "count of finite cases whose error exceeds the stated threshold.",
              "- **Failed / all:** unusable numerical fits and the full declared cohort size; failures are never silently dropped.",
              "- **Input admitted:** coarse computation passed its fixed solver and time-step checks.",
              "- **Warm query:** fitting and evaluating one field after reusable setup and GPU warm-up.",
              "- **Wins / ties / losses:** paired comparisons using the source-specific resolution rule; failed pairs are separate.",
              "- **Median error ratio:** method cohort median divided by the interpolation cohort median.",
              "- **Full sine / cubic / linear:** boundary-compatible spectral interpolation / tensor cubic spline / tensor linear interpolation.",
              "- **Matched observations / extra fine samples:** training observations matching the bank's recorded sampling "
              "pattern / a deliberately stronger comparator with additional fine-grid values.",
              "- **Oracle:** a diagnostic fitted directly to fine targets, unavailable to an admissible reconstruction method.",
              "- **Observability / amplification:** whether coarse samples identify basis coefficients / how much an input "
              "perturbation can change the reconstructed field.",
              "- **Reference uncertainty:** estimated numerical uncertainty of the fine comparison field.",
              "- **Max reference ε:** largest case-specific reference uncertainty in the displayed cohort; "
              "paired decisions and worst-error envelopes still use each case's own value.",
              "- **Exposed / sealed / out-of-distribution:** already available for method development / reserved and unopened / "
              "outside the tested training-family conditions.",
              "- **PDE / rollout:** partial differential equation / advancing a state through time.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    args.out.with_suffix(".json").write_text(json.dumps(dict(
        provenance=provenance, gate=GATE, cases=allrows, comparisons=comparisons,
        primary=primary, diagnostic_evidence=evidence), indent=2, allow_nan=False)+"\n")
    configuration_rows = []
    for (dim, lc, source, config), rr in sorted(groups.items()):
        configuration_rows.append(dict(dimension=dim, coarse_intervals=lc, fine_intervals=rr[0]['Lf'],
                                       source=source, configuration=config, category=rr[0]['category'],
                                       selected=rr[0]['selected'], strict_control=rr[0]['strict'], **summarize(rr)))
    with args.out.with_name(args.out.stem+'.configurations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(configuration_rows[0]))
        writer.writeheader(); writer.writerows(configuration_rows)
    print(args.out)


if __name__ == "__main__":
    main()
