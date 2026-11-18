"""Build the campaign's numerical report from explicitly registered local records.

No solver, model import, network access, or selection of scientific cases occurs.
The coordinator registers completed/retained records in the source list. Frozen
copies are immutable; a changed source must receive a new identifier.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["svg.hashsalt"] = "jcp-controlled-experiments"

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports/2026-11-18-jcp-controlled-experiments-assets"
REPORT = ROOT / "reports/2026-11-18-jcp-controlled-experiments.md"
BEGIN = "<!-- GENERATED EXPERIMENT RESULTS BEGIN -->"
END = "<!-- GENERATED EXPERIMENT RESULTS END -->"
SUMMARY_BEGIN = "<!-- GENERATED EXPERIMENT SUMMARY BEGIN -->"
SUMMARY_END = "<!-- GENERATED EXPERIMENT SUMMARY END -->"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def freeze(entry):
    source = ROOT / entry["source"]
    raw = source.read_bytes()
    dest = ASSETS / "evidence" / (entry["id"] + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.read_bytes() != raw:
        raise ValueError(f"Registered evidence changed: {entry['id']}")
    dest.write_bytes(raw)
    return json.loads(raw), {**entry, "sha256": digest(raw), "frozen": str(dest)}


def head_rows(data):
    gtol = min(data["config"]["solver"]["gtols"])
    rows = []
    for setting in data["config"]["test_settings"]:
        paired = []
        for case in data["cases"]:
            runs = {r["arm"]: r for r in case["runs"]
                    if r["rows"] == setting["rows"] and r["dt"] == setting["dt"]
                    and r["gtol"] == gtol}
            if not {"head_common", "span_common"} <= runs.keys():
                continue
            paired.append({"case": case["index"],
                           "head": runs["head_common"]["evolved_worst"],
                           "span": runs["span_common"]["evolved_worst"]})
        if not paired:
            continue
        h = np.array([p["head"] for p in paired])
        s = np.array([p["span"] for p in paired])
        rows.append({**setting, "cases": paired, "head_median_percent": 100*np.median(h),
                     "head_worst_percent": 100*np.max(h), "span_median_percent": 100*np.median(s),
                     "span_worst_percent": 100*np.max(s), "head_lower_count": int(np.sum(h < s)),
                     "paired_ratio_median": float(np.median(s/h))})
    return rows


def head_plot(entry, data):
    cfg = data["config"]
    settings = cfg["test_settings"]
    gtol = min(cfg["solver"]["gtols"])
    fig, axes = plt.subplots(1, len(settings), figsize=(12.5, 3.7), squeeze=False)
    for ax, setting in zip(axes[0], settings):
        for arm, label, color, style in (
                ("head_common", "Head, common initial state", "#157c75", "-"),
                ("span_common", "Span, common initial state", "#ad601e", "-"),
                ("span_native", "Span, native projection", "#777385", "--")):
            runs = [r for c in data["cases"] for r in c["runs"]
                    if r["arm"] == arm and r["rows"] == setting["rows"]
                    and r["dt"] == setting["dt"] and r["gtol"] == gtol]
            if not runs:
                continue
            errors = 100*np.array([[m["relative_error"] for m in r["metrics"]] for r in runs])
            times = np.arange(errors.shape[1])*setting["dt"]
            ax.plot(times, np.median(errors, axis=0), color=color, ls=style, label=label)
            ax.fill_between(times, errors.min(0), errors.max(0), color=color, alpha=.12)
        ax.set(title=f"M = {setting['rows']}; Δt = {setting['dt']:g}",
               xlabel="Nondimensional time", ylabel="Velocity error (% of initial field norm)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    status = "Numerical gates passed" if data["qualification"].get("passed") else "PROVISIONAL: numerical gates failed"
    fig.suptitle(f"Measured matched head/span trajectories • {len(data['cases'])} development cases • {status}", fontsize=11)
    fig.text(.5, .015, f"Lines: pointwise medians. Shading: case range, not uncertainty intervals. Source: {entry['id']}, job {data['job_id']}.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .94])
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / f"{entry['id']}-trajectories.{suffix}", dpi=170, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    return ASSETS / f"{entry['id']}-trajectories.png"


def weak_rows(data):
    result = []
    for case_id, case in data["cases"].items():
        for m, arm in case["arms"].items():
            t = arm["levels"]["tighter"]
            result.append({"case": case_id, "M": int(m),
                           "max_error_percent": 100*max(t["max_error_per_repetition"]),
                           "numerically_eligible": arm["numerically_eligible"],
                           "stationarity": t["max_stationarity"],
                           "independent_shell_residual": arm["independent_scaled_shell_max"],
                           "tight_tighter_difference": arm["tight_tighter_difference"]})
    return result


def head_probes(entry, data, manifest):
    """Freeze and plot the independent one-step, same-previous-state probes."""
    records = []
    source_root = (ROOT / entry["source"]).parent
    for case in data["cases"]:
        for setting in data["config"]["test_settings"]:
            name = f"reset_m{setting['rows']}_dt{setting['dt']:g}.json"
            source = source_root / f"case_{case['index']:03d}" / name
            aux = {"id": f"{entry['id']}-case{case['index']:03d}-{name[:-5]}",
                   "kind": "head_probe", "source": str(source.relative_to(ROOT))}
            probes, provenance = freeze(aux)
            manifest.append(provenance)
            for p in probes:
                row = {"case": case["index"], **setting, "step": p["step"],
                                "stationary": p["span_result"]["reason"] == "stationarity",
                                "residual_ratio": p["span_result"]["residual_norm"]/p["head_seed_residual_norm"],
                                "physical_error_ratio": p["span_physical_error"]/p["head_physical_error"]}
                geo = p.get("shared_state_geometry")
                if geo:
                    row.update(least_visible_outside_head_norm_fraction=geo["least_visible_span_outside_head_fraction"][0],
                               actual_departure_outside_head_norm_fraction=geo["actual_step_linearized_outside_head_fraction"],
                               physical_head_rank=geo["head_rank"], physical_span_rank=geo["span_rank"])
                records.append(row)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    colors = ["#446880", "#777385", "#157c75"]
    for setting, color in zip(data["config"]["test_settings"], colors):
        ss = [p for p in records if p["rows"] == setting["rows"] and p["dt"] == setting["dt"]]
        valid = [p for p in ss if p["stationary"]]
        invalid = [p for p in ss if not p["stationary"]]
        ax.scatter([p["residual_ratio"] for p in valid], [p["physical_error_ratio"] for p in valid],
                   s=24, alpha=.7, color=color, label=f"M={setting['rows']}, Δt={setting['dt']:g}")
        if invalid:
            ax.scatter([p["residual_ratio"] for p in invalid], [p["physical_error_ratio"] for p in invalid],
                       s=35, marker="x", color=color, label="Unconverged probe")
    ax.axvline(1, color=".5", ls="--", lw=1)
    ax.axhline(1, color=".5", ls="--", lw=1)
    ax.set(xlabel="Span / head minimized weak-residual norm",
           ylabel="Span / head physical error", xlim=(0, 1.08), ylim=(0, None))
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc="upper right")
    status = "numerical gates passed" if data["qualification"].get("passed") else "PROVISIONAL"
    ax.set_title(f"Same previous field: lower residual can give greater physical error\n{entry['id']} • {status}", fontsize=11)
    fig.text(.5, .015, "Each dot is one restarted time-step solve; steps within a case are correlated.\nBelow 1 horizontally: lower residual. Above 1 vertically: greater physical error.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, 1])
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / f"{entry['id']}-residual-versus-error.{suffix}", dpi=170, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    return records, ASSETS / f"{entry['id']}-residual-versus-error.png"


def weak_plot(entry, data):
    case_rows = {c: sorted((v for v in weak_rows(data) if v["case"] == c), key=lambda v: v["M"])
                 for c in data["cases"]}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    matrices = [[], [], []]
    for case, rows in case_rows.items():
        ms = [v["M"] for v in rows]
        errors = [v["max_error_percent"] for v in rows]
        independent = [v["independent_shell_residual"]/rows[0]["independent_shell_residual"] for v in rows]
        fixed = data["cases"][case]["fixed_state_spectra"][-1]
        visibility = [fixed["spectra"][str(m)]["physical_min"] for m in ms]
        for ax, values, array in zip(axes, (errors, independent, visibility), matrices):
            array.append(values)
            ax.plot(ms, values, color=".7", lw=.8, alpha=.75)
    for ax, values in zip(axes, matrices):
        values = np.asarray(values)
        ax.plot(ms, np.median(values, axis=0), color="#157c75", marker="o", label="Case median")
        ax.fill_between(ms, values.min(0), values.max(0), color="#157c75", alpha=.1, label="Case range")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ms, labels=[str(m) for m in ms])
        ax.set_xlabel("Scalar weak-test rows M")
        ax.grid(alpha=.18)
    axes[0].set(title="Physical prediction", ylabel="Maximum saved-time error (% initial norm)")
    axes[1].set(title="Unminimized equations", ylabel="Independent residual / smallest-M residual")
    axes[2].set(title="Visibility at a common projected state", ylabel="Smallest mass-normalized singular value")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    qualified = all(v["numerically_eligible"] for v in weak_rows(data))
    status = "all numerical gates passed" if qualified else "PROVISIONAL: some numerical gates failed"
    fig.suptitle(f"Measured Burgers weak-test enrichment • {len(case_rows)} cases • {status}", fontsize=11)
    fig.text(.5, .015, f"{entry['id']}: same-grid/same-step discrete reference. Error includes initial time. Shading is case range.\nSingular-value monotonicity for nested unchanged rows is algebraic; physical improvement must be measured separately.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, .94])
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / f"{entry['id']}-test-enrichment.{suffix}", dpi=170, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    return ASSETS / f"{entry['id']}-test-enrichment.png"


def cost_rows(data):
    rows = []
    for target, groups in data["comparisons"].items():
        t = data["confirmation"][target]
        for group, comp in groups.items():
            name = comp.get("comparator")
            c = data["confirmation"].get(name, {})
            ratios = []
            if comp["qualified"]:
                ratios = [c["cases"][k]["median_ms"]/v["median_ms"] for k, v in t["cases"].items()]
            rows.append({"target": target, "group": group, "comparator": name,
                         "case_count": len(t["case_ids"]), "qualified": comp["qualified"],
                         "target_median_ms": t["median_case_ms"],
                         "target_median_error_percent": 100*t["median_case_evolved_max_error"],
                         "target_worst_error_percent": 100*t["worst_evolved_error"],
                         "comparator_median_ms": c.get("median_case_ms"),
                         "comparator_over_target": comp.get("comparator_over_target"),
                         "individual_case_ratios": ratios,
                         "paired_ratio_median": float(np.median(ratios)) if ratios else None,
                         "target_faster_case_count": sum(v > 1 for v in ratios),
                         "target_timing_outlier_count": t["outlier_count_tukey"],
                         "comparator_timing_outlier_count": c.get("outlier_count_tukey"),
                         "reason": comp.get("reason", "")})
    return rows


def projection_screen_plot(entry, audit):
    records = audit["records"]
    widths = sorted({v["width"] for v in records})
    cases = sorted({v["case"] for v in records})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))
    tables = []
    for width in widths:
        rr = [v for v in records if v["width"] == width]
        tables.append({"width": width, "case_count": len(rr),
                       "median_max_percent": 100*np.median([v["maximum_all_times_error"] for v in rr]),
                       "worst_max_percent": 100*max(v["maximum_all_times_error"] for v in rr),
                       "worst_initial_percent": 100*max(v["initial_error"] for v in rr),
                       "admitted_to_dynamics": width in audit["admitted_widths"]})
    for case in cases:
        rr = sorted((v for v in records if v["case"] == case), key=lambda v: v["width"])
        axes[0].plot(widths, [100*v["maximum_all_times_error"] for v in rr], color=".75", lw=.8)
    axes[0].plot(widths, [v["worst_max_percent"] for v in tables], "o-", color="#ad601e", label="Worst case")
    axes[0].plot(widths, [v["median_max_percent"] for v in tables], "o-", color="#157c75", label="Case median")
    axes[0].set(xscale="log", yscale="log", xticks=widths, xticklabels=[str(w) for w in widths],
                xlabel="Frozen basis functions R", ylabel="Maximum projection error (% initial norm)",
                title="Representation capacity over all saved times")
    axes[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].legend(fontsize=8)
    dt = audit["fixed_contract"]["dt"]
    for width, color in zip(widths, ("#ad601e", "#667c87", "#157c75")):
        errors = 100*np.array([v["errors"] for v in records if v["width"] == width])
        axes[1].plot(dt*np.arange(errors.shape[1]), np.max(errors, axis=0), color=color, label=f"R={width}")
    axes[1].set(xlabel="Nondimensional time", ylabel="Worst-case projection error (% initial norm)",
                title="Best linear fits of the reference fields", yscale="log")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle(f"Frozen-basis projection screen • {len(cases)} development cases • independently checked", fontsize=11)
    fig.text(.5, .015, "Projection uses each reference field at every time. These are approximation floors, not predicted PDE trajectories.\nThe checkpoint and ordered basis are unchanged; no training or fresh cases are involved.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .12, 1, .94])
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / f"{entry['id']}-projection-floors.{suffix}", dpi=170, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    return tables, ASSETS / f"{entry['id']}-projection-floors.png"


def enrichment_plot(entry, data, baseline):
    ids = {c["index"] for c in data["cases"]}
    dt = data["config"]["test_settings"][0]["dt"]
    records = []
    for source, label in ((baseline, "baseline"), (data, "enrichment")):
        for case in source["cases"]:
            if case["index"] not in ids:
                continue
            for run in case["runs"]:
                if run["dt"] != dt or run["gtol"] != min(source["config"]["solver"]["gtols"]):
                    continue
                if run["arm"] in ("head_common", "span_common"):
                    records.append({"source": label, "case": case["index"], "rows": run["rows"],
                                    "arm": run["arm"], "error_percent": 100*run["evolved_worst"]})
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ms = sorted({r["rows"] for r in records})
    for arm, color, name in (("head_common", "#157c75", "Head"), ("span_common", "#ad601e", "Span")):
        for index, case in enumerate(sorted(ids)):
            rows = sorted((v for v in records if v["arm"] == arm and v["case"] == case), key=lambda v: v["rows"])
            ax.plot([v["rows"] for v in rows], [v["error_percent"] for v in rows], color=color,
                    ls="-" if index == 0 else "--", marker="o" if index == 0 else "s", label=f"{name}, case {case}")
    ax.set(xscale="log", yscale="log", xticks=ms, xticklabels=[str(m) for m in ms],
           xlabel="Scalar weak-test rows M (fixed time step)", ylabel="Maximum trajectory error\n(% of initial field norm)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=.18)
    status = "numerical gates passed" if data["qualification"].get("passed") else "PROVISIONAL"
    ax.set_title(f"Richer tests on the fixed pilot cases • Δt={dt:g} • {status}", fontsize=11)
    fig.text(.5, .015, "Same frozen bank/head, common initial reconstruction and discrete reference. Finite enrichment; no asymptotic convergence claim.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .06, 1, 1])
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / f"{entry['id']}-richer-tests.{suffix}", dpi=170, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    return records, ASSETS / f"{entry['id']}-richer-tests.png"


def cost_plot(entry, data, selection, provenance):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"classical": "#667c87", "pod": "#82798e", "target": "#157c75"}
    display_cap_percent = 100
    def finite_metrics(v):
        return all(isinstance(v.get(k), (int, float)) and np.isfinite(v[k])
                   for k in ("worst_evolved_error", "median_case_ms"))
    overflow = sum(v["complete"] and finite_metrics(v) and 100*v["worst_evolved_error"] > display_cap_percent for v in selection.values())
    failed = sum(not v["complete"] or not finite_metrics(v) for v in selection.values())
    for family, marker in (("classical", "."), ("pod", "s"), ("target", "*")):
        chosen = {k: v for k, v in selection.items() if v["family"] == family and v["complete"] and finite_metrics(v)
                  and 100*v["worst_evolved_error"] <= display_cap_percent}
        axes[0].scatter([v["median_case_ms"] for v in chosen.values()],
                        [100*v["worst_evolved_error"] for v in chosen.values()],
                        marker=marker, color=colors[family], s=95 if family == "target" else 28, label=family)
        if family == "target":
            for name, v in chosen.items():
                axes[0].annotate(name.replace("learned_", ""), (v["median_case_ms"], 100*v["worst_evolved_error"]),
                                 xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[0].set(xscale="log", yscale="log", title="Finite development candidate set",
                xlabel="Median complete-query time (ms)", ylabel="Worst error at evolved requested outputs\n(% of initial field norm)")
    axes[0].legend(fontsize=8)
    axes[0].text(.02, .025, f"Outside display: {overflow} above {display_cap_percent}% error; {failed} failed / nonfinite.\nAll retained in the raw selection table.",
                 transform=axes[0].transAxes, fontsize=8, va="bottom")
    qualified = [r for r in cost_rows(data) if r["qualified"]]
    labels = []
    for index, row in enumerate(qualified):
        ratios = row["individual_case_ratios"]
        median = float(np.median(ratios))
        axes[1].plot([min(ratios), max(ratios)], [index, index], color=".6", lw=3)
        axes[1].scatter([median], [index], color="#157c75", zorder=3)
        target_label = {"learned_head": "Head", "learned_span64": "Span"}.get(row["target"], row["target"])
        group_label = {"any_mesh_classical": "any-mesh FOM", "same_mesh_classical": "same-mesh FOM", "pod": "POD"}.get(row["group"], row["group"])
        labels.append(target_label+" vs "+group_label)
    axes[1].axvline(1, color=".5", ls="--", lw=1)
    axes[1].set(yticks=range(len(labels)), yticklabels=labels,
                xlabel="Comparator / target time (above 1 favors target)", title="Qualified confirmation speed ratios")
    for ax in axes:
        ax.grid(alpha=.17)
    fig.suptitle(f"{entry['id']}: instrumented complete-query measurements\n{provenance['gpu'].split(',')[0]}", fontsize=11)
    fig.text(.5, .015, "Left: displayed candidate subset with exclusions stated. Right: median paired case ratios and case range.\nPrimary accuracy: five evolved fields; initial error retained separately. Six fields returned, native fitting and transfers included.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .10, 1, .93])
    for suffix in ("png", "svg"):
        fig.savefig(ASSETS / f"{entry['id']}-complete-query.{suffix}", dpi=170, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    return ASSETS / f"{entry['id']}-complete-query.png"


def main():
    entries = json.loads((ASSETS / "sources.json").read_text())
    loaded = []
    manifest = []
    values = {}
    for entry in entries:
        data, provenance = freeze(entry)
        loaded.append((entry, data, provenance))
        manifest.append(provenance)
    chunks = ["## Measured development results", "",
              "These are new measured development runs. A numerical failure remains visible even when a physical-error comparison is favorable. The earlier optimistic schematics are separate hypothetical artifacts.", ""]
    heads = [(e, d, p) for e, d, p in loaded if e["kind"] == "head"]
    if heads:
        chunks += ["### Matched head/span dynamics", "",
                   "| Retained run / job | Cases | Numerical qualification | Failed gates |",
                   "|---|---:|---|---|"]
        for e, d, p in heads:
            q = d["qualification"]
            failures = [k for k, v in q.items() if v is False and k != "passed"]
            chunks.append(f"| [{e['id']}]({p['frozen']}) / {d['job_id']} | {len(d['cases'])} | {'Pass' if q.get('passed') else 'Provisional: failed'} | {', '.join(failures) or 'None'} |")
            values[e["id"]] = head_rows(d)
        # Display the latest registered run, not the most favorable outcome.
        e, d, p = heads[-1]
        title = "Numerically qualified development comparison" if d["qualification"].get("passed") else "Provisional comparison; numerical qualification failed"
        chunks += ["", f"**{title} — {e['id']}.** Errors below are maxima over each evolved trajectory, normalized by its original physical initial field. Both primary arms have identical reconstructed initial states. Median and worst are across the same cases.", "",
                   "| Scalar weak rows M | Time step | Head median / worst error (%) | Span median / worst error (%) | Cases with lower head error |",
                   "|---:|---:|---:|---:|---:|"]
        for v in values[e["id"]]:
            chunks.append(f"| {v['rows']} | {v['dt']:g} | {v['head_median_percent']:.6f} / {v['head_worst_percent']:.6f} | {v['span_median_percent']:.6f} / {v['span_worst_percent']:.6f} | {v['head_lower_count']} / {len(v['cases'])} |")
        plot = head_plot(e, d)
        chunks += ["", f"![Measured matched head/span trajectories]({plot})", "",
                   ("The admitted comparison isolates the head constraint under the registered numerical settings." if d["qualification"].get("passed") else "The comparison is designed to isolate the head constraint, but the failed numerical gates currently prevent admitting that conclusion."),
                   "Its scope excludes a universal stability guarantee, fresh-test generalization, a global manifold distance, and a complete-query speedup.", ""]
        probes, probe_plot = head_probes(e, d, manifest)
        values[e["id"]+"-probes"] = probes
        good = [p for p in probes if p["stationary"]]
        paradox = sum(p["residual_ratio"] <= 1 and p["physical_error_ratio"] > 1 for p in good)
        chunks += [f"In the same-previous-state control, **{paradox} of {len(good)} stationary probes** reduce the minimized weak residual while increasing physical error; {len(probes)-len(good)} additional probes do not meet stationarity. Each span solve starts from the head's next coefficients and uses the identical previous field and objective. These are correlated step diagnostics across {len(d['cases'])} cases, not independent statistical samples.", "",
                   f"![Same-previous-state residual and physical-error comparison]({probe_plot})", ""]
        geometric = [v for v in probes if "least_visible_outside_head_norm_fraction" in v]
        if geometric:
            outside = np.median([v["least_visible_outside_head_norm_fraction"] for v in geometric])
            departure = np.median([v["actual_departure_outside_head_norm_fraction"] for v in geometric])
            ranks = sorted(set((v["physical_head_rank"], v["physical_span_rank"]) for v in geometric))
            chunks += [f"At the same reached head states, the median physical-norm fraction of the least visible span direction outside the head-plus-frame tangent is **{outside:.6f}**. The corresponding median fraction of the actual span departure, evaluated with the local linearized field map, is **{departure:.6f}**. Retained physical (head, span) ranks are {ranks}. These are norm fractions of sampled local directions, not fractions of cases or directions, energy fractions, global manifold distances, or uniform stability constants. The run's qualification status above applies to this diagnostic too.", ""]
    enriched = [(e, d, p) for e, d, p in loaded if e["kind"] == "head_enrichment"]
    if enriched:
        chunks += ["### Richer-test check for the head mechanism", "",
                   "This follow-up retains the fixed pilot cases, model, common initial reconstruction, time step and convergence gates. It tests whether additional residual information closes the span/head gap. The original development-panel result remains separate.", "",
                   "| Run / job | Cases | Numerical qualification | Rows |",
                   "|---|---:|---|---|"]
        for e, d, p in enriched:
            chunks.append(f"| [{e['id']}]({p['frozen']}) / {d['job_id']} | {len(d['cases'])} | {'Pass' if d['qualification'].get('passed') else 'Provisional: failed'} | {', '.join(str(s['rows']) for s in d['config']['test_settings'])} |")
            records, plot = enrichment_plot(e, d, heads[-1][1])
            values[e["id"]] = records
            chunks += ["", f"![Measured richer-test head/span comparison]({plot})", ""]
    weaks = [(e, d, p) for e, d, p in loaded if e["kind"] == "weak"]
    if weaks:
        chunks += ["### Weak-test visibility", "",
                   "The Burgers diagnostic model evolves arbitrary coefficients in the linear span of a frozen learned spatial basis; there are no frame variables. Every panel uses the same-grid, same-time-step discrete Burgers reference. No continuum-error claim is made. Maximum field error includes the initial saved time. Spectral visibility and physical accuracy are separate diagnostics.", "",
                   "| Retained run | Cases | Numerically qualified case/row settings |",
                   "|---|---:|---:|"]
        for e, d, p in weaks:
            rows = weak_rows(d)
            values[e["id"]] = rows
            chunks.append(f"| [{e['id']}]({p['frozen']}) | {len(d['cases'])} | {sum(v['numerically_eligible'] for v in rows)} / {len(rows)} |")
        e, d, p = weaks[-1]
        rows = values[e["id"]]
        chunks += ["", "Latest registered panel:", "",
                   "| Weak rows M | Median / worst maximum field error (%) | Numerically qualified cases |",
                   "|---:|---:|---:|"]
        for m in sorted({v["M"] for v in rows}):
            rr = [v for v in rows if v["M"] == m]
            ee = [v["max_error_percent"] for v in rr]
            chunks.append(f"| {m} | {np.median(ee):.6f} / {max(ee):.6f} | {sum(v['numerically_eligible'] for v in rr)} / {len(rr)} |")
        plot = weak_plot(e, d)
        chunks += ["", f"![Measured weak-test enrichment]({plot})", ""]
        for extra, detail, ep in loaded:
            if extra["kind"] == "weak_detail":
                max_m = max(v["M"] for v in detail)
                final = [v for v in detail if v["M"] == max_m]
                initial_count = sum(v["maximum_error_time"] == 0 for v in final)
                min_m = min(v["M"] for v in detail)
                original = [v for v in detail if v["M"] == min_m]
                chunks += [f"The largest total error remains limited by the initial projection. At M={max_m}, {initial_count} of {len(final)} cases attain their maximum at initialization. The worst actual distance from the reference's mass projection decreases from {100*max(v['evolution_distance_max'] for v in original):.6f}% to {100*max(v['evolution_distance_max'] for v in final):.6f}%. This distance is computed from the difference of fields, not by subtracting error norms. [Casewise source table]({ep['frozen']}).", ""]
            if extra["kind"] == "weak_counterexamples":
                comparisons = detail["comparisons"]
                increased = [v for v in comparisons if v["increased_time_count"] > 0]
                maximum = max(v["maximum_increase_in_reported_relative_error"] for v in comparisons)
                chunks += [f"**Retained counterexample:** {len(increased)} of {len(comparisons)} adjacent case/test-space transitions increase error at some times, even though their maximum-over-time errors do not worsen. The largest pointwise increase is {100*maximum:.6f} percentage points. This is a separately labeled post-run diagnostic; it precludes pointwise error-dominance claims. [All comparisons]({ep['frozen']}).", ""]
        chunks += ["", "Adding unchanged rows cannot decrease the smallest singular value at an identical state. That algebraic property alone does not demonstrate improved physical prediction.", ""]
    screens = [(e, d, p) for e, d, p in loaded if e["kind"] == "weak_projection_screen"]
    for e, d, p in screens:
        assert d["W0_raw_audit_passed"] and d["R128_reproduction_passed"]
        source = Path(d["source_output"]) / "summary.json"
        raw, rp = freeze({"id": e["id"]+"-raw", "kind": "weak_projection_raw", "source": str(source.relative_to(ROOT))})
        assert rp["sha256"] == d["summary_sha256"]
        manifest.append(rp)
        rows, plot = projection_screen_plot(e, d)
        values[e["id"]] = rows
        chunks += ["### Frozen-width follow-up: projection screen", "",
                   "This prospectively bounded follow-up targets the initial projection floor diagnosed above. It takes longer prefixes of the same frozen ordered bank and uses all original development cases. Each curve fits the reference at each saved time; it measures representation capacity and does not predict a trajectory. The physical operator, reference, parameters and error denominator are unchanged.", "",
                   "| Basis width R | Median / worst all-time projection error (%) | Worst initial projection error (%) | Admission to equation solves |",
                   "|---:|---:|---:|---|"]
        for row in rows:
            status = "Baseline" if row["width"] == min(r["width"] for r in rows) else "Pass" if row["admitted_to_dynamics"] else "Fail"
            chunks.append(f"| {row['width']} | {row['median_max_percent']:.6f} / {row['worst_max_percent']:.6f} | {row['worst_initial_percent']:.6f} | {status} |")
        chunks += ["", f"[Projection audit and admission]({p['frozen']}) bind the results to the saved model, basis ordering, references and numerical settings. All registered rank, projection and original-baseline reproduction checks pass. A prefix enters the PDE study only if both worst initial and worst all-time projection errors are at most 80% of baseline. No training was performed.", "",
                   f"![Measured frozen-bank projection floors]({plot})", ""]
    costs = [(e, d, p) for e, d, p in loaded if e["kind"] == "cost"]
    if costs:
        chunks += ["### Complete-query cost", "",
                   "These are instrumented queries from the original host-resident physical input to six complete host-resident fields, including native initial fitting, transfers, evolution and recovery. Primary accuracy is the maximum error over the five evolved requested fields; initial error is retained separately. This differs from both the denser mechanism-history grid and the Burgers metric that includes initialization. Candidate selection and independent timing repetitions use exposed development cases within each allocation. Eligibility requires both the median and worst casewise evolved error no greater than the target; it does not require every individual case to match. A finite candidate search does not certify the fastest possible comparator.", "",
                   "| Run / target | Cases | Comparator | Target time (ms) | Comparator time (ms) | Comparator / target | Qualification |",
                   "|---|---:|---|---:|---:|---:|---|"]
        for e, d, p in costs:
            rows = cost_rows(d)
            values[e["id"]] = rows
            for row in rows:
                ct = f"{row['comparator_median_ms']:.4f}" if row["comparator_median_ms"] is not None else "—"
                ratio = f"{row['comparator_over_target']:.6f}" if row["qualified"] else "—"
                status = "Pass: same-allocation accuracy and timing" if row["qualified"] else row["reason"] or "Failed qualification"
                chunks.append(f"| [{e['id']}]({p['frozen']}) / {row['target']} | {row['case_count']} | {row['comparator'] or row['group']} | {row['target_median_ms']:.4f} | {ct} | {ratio} | {status} |")
            source = ROOT / e["source"]
            selection, sp = freeze({"id": e["id"]+"-selection", "kind": "cost_selection",
                                    "source": str((source.parent/"selection.json").relative_to(ROOT))})
            provenance, pp = freeze({"id": e["id"]+"-provenance", "kind": "cost_provenance",
                                     "source": str((source.parent.parent/"provenance.json").relative_to(ROOT))})
            manifest.extend((sp, pp))
            chunks += ["", f"{e['id']} used **{provenance['gpu'].split(',')[0]}**, job **{provenance['job_id']}**. Ratios above one favor the target; no times are compared across jobs. A missing eligible POD comparator is an unresolved match in this candidate set, not a POD speed defeat.", ""]
            chunks += ["The main table divides cohort median times. The following diagnostics pair case medians; outliers use the per-case Tukey rule on all retained confirmation repetitions, without deleting them.", "",
                       "| Target / comparator | Median paired time ratio | Cases with faster target | Timing outliers: target / comparator |",
                       "|---|---:|---:|---:|"]
            for row in rows:
                if row["qualified"]:
                    chunks.append(f"| {row['target']} / {row['comparator']} | {row['paired_ratio_median']:.6f} | {row['target_faster_case_count']} / {row['case_count']} | {row['target_timing_outlier_count']} / {row['comparator_timing_outlier_count']} |")
            chunks += [""]
        plot = cost_plot(e, d, selection, provenance)
        chunks += [f"![Measured complete-query accuracy and cost]({plot})", "",
                   "Cold measurements, where retained in the experiment lane, cover asset activation, compilation and the first query within an already initialized runtime. They retain the operating-system page cache and exclude runtime/context startup. They cannot be relabeled process-cold deployment costs.", ""]
    cost_failures = [(e, d, p) for e, d, p in loaded if e["kind"] == "cost_failure"]
    if cost_failures:
        chunks += ["Retained cost attempts that stopped before accepted timing:", "",
                   "| Run / job | Outcome | Allocated GPU-seconds |",
                   "|---|---|---:|"]
        for e, d, p in cost_failures:
            chunks.append(f"| [{e['id']}]({p['frozen']}) / {d['job_id']} | {d['failure']} | {d['allocated_gpu_seconds']} |")
        chunks += ["", "A scalar compiled replay subsequently tests the same targets and comparator grid with unchanged convergence thresholds. Its different GPU allocation prevents attributing any successful repair exclusively to the earlier batched arithmetic.", ""]
    chunks += ["### Reproducibility", "",
               f"The [builder]({Path(__file__).resolve()}) reads the [registered sources]({ASSETS/'sources.json'}), freezes the JSON evidence, and generates every numerical table and plot above. [Source hashes]({ASSETS/'source-manifest.json'}) and [derived values]({ASSETS/'measured-values.json'}) are saved separately. Full histories, configurations, numerical arrays, hardware logs and checksum manifests remain in the linked experiment lanes. No synthetic values enter these figures.", ""]
    for e, d, p in loaded:
        if e["kind"] == "environment_review":
            chunks += [f"The [environment review]({p['frozen']}) documents retained host CPU-affinity messages in job {d['job_id']}. Explicit GPU and numerical witnesses support accuracy-only admission; diagnostic elapsed times are not used for a speed claim. The message matches the [upstream XLA report]({d['source']}).", ""]
    block = BEGIN + "\n" + "\n".join(chunks) + "\n" + END
    report = REPORT.read_text()
    if BEGIN in report:
        before, tail = report.split(BEGIN, 1)
        _, after = tail.split(END, 1)
        report = before + block + after
    else:
        report = report.replace("## Glossary", block + "\n\n## Glossary")
    overview = ["These measurements support a constrained-dynamics explanation for the head and identify a separate representation limit in Burgers.", ""]
    if heads:
        e, d, _ = heads[-1]
        v = values[e["id"]][-1]
        overview += [f"- **Head mechanism ({'qualified' if d['qualification']['passed'] else 'provisional'}):** at the last registered setting, M={v['rows']} and $\\Delta t={v['dt']:g}$, median maximum evolved error is {v['head_median_percent']:.4f}% for the head versus {v['span_median_percent']:.4f}% for the span. The median paired span/head error ratio is {v['paired_ratio_median']:.2f}; {v['head_lower_count']}/{len(v['cases'])} cases favor the head."]
    if weaks:
        e, _, _ = weaks[-1]
        rr = values[e["id"]]
        first_m, last_m = min(v["M"] for v in rr), max(v["M"] for v in rr)
        first = [v["max_error_percent"] for v in rr if v["M"] == first_m]
        last = [v["max_error_percent"] for v in rr if v["M"] == last_m]
        overview += [f"- **Weak-test enrichment:** increasing M from {first_m} to {last_m} reduces median maximum saved-time error from {np.median(first):.4f}% to {np.median(last):.4f}%. Worst error remains {max(last):.4f}%, set by the initial projection. This is the original frozen-width baseline."]
    if costs:
        e, _, _ = costs[-1]
        rr = values[e["id"]]
        for target, name in (("learned_head", "Head"), ("learned_span64", "Span")):
            row = next((v for v in rr if v["target"] == target and v["group"] == "any_mesh_classical"), None)
            if row and row["qualified"]:
                overview += [f"- **{name} complete-query cost, latest {row['case_count']}-case panel:** selected accuracy-eligible any-mesh classical time / {name.lower()} time is {row['comparator_over_target']:.3f}. Above one favors the ROM. This is a finite development comparison, with both methods measured in the same allocation."]
    overview += ["", "The figures below contain measured values. All cohorts here are exposed development data; untouched confirmation and broader JCP claims remain separate work."]
    summary = SUMMARY_BEGIN + "\n" + "\n".join(overview) + "\n" + SUMMARY_END
    if SUMMARY_BEGIN in report:
        before, tail = report.split(SUMMARY_BEGIN, 1)
        _, after = tail.split(SUMMARY_END, 1)
        report = before + summary + after
    else:
        report = report.replace("## Separate ownership and initial resources", summary + "\n\n## Separate ownership and initial resources")
    report = report.replace("No new experimental result has yet been accepted; numerical tables will be generated from saved run records as the studies finish.", "New measured development runs and their qualification status are generated below from retained records; failed attempts remain visible.")
    REPORT.write_text(report)
    (ASSETS / "source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (ASSETS / "measured-values.json").write_text(json.dumps(values, indent=2)+"\n")
    print(f"Updated {REPORT}; registered {len(entries)} retained records.")


if __name__ == "__main__":
    main()
