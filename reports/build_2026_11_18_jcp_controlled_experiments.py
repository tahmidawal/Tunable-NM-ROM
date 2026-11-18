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

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports/2026-11-18-jcp-controlled-experiments-assets"
REPORT = ROOT / "reports/2026-11-18-jcp-controlled-experiments.md"
BEGIN = "<!-- GENERATED EXPERIMENT RESULTS BEGIN -->"
END = "<!-- GENERATED EXPERIMENT RESULTS END -->"


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
        fig.savefig(ASSETS / f"{entry['id']}-trajectories.{suffix}", dpi=170)
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
    weaks = [(e, d, p) for e, d, p in loaded if e["kind"] == "weak"]
    if weaks:
        chunks += ["### Weak-test visibility", "",
                   "Every panel uses the same-grid, same-time-step discrete Burgers reference. No continuum-error claim is made. Maximum field error includes the initial saved time. Spectral visibility and physical accuracy are separate diagnostics.", "",
                   "| Retained run | Case | M | Maximum field error (%) | Numerical qualification | Independent shell residual |",
                   "|---|---:|---:|---:|---|---:|"]
        for e, d, p in weaks:
            rows = weak_rows(d)
            values[e["id"]] = rows
            for v in rows:
                chunks.append(f"| [{e['id']}]({p['frozen']}) | {v['case']} | {v['M']} | {v['max_error_percent']:.6f} | {'Pass' if v['numerically_eligible'] else 'Provisional: failed'} | {v['independent_shell_residual']:.6g} |")
        chunks += ["", "Adding unchanged rows cannot decrease the smallest singular value at an identical state. That algebraic property alone does not demonstrate improved physical prediction.", ""]
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
    report = report.replace("No new experimental result has yet been accepted; numerical tables will be generated from saved run records as the studies finish.", "New measured development runs and their qualification status are generated below from retained records; failed attempts remain visible.")
    REPORT.write_text(report)
    (ASSETS / "source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (ASSETS / "measured-values.json").write_text(json.dumps(values, indent=2)+"\n")
    print(f"Updated {REPORT}; registered {len(entries)} retained records.")


if __name__ == "__main__":
    main()
