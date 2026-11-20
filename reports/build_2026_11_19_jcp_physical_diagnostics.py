"""Publish registered, measured JCP diagnostics without running scientific models.

Sources are immutable once registered. Numerical prose/table values are derived
here from JSON; plot builders remain in their independently archived lanes.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports/2026-11-19-jcp-physical-diagnostics-assets"
REPORT = ROOT / "reports/2026-11-19-jcp-physical-diagnostics-and-confirmation.md"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def freeze(entry, folder):
    source = ROOT / entry["source"]
    raw = source.read_bytes()
    destination = ASSETS / folder / (entry["id"] + source.suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError(f"Registered artifact changed: {entry['id']}")
    destination.write_bytes(raw)
    return destination, {**entry, "sha256": sha(raw), "bytes": len(raw),
                         "frozen": str(destination.relative_to(ROOT))}


def head_values(data):
    cfg = data["config"]
    threshold = min(cfg["solver"]["gtols"])
    output = []
    for setting in cfg["test_settings"]:
        for arm in ("head_common", "span_common", "span_native"):
            selected = [(case["index"], run) for case in data["cases"]
                        for run in case["runs"] if run["arm"] == arm
                        and run["rows"] == setting["rows"]
                        and run["dt"] == setting["dt"]
                        and run["gtol"] == threshold]
            if not selected:
                continue
            row = {**setting, "arm": arm, "case_count": len(selected), "cases": []}
            for case_id, run in selected:
                metrics = run["metrics"][1:]
                row["cases"].append({"case": case_id,
                    "field_max": max(m["relative_error"] for m in metrics),
                    "aligned_max": max(m["shape_aligned_relative_error"] for m in metrics),
                    "phase_max_domain_fraction": max(float(np.linalg.norm(m["phase_displacement"])) for m in metrics)})
            for key in ("field_max", "aligned_max", "phase_max_domain_fraction"):
                values = [c[key] for c in row["cases"]]
                row[key + "_median"] = float(np.median(values))
                row[key + "_worst"] = max(values)
            output.append(row)
    return output


def width_values(data):
    groups = {}
    if "arms" in data and "case_count" in data:
        for arm in data["arms"].values():
            row = {"width": arm["R"], "tests": arm["M"], "cases": []}
            groups[(arm["R"], arm["M"])] = row
            for case in arm["cases"]:
                row["cases"].append({"case": case["case"], "qualified": case["qualified"],
                    **{name + "_max": max(case[name]) for name in
                       ("error", "gradient_error", "upwind_advection_error", "laplacian_error")}})
    for key, case in data.get("cases", {}).items():
        width = int(key.split("-", 1)[0][1:])
        for count, arm in case["arms"].items():
            pair = (width, int(count))
            row = groups.setdefault(pair, {"width": width, "tests": int(count), "cases": []})
            row["cases"].append({"case": key,
                "qualified": arm["numerically_eligible"],
                **{name + "_max": max(arm[name]) for name in
                   ("error", "gradient_error", "upwind_advection_error", "laplacian_error")}})
    for row in groups.values():
        row["case_count"] = len(row["cases"])
        row["qualified_count"] = sum(c["qualified"] for c in row["cases"])
        for name in ("error", "gradient_error", "upwind_advection_error", "laplacian_error"):
            values = [c[name + "_max"] for c in row["cases"]]
            row[name + "_median"] = float(np.median(values))
            row[name + "_worst"] = max(values)
    return [groups[key] for key in sorted(groups)]


def cost_values(data, qualifications=None):
    values = []
    for name, group in data["confirmation"].items():
        cases = group["cases"]
        for case in cases.values():
            assert np.isclose(np.median(case["repetitions_ms"]), case["median_ms"], rtol=1e-12)
        metrics = [c["nominal_metrics"] for c in (qualifications or []) if c["subject"] == name]
        errors = [m["evolved_max_error"] for m in metrics] if metrics else [c["evolved_max_error"] for c in cases.values()]
        values.append({"subject": name, "case_count": len(cases),
            "metric_case_count": len(errors),
            "median_case_ms": float(np.median([c["median_ms"] for c in cases.values()])),
            "median_error": float(np.median(errors)), "worst_error": max(errors),
            "complete": group["complete"], "failure_count": group["failure_count"],
            "numerical_pass": data.get("numerical_pass", {}).get(name)})
    return values


def sparse_values(data):
    return [{"rule": name, "samples": r["sample_count"], "positive_weights": r["actual_positive_weights"],
             "stencil_sites": r["unique_stencil_interior_sites"], "dense_sites": r["dense_interior_sites"],
             "residual": r["probe_residual_max"], "action": r["probe_action_max"],
             "initial_fit": r["cold_start_max"], "fit_termination": r["fit"]["message"],
             "admitted": r["admitted"], "gates": r["gates"]} for name, r in data["rules"].items()]


def table(kind, values):
    if kind == "head":
        lines = ["| Tests / step | Arm | Cases | Field median / worst (%) | Aligned median / worst (%) | Phase median / worst (% domain length) |",
                 "|---|---|---:|---:|---:|---:|"]
        for r in values:
            cells = [f"{r['rows']} / {r['dt']:g}", r["arm"], str(r["case_count"])]
            for k in ("field_max", "aligned_max", "phase_max_domain_fraction"):
                cells.append(f"{100*r[k+'_median']:.6f} / {100*r[k+'_worst']:.6f}")
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    if kind == "width":
        lines = ["| Width / tests | Qualified / cases | Field median / worst (%) | Gradient median / worst (%) | Advection median / worst (%) | Laplacian median / worst (%) |",
                 "|---|---:|---:|---:|---:|---:|"]
        for r in values:
            cells = [f"{r['width']} / {r['tests']}", f"{r['qualified_count']} / {r['case_count']}"]
            for k in ("error", "gradient_error", "upwind_advection_error", "laplacian_error"):
                cells.append(f"{100*r[k+'_median']:.6f} / {100*r[k+'_worst']:.6f}")
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    if kind == "cost":
        lines = ["| Frozen subject | Error / valid timed cases | Complete / numerical pass | Full-cohort median time (ms) | Nominal median / worst error (%) | Failed timed invocations |",
                 "|---|---:|---|---:|---:|---:|"]
        for r in values:
            timing = f"{r['median_case_ms']:.6f}" if r['complete'] else "not reported (incomplete)"
            lines.append(f"| {r['subject']} | {r['metric_case_count']} / {r['case_count']} | {r['complete']} / {r['numerical_pass']} | {timing} | {100*r['median_error']:.6f} / {100*r['worst_error']:.6f} | {r['failure_count']} |")
        return "\n".join(lines)
    if kind == "sparse":
        lines = ["| Tested rule | Positive sites / stencil sites | Residual discrepancy | Jacobian-action relative discrepancy | Initial-fit discrepancy | Admitted |",
                 "|---|---:|---:|---:|---:|---|"]
        for r in values:
            lines.append(f"| {r['rule']} | {r['positive_weights']} / {r['stencil_sites']} | {r['residual']:.6g} | {r['action']:.6g} | {r['initial_fit']:.6g} | {r['admitted']} |")
        return "\n".join(lines)
    return ""


def prose_values(loaded, derived):
    result = {}
    for source, label in (("development-head", "DEV"), ("fresh-head", "FRESH")):
        rows = derived[source]
        h = next(r for r in rows if r["arm"] == "head_common")
        s = next(r for r in rows if r["arm"] == "span_common")
        for key, metric in (("FIELD_RATIO", "field_max"), ("SHAPE_RATIO", "aligned_max")):
            result[label + "_" + key] = f"{np.median([b[metric]/a[metric] for a,b in zip(h['cases'],s['cases'])]):.3f}"
        result[label + "_COUNT"] = str(h["case_count"])
        result[label + "_PHASE_WINS"] = str(sum(a["phase_max_domain_fraction"] < b["phase_max_domain_fraction"] for a,b in zip(h['cases'],s['cases'])))
    audit = loaded["phase-audit"]
    result["IC_FAILURE_COUNT"] = str(len(audit["initial_fit_failed_case_indices"]))
    result["MIN_CONCENTRATION"] = f"{audit['minimum_fresh_concentration_all_runs']:.4f}"
    result["PROBE_COUNT"] = str(audit["probe_counts"]["total"])
    width = derived["fresh-width"]
    for r in width:
        label = "BASELINE" if r["width"] == min(v["width"] for v in width) else "SELECTED"
        result["WIDTH_"+label+"_WORST"] = f"{100*r['error_worst']:.3f}"
        result["WIDTH_"+label+"_MEDIAN"] = f"{100*r['error_median']:.3f}"
    result["WIDTH_CASES"] = str(loaded["fresh-width"]["case_count"])
    selected = next(r for r in derived["development-width"] if r["width"] == 512 and r["tests"] == 8192)
    result["DEV_LAPLACIAN_WORST"] = f"{100*selected['laplacian_error_worst']:.2f}"
    result["DEV_FIELD_WORST"] = f"{100*selected['error_worst']:.3f}"
    result["MIN_LAPLACIAN_NORM"] = f"{loaded['derivative-summary']['denominator_ranges']['laplacian_error']['minimum']:.3f}"
    result["GPU_JOBS"] = str(loaded["scheduler"]["job_count"])
    result["GPU_SECONDS"] = str(loaded["scheduler"]["total_allocated_gpu_seconds"])
    result["GPU_HOURS"] = f"{loaded['scheduler']['total_allocated_gpu_seconds']/3600:.6f}"
    result["TIMED_ROWS"] = str(loaded["coordinator-audit"]["timed_invocation_count"])
    result["FAILED_TIMED_ROWS"] = str(loaded["coordinator-audit"]["failed_timed_invocations"])
    return result


def main():
    registry = json.loads((ASSETS / "sources.json").read_text())
    derived, manifests, replacements, loaded = {}, [], {}, {}
    for entry in registry["sources"]:
        dest, provenance = freeze(entry, "evidence")
        manifests.append(provenance)
        if dest.suffix == ".json":
            loaded[entry["id"]] = json.loads(dest.read_text())
        replacements["SOURCE_" + entry["id"]] = str(dest)
    for entry in registry["sources"]:
        kind = entry.get("kind")
        if kind in ("head", "width", "cost", "sparse"):
            data = loaded[entry["id"]]
            if kind == "cost":
                values = cost_values(data, loaded.get(entry.get("qualification_id")))
            else:
                values = {"head": head_values, "width": width_values, "sparse": sparse_values}[kind](data)
            if entry.get("select"):
                values = [r for r in values if all(r[k] == v for k,v in entry["select"].items())]
            derived[entry["id"]] = values
            replacements["TABLE_" + entry["id"]] = table(kind, values)
    for entry in registry.get("figures", []):
        dest, provenance = freeze(entry, "figures")
        manifests.append(provenance)
        replacements["FIGURE_" + entry["id"]] = str(dest)
    for key, value in registry.get("text", {}).items():
        replacements[key] = value
    replacements.update(prose_values(loaded, derived))
    write_json(ASSETS / "measured-values.json", derived)
    text = (ASSETS / "report-template.md").read_text()
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", value)
    unfilled = re.findall(r"\{\{[^{}]+\}\}", text)
    if unfilled:
        raise ValueError(f"Unfilled template fields: {unfilled}")
    REPORT.write_text(text)
    links = re.findall(r"\]\((?:<)?(/[^)>]+)(?:>)?\)", text)
    for link in links:
        path = re.sub(r":\d+$", "", link)
        assert Path(path).exists(), path
    write_json(ASSETS / "verification.json", {"status": "passed", "sources": manifests,
        "report_sha256": sha(REPORT.read_bytes()), "local_link_occurrences": len(links),
        "numeric_values_sha256": sha((ASSETS / "measured-values.json").read_bytes()),
        "scope": "registered-source hashes, raw repetition medians, generated tables and link existence; scientific qualification is separately linked"})
    print(json.dumps({"report": str(REPORT), "registered_artifacts": len(manifests), "local_links": len(links)}))


if __name__ == "__main__":
    main()
