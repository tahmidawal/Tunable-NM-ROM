"""Read-only numerical audit of existing lane records; writes reports on main only.

This independently recomputes supported aggregates from invocation/per-case arrays.
It does not import lane generators, run JAX, or modify any experiment checkout.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/2026-09-19-handoff/codex-audits"
INPUTS: dict[str, str] = {}


def read(path: Path):
    blob = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
    return json.loads(blob)


def rows_of(obj):
    return obj["rows"] if isinstance(obj, dict) else obj


def equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-13)
    return a == b


def add(checks, row, expected, path, calculation):
    checks.append(dict(
        subject=row.get("subject", row.get("arm")),
        metric=row["metric"], mesh=row.get("mesh"),
        cohort=row.get("cohort"), job_id=str(row["job_id"]),
        reported=row["value"], recomputed=expected,
        passed=equal(row["value"], expected),
        source=str(path.relative_to(ROOT)), calculation=calculation,
    ))


def lane_root(name):
    return ROOT / f"worktrees/2026-09-17-{name}/experiments/{name}"


def audit_poisson():
    lane = lane_root("p-linear")
    summary = rows_of(read(lane / "reports/summary.json"))
    checks, jobs, provenance = [], {}, []
    for attempt in ("plin256", "plin1024b", "plhead1"):
        path = lane / f"artifacts/{attempt}/result.json"
        result = read(path)
        jobs[str(result["job_id"])] = (path, result)
    for row in summary:
        item = jobs.get(str(row["job_id"]))
        if not item:
            continue
        path, result = item
        inv = [x for x in result.get("invocations", []) if x["name"] == row["subject"]]
        if not inv:
            continue
        by_case = defaultdict(list)
        for x in inv:
            by_case[x["case"]].append(x)
        err = [max(x["same_grid_error"] for x in a) for a in by_case.values()]
        physical = [max(x["physical_error"] for x in a) for a in by_case.values()]
        values = {
            "worst_same_grid": (max(err), "max_case max_rep same_grid_error"),
            "median_same_grid": (statistics.median(err), "median_case max_rep same_grid_error"),
            "worst_physical": (max(physical), "max_case max_rep physical_error"),
            "median_total_ms": (1000 * statistics.median(x["total_seconds"] for x in inv), "1000 * median(invocations.total_seconds)"),
            "median_device_ms": (1000 * statistics.median(x.get("fused_device_seconds", x.get("solver_seconds", 0) + x.get("projection_init_seconds", 0)) for x in inv), "1000 * median(invocation device solver plus initialization seconds)"),
        }
        if row["metric"] in values:
            expected, calculation = values[row["metric"]]
            add(checks, row, expected, path, calculation)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get("source_sha256"):
            provenance.append(dict(subject=row["subject"], job_id=str(row["job_id"]), passed=digest.startswith(row["source_sha256"]), check="result digest prefix"))
    return checks, provenance


def audit_waves():
    lane = lane_root("w-ladder")
    summary = rows_of(read(lane / "reports/summary.json"))
    attempts, checks, provenance = {}, [], []
    for attempt in sorted({r["attempt"] for r in summary}):
        path = lane / f"artifacts/{attempt}/result.json"
        attempts[attempt] = (path, read(path))
    for row in summary:
        path, result = attempts[row["attempt"]]
        inv = [x for x in result["invocations"] if x["method"] == row["subject"]]
        if not inv:
            continue
        first = [x for x in inv if x["repetition"] == 0]
        values = {}
        for field, metric in (("energy_state", "energy_state"), ("displacement", "displacement"), ("velocity", "velocity")):
            arr = [max(x["same_grid_discrepancy"][field]["initial_normalized"]) for x in first]
            values[f"worst_{metric}"] = (max(arr), f"max_case max_time {field}.initial_normalized, repetition=0")
            values[f"median_{metric}"] = (statistics.median(arr), f"median_case max_time {field}.initial_normalized, repetition=0")
        values.update({
            "worst_t0_energy_state": (max(x["same_grid_discrepancy"]["energy_state"]["initial_normalized"][0] for x in first), "max_case energy_state.initial_normalized[0]"),
            "worst_evolved_energy_state": (max(max(x["same_grid_discrepancy"]["energy_state"]["initial_normalized"][1:]) for x in first), "max_case max_evolved_time energy_state.initial_normalized"),
            "median_gpu_ms": (1000 * statistics.median(x["seconds"]["complete_device_query"] for x in inv), "1000 * median complete_device_query seconds"),
            "median_complete_ms": (1000 * statistics.median(x["device_plus_output_transfer_seconds"] for x in inv), "1000 * median device_plus_output_transfer_seconds"),
            "median_evolution_ms": (1000 * statistics.median(x["seconds"]["evolution"] for x in inv), "1000 * median evolution seconds"),
        })
        if row["metric"] in values:
            expected, calculation = values[row["metric"]]
            add(checks, row, expected, path, calculation)
        provenance.append(dict(subject=row["subject"], job_id=str(row["job_id"]), passed=(str(result["provenance"]["job_id"]) == str(row["job_id"]) and hashlib.sha256(path.read_bytes()).hexdigest() == row["source_sha256"]), check="job id and full result digest"))
    return checks, provenance


def audit_operators():
    lane = lane_root("no-second")
    wt = lane.parents[1]
    summary = rows_of(read(lane / "reports/summary.json"))
    cache, checks, provenance = {}, [], []
    supported = {"worst_fixed_initial_error", "median_fixed_initial_error", "mean_fixed_initial_error"}
    for row in summary:
        if row["metric"] not in supported or row.get("operator") not in ("U-Net", "Transolver", "FNO") or "rung" in row:
            continue
        path = Path(row["source"])
        if not path.is_absolute():
            path = wt / path
        if path not in cache:
            cache[path] = read(path)
        data = cache[path]
        if row["cohort"] == "validation-32":
            model = data.get("arms", data.get("models", {})).get(row["arm"])
        elif row["cohort"] == "diagnosis-8":
            model = data.get("cohort", data.get("diagnosis_cohort", {})).get("models", {}).get(row["arm"])
        else:
            continue
        if not model or "per_time_errors" not in model:
            continue
        errors = [max(a) for a in model["per_time_errors"]]
        values = {"worst_fixed_initial_error": max(errors), "median_fixed_initial_error": statistics.median(errors), "mean_fixed_initial_error": statistics.mean(errors)}
        add(checks, row, values[row["metric"]], path, f"{row['metric'].split('_')[0]} over per-case maxima recomputed from per_time_errors")
        provenance.append(dict(subject=row["arm"], job_id=str(row["job_id"]), passed=(str(data["job_id"]) == str(row["job_id"]) and hashlib.sha256(path.read_bytes()).hexdigest() == row["source_sha256"]), check="job id and source audit digest"))
    return checks, provenance


def audit_lowvisc():
    lane = lane_root("b-lowvisc")
    summary = rows_of(read(lane / "reports/summary.json"))
    path = lane / "runs/lvp01/audit-panel.json"
    audit = read(path)
    result_path = lane / "runs/lvp01/archive/output/panel/result.json"
    result = read(result_path)
    arms = {a["arm"]: a for a in audit["arms"]}
    checks, provenance = [], []
    for row in summary:
        if str(row["job_id"]) != str(audit["job_id"]) or row["arm"] not in arms:
            continue
        arm = arms[row["arm"]]
        times = arm["per_time_same_grid_percent"]
        values = {
            "worst_evolved_percent": (max(max(a[1:]) for a in times.values()), "max_case max_evolved_time per_time_same_grid_percent"),
            "worst_all_times_percent": (max(max(a) for a in times.values()), "max_case max_all_times per_time_same_grid_percent"),
            "worst_t0_compression_percent": (max(a[0] for a in times.values()), "max_case per_time_same_grid_percent[0]"),
        }
        inv = [x for x in result["invocations"] if x["name"] == row["arm"]]
        if row["metric"] == "median_gpu_ms":
            add(checks, row, 1000 * statistics.median(x["gpu_seconds"] for x in inv), result_path, "1000 * median(invocations.gpu_seconds)")
        elif row["metric"] in values:
            val, calculation = values[row["metric"]]
            add(checks, row, val, path, calculation)
        provenance.append(dict(subject=row["arm"], job_id=str(row["job_id"]), expected=result["commit"], reported=row.get("source_commit"), passed=(row.get("source_commit") == result["commit"]), check="summary row source_commit equals panel result commit"))
    return checks, provenance


NOTES = {
    "no-second": [
        "The declared main comparison is accuracy across identical cases, not speed across jobs. Resolution speed ratios are within each operator's own job; this does not establish a ROM/operator cost ranking.",
        "Validation selects mean case-maximum error; the selected arm need not minimize the worst case. The report discloses this choice and retains alternatives. Preserve it in paper comparisons.",
        "The initial protocol explicitly permits float32 U-Net/Transolver internals and supplies a float64 control. That historical exception must not silently become the new 3D policy.",
        "Most arms ended at wall budgets, with the still-improving label based on checkpoint recency. This is a heuristic, not a convergence proof. Describe achieved finite-budget errors; calling errors a lower bound on attainable error is mathematically misleading.",
        "Further work: validation-selected continuations and independent seeds, plus a same-allocation ROM/operator/FOM panel. The reported operator resolution knob must remain in the paper; the earlier no-knob premise was withdrawn.",
    ],
    "p-linear": [
        "DESIGN amendment A8 changes the deciding interpretation of falsification after the coarse-mesh result was observed. It is prior to the fine-mesh job but post-observation for the campaign. Report literal and amended conclusions separately.",
        "The direct QR free-bank endpoint is appropriate; the redundant eliminated full-rank neural endpoint contains inert iterations and is not the efficient top-rung cost. The endpoint replacement is documented in amendment A4.",
        "Augmented best-found reconstruction with nonzero correction rank was retracted under amendment A10 because its projector was not orthonormal. The corrected implementation was checked locally, but the old full-cohort oracle column was not rerun. Do not reinstate it through later table generation.",
        "The within-job direct-solver and POD comparisons remain distinct from cross-mesh scaling. The head-capacity job does not justify comparing its timings to the fine-mesh job on another GPU.",
        "Further work: recompute the corrected oracle and diagnose the learned bank/head gap. The algebraic linear limit is supported without claiming a neural advantage over the linear controls.",
    ],
    "w-ladder": [
        "DESIGN amendment A2 introduces an integrator tie band after the local smoke showed a strict head accuracy advantage. It predates cluster jobs but is still informed by observed numerical results. Report strict and banded D1, not only the amended verdict.",
        "The tie band is the difference between worst-case errors from two integrators; it is an empirical diagnostic rather than a certified bound on every case's integration error.",
        "Closed-form linear evolution and nonlinear stepping have different algorithmic costs. The matched-integrator control helps distinguish this from a neural-manifold cost effect and should remain visible.",
        "The retained-value gate was changed after a failed job under amendment A4 to handle tied initializers. Preserve the failure and the field/objective parity evidence; do not describe every original gate as having passed.",
        "Further work: initial/velocity representation and physical trajectory accuracy. Energy conservation alone does not establish accurate wave propagation. The linear-bank result is near the best corrected head in accuracy, not uniformly strictly more accurate.",
    ],
    "b-lowvisc": [
        "The mesh-refinement failure remains explicit. Reduced-versus-reduced comparisons on this mesh do not establish continuum-accurate low-viscosity performance.",
        "The report records a failed POD orthogonality audit check and bounds its effect using an oblique projector. This is a disclosed exception, not a clean all-checks-pass record.",
        "The main panel fails its full-order-frontier criterion; the reduced-only frontier and the correction-ladder span answer different questions. Retain these separate verdicts.",
        "The source-label inheritance bug was repaired in commit 5760a7e256ef3c003956bef7a185a86c88b823b6. Gate, training and panel sources are now attributed separately, and the panel labels are rechecked here against its retained result. The lane's source catalog traces archived bytes to Git objects; historical incumbent-training records lacking a recoverable run commit are explicitly marked unknown with retained artifact hashes. The combined report must not invent one shared scientific source commit.",
        "Further work: a resolved refinement confirmation with the same physical family and fair tuned FOM controls, then any claim of a nonlinear advantage beyond the reduced-model comparison.",
    ],
}


def sample(checks, count=15):
    numeric = [c for c in checks if isinstance(c["reported"], (int, float)) and not isinstance(c["reported"], bool)]
    ordered = sorted(numeric, key=lambda x: (x["source"], str(x["subject"]), x["metric"], str(x["cohort"])))
    if len(ordered) <= count:
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]


def display(v):
    return f"{v:.12g}" if isinstance(v, float) else str(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, func in [("no-second", audit_operators), ("p-linear", audit_poisson), ("w-ladder", audit_waves), ("b-lowvisc", audit_lowvisc)]:
        checks, provenance = func()
        assert len(checks) >= 15, (name, len(checks))
        wt = lane_root(name).parents[1]
        pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=wt, text=True).strip()
        failed = [c for c in checks if not c["passed"]]
        pfailed = [c for c in provenance if not c["passed"]]
        record = dict(lane=name, commit=pin, checks=checks, provenance=provenance, findings=NOTES[name], numerical_failures=len(failed), provenance_failures=len(pfailed), scope="Aggregate and report/design audit; not a fresh full-field or solver validation.")
        all_results[name] = record
        md = [f"# Independent report audit: {name}", "", "Audited from retained records on September 20. This audit recomputes aggregates independently of the lane generator and reviews the documented protocol; it does not rerun solvers or independently validate every saved field.", "", f"Source checkout: `{pin}`. Numeric checks: **{len(checks) - len(failed)}/{len(checks)} pass**. Provenance checks: **{len(provenance) - len(pfailed)}/{len(provenance)} pass**. Methodological findings remain open below.", "", "## Verified-number sample", "", "Values retain the units of their named metric; percentage metrics are explicitly suffixed. The complete machine-readable audit retains every checked row.", "", "| Subject | Metric | Reported | Recomputed | Job | Source and calculation | Pass |", "| --- | --- | ---: | ---: | --- | --- | :---: |"]
        for c in sample(checks):
            md.append(f"| `{c['subject']}` | `{c['metric']}` | {display(c['reported'])} | {display(c['recomputed'])} | `{c['job_id']}` | `{c['source']}`: {c['calculation']} | {c['passed']} |")
        md += ["", "## Mismatches and provenance", ""]
        md += ["No numerical mismatch was found among the recomputed aggregates."] if not failed else [f"- `{c['subject']}` / `{c['metric']}`: reported {display(c['reported'])}, recomputed {display(c['recomputed'])}." for c in failed]
        if pfailed:
            distinct = {(c['job_id'], str(c.get('reported')), str(c.get('expected')), c['check']) for c in pfailed}
            md += ["", "The following provenance failures affect repeated rows:"]
            md += [f"- Job `{j}`: {check}; reported `{rep}`, expected `{exp}`." for j, rep, exp, check in sorted(distinct)]
        else:
            md += ["The source/job checks performed here pass."]
        md += ["", "## Gate history, comparisons, retractions and weakest claims", ""]
        md += [f"- {note}" for note in NOTES[name]]
        md += ["", "Remaining methodological findings are not resolved by aggregate agreement. Full-field audits already retained by the lane are supporting evidence, not new independent field checks performed here. Any altered claim must be applied to the manuscript and its source report before submission.", "", "## Glossary", "", "- **Subject:** a particular model or solver configuration.", "- **Metric:** the named error, time or statistic; same-grid error compares against a numerical solution on the same mesh.", "- **Reported/recomputed:** value in the lane summary/value independently calculated from retained case or invocation arrays.", "- **Job/source:** the cluster allocation identifier/the file containing the checked inputs.", "- **Provenance:** evidence identifying the exact code and data that produced a result.", "- **Gate/amendment:** an acceptance rule/a documented change to the prospective protocol.", "- **FOM/POD:** a full-grid numerical solver/a linear basis derived from training snapshots.", "- **Bank/head/correction rank:** learned spatial functions/their nonlinear coefficient network/the number of additional solved linear directions.", "- **Frontier:** configurations for which no eligible comparator is both no slower and no less accurate, with one strict improvement.", "- **Held-out/validation:** cases excluded from training/cases used to select model settings.", "- **Oracle:** a best-found representation estimate, not a guarantee of the global optimum."]
        (OUT / f"{name}.md").write_text("\n".join(md) + "\n")
        print(name, "numeric", len(checks), "failures", len(failed), "provenance failures", len(pfailed))
    output = ROOT / "reports/2026-09-20-paper-lane-audits.json"
    output.write_text(json.dumps(dict(generated=datetime.now(timezone.utc).isoformat(), generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs=INPUTS, lanes=all_results), indent=2) + "\n")


if __name__ == "__main__":
    main()
