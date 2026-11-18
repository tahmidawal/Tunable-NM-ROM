"""Independent retained-JSON audit of paired cost, selection and overflow exclusion."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
COST = HERE.parents[2]/"2026-11-18-jcp-paired-cost/experiments/jcp-paired-cost"
parser = argparse.ArgumentParser()
parser.add_argument("tag", choices=("p03", "p04"))
args = parser.parse_args()
JOB = COST/"runs"/args.tag
OUT = JOB/"output"
PAIR = OUT/"paired"
load = lambda p: json.loads(p.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def finite(row):
    if row["status"] != "ok":
        return False
    values = row.get("relative_errors", [])
    return (len(values) == 6 and all(x is not None and math.isfinite(x) for x in values)
            and row.get("evolved_max_error") is not None and math.isfinite(row["evolved_max_error"]))


verified = []
for line in (JOB/"COMPLETE.sha256").read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    assert sha(JOB/name) == expected, name
    verified.append({"relative_path": name, "sha256": expected})
rows = [json.loads(line) for line in (PAIR/"invocations.jsonl").read_text().splitlines()]
contract, result = load(PAIR/"contract.json"), load(PAIR/"result.json")
selection_saved = load(PAIR/"selection.json")
frozen = load(PAIR/"frozen_comparators.json")
provenance = load(OUT/"provenance.json")
cases, targets = contract["case_ids"], contract["target_names"]
subjects = {s["name"]: s for s in contract["subjects"]}
assert cases == list(range(16)) and len(subjects) == (105 if args.tag == "p03" else 107)
assert len(targets) == (2 if args.tag == "p03" else 4)
assert provenance["backend"] == "gpu" and provenance["x64"] and provenance["precision"] == "highest"
assert provenance["config"]["eval_seed"] == 202609202
assert frozen["selection_sha256"] == sha(PAIR/"selection.json")
fixture = load(JOB/"fixture.json")
assert fixture["passed"] and fixture["job_id"] == str(provenance["job_id"])
assert fixture["source_sha256"]["ns_query.py"] == provenance["query_sha256"]
parity = load(OUT/"implementation_parity.json")
assert len(parity) == len(targets)*len(cases) and all(p["passed"] for p in parity)
assert {(p["subject"], p["case"]) for p in parity} == {(t,c) for t in targets for c in cases}
for p in parity:
    d = p["solver_diagnostics"]
    assert d["initial_fit_reason"] == 1 and d["initial_fit_gradient"] <= provenance["config"]["rom"]["fit_gtol"]
    assert all(x == 1 for x in d["step_reasons"])
    assert max(d["step_gradients"]) <= provenance["config"]["rom"]["gtol"]
    assert p["max_relative_field_difference"] <= provenance["config"]["implementation_parity_gate"]
    assert p["reference_refinement_difference"] <= p["reference_refinement_bound"]
invalid = [r for r in rows if r["status"] == "ok" and not finite(r)]
affected = {r["subject"] for r in invalid}
for row in rows:
    assert math.isfinite(row["elapsed_ms"]) and row["elapsed_ms"] >= 0
    if finite(row):
        assert row["output_dtype"] == "float64" and row["output_shape"] == [6,3,64,64,64]
        assert row["evolved_max_error"] == max(row["relative_errors"][1:])
        assert row["initial_error"] == row["relative_errors"][0] and row["final_error"] == row["relative_errors"][-1]


def summarize(block, names, ids, repetitions):
    answer = {}
    for name in names:
        rr = [r for r in rows if r["block"] == block and r["subject"] == name]
        assert len(rr) == repetitions*len(ids), (block, name, len(rr))
        assert {(r["case_id"],r["repetition"]) for r in rr} == {(c,k) for c in ids for k in range(repetitions)}
        per_case = {}
        for c in ids:
            good = [r for r in rr if r["case_id"] == c and finite(r)]
            if good:
                per_case[c] = {"median_ms": float(np.median([r["elapsed_ms"] for r in good])),
                    "worst_error": max(r["evolved_max_error"] for r in good),
                    "times": [r["elapsed_ms"] for r in good]}
        complete = len(per_case) == len(ids) and all(finite(r) for r in rr)
        answer[name] = {"complete": complete, "cases": per_case,
            "median_ms": float(np.median([v["median_ms"] for v in per_case.values()])) if per_case else None,
            "median_error": float(np.median([v["worst_error"] for v in per_case.values()])) if per_case else None,
            "worst_error": max(v["worst_error"] for v in per_case.values()) if per_case else None}
    return answer


def eligible(c, t):
    return c["complete"] and t["complete"] and c["median_error"] <= t["median_error"] and c["worst_error"] <= t["worst_error"]


selection = summarize("selection", subjects, cases, contract["selection_repetitions"])
for name, entry in selection.items():
    if entry["complete"]:
        old = selection_saved[name]
        assert old["complete"]
        for new_key, old_key in (("median_ms","median_case_ms"),("median_error","median_case_evolved_max_error"),("worst_error","worst_evolved_error")):
            assert np.isclose(entry[new_key],old[old_key],rtol=1e-13,atol=1e-16)
confirmed_names = set(targets)
for t in targets:
    assert selection[t]["complete"]
    for group in ("pod", "same_mesh_classical", "any_mesh_classical"):
        names = [n for n,s in subjects.items() if eligible(selection[n],selection[t]) and
            (s["family"] == "pod" if group == "pod" else s["family"] == "classical" and (group == "any_mesh_classical" or s["mesh"] == 64))]
        expected = min(names,key=lambda n:(selection[n]["median_ms"],n)) if names else None
        choice = frozen["comparators"][t][group]
        assert sorted(names) == choice["eligible"] and expected == choice["selected"], (t,group)
        assert not (set(names) & affected)
        if expected is not None:
            confirmed_names.add(expected)
assert not (confirmed_names & affected)
assert set(result["confirmation"]) == confirmed_names
assert {r["subject"] for r in rows if r["block"] == "confirmation"} == confirmed_names
confirmation = summarize("confirmation", confirmed_names, cases, contract["confirmation_repetitions"])
before = summarize("sentinel_before", subjects, cases[:1],3)
after = summarize("sentinel_after", confirmed_names,cases[:1],3)
comparisons = {}
sym = lambda a,b:max(a/b,b/a)
for t in targets:
    comparisons[t] = {}
    for group, choice in frozen["comparators"][t].items():
        n = choice["selected"]
        saved = result["comparisons"][t][group]
        if n is None:
            assert not saved["qualified"]
            comparisons[t][group] = {"qualified":False,"reason":"no eligible comparator"}
            continue
        within, sentinel, selection_gap = {}, {}, {}
        for arm in (t,n):
            per_case = confirmation[arm]["cases"]
            within[arm] = max(sym(np.median(v["times"][:len(v["times"])//2]),np.median(v["times"][-(len(v["times"])//2):])) for v in per_case.values())
            sentinel[arm] = sym(before[arm]["median_ms"],after[arm]["median_ms"])
            selection_gap[arm] = sym(selection[arm]["median_ms"],confirmation[arm]["median_ms"])
        accuracy = eligible(confirmation[n],confirmation[t])
        stable = all(x <= contract["drift_gate_ratio"] for d in (within,sentinel,selection_gap) for x in d.values())
        qualified = bool(accuracy and stable)
        ratio = confirmation[n]["median_ms"]/confirmation[t]["median_ms"]
        assert saved["qualified"] == qualified and saved["accuracy_eligible"] == accuracy
        assert saved["all_drift_gates_pass"] == stable
        assert (saved["comparator_over_target"] is None) if not qualified else np.isclose(saved["comparator_over_target"],ratio,rtol=1e-13)
        paired = [confirmation[n]["cases"][c]["median_ms"]/confirmation[t]["cases"][c]["median_ms"] for c in cases]
        comparisons[t][group] = {"comparator":n,"qualified":qualified,"accuracy_eligible":accuracy,
            "within_confirmation_drift":within,"sentinel_drift":sentinel,"selection_confirmation_drift":selection_gap,
            "ratio_of_median_costs_diagnostic":ratio,"median_casewise_ratio_diagnostic":float(np.median(paired)),
            "target_faster_case_count":sum(x>1 for x in paired),"case_count":len(cases),
            "speed_claim_allowed":qualified}

accounting = list(csv.DictReader((JOB/"sacct.txt").read_text().splitlines(),delimiter="|"))
allocation = next(r for r in accounting if r["JobID"] == str(provenance["job_id"]))
assert allocation["State"] == "COMPLETED" and allocation["ExitCode"] == "0:0"
log = (JOB/"job.log").read_text()
assert "jax_backend=gpu" in log
assert not any(s in log.lower() for s in ("out of memory","resource_exhausted","captured constant","large constant","no space left"))
record = {"reviewer":"head_mechanism_run","decision":"Raw selection/confirmation audit accepted; only comparisons passing every recorded accuracy/drift gate permit speed claims.",
    "job_id":provenance["job_id"],"gpu":provenance["gpu"],"allocated_gpu_seconds":int(allocation["ElapsedRaw"]),
    "verified_file_count":len(verified),"rehashed_files":verified,"raw_invocation_sha256":sha(PAIR/"invocations.jsonl"),
    "invocation_count":len(rows),"target_parity_checks":len(parity),
    "undefined_metric_candidate_ids":sorted(affected),"undefined_metric_invocation_count":len(invalid),
    "undefined_metric_invocations":[{k:r[k] for k in ("block","repetition","order","subject","case_id","output_sha256")} for r in invalid],
    "undefined_metric_candidates_never_eligible_selected_or_confirmed":True,
    "comparisons":comparisons,"confirmation_recomputed":confirmation,
    "scope":"Same allocation and original exposed cases. Legacy ok/null-metric records remain unmodified and are excluded by independent replay; no batching-only causal attribution across different GPUs."}
destination = HERE/"reviews"/f"p3-{args.tag}-independent-result-review.json"
destination.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("rehashed_files","undefined_metric_invocations","confirmation_recomputed")},indent=2))
