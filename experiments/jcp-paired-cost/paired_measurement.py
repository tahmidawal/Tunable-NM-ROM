"""Same-invocation complete-query measurement with retained randomized repetitions.

This module intentionally imports no JAX. A scientific adapter must provide pure
queries accepting the original host physical field and returning host fields. It
owns GPU warmup and synchronization inside that contract, as well as provenance.
The first campaign does not allow this harness to open a new confirmation cohort.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Case:
    case_id: int
    initial: np.ndarray
    viscosity: float
    truth: np.ndarray


@dataclass(frozen=True)
class QueryResult:
    """Optional solver diagnostics from the very invocation whose cost is timed."""
    fields: np.ndarray
    numerical_valid: bool
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Subject:
    name: str
    family: str  # target, pod, or classical
    mesh: int
    # A finite output is insufficient for iterative solvers: query must raise on
    # invalid internal solves or return QueryResult(numerical_valid=False,...).
    # Fixed-iteration configurations may instead pin an explicit parity and
    # numerical qualification record in metadata; no convergence is inferred.
    query: Callable[[np.ndarray, float], np.ndarray | QueryResult]
    metadata: dict = field(default_factory=dict)
    # Cold callback owns asset activation/cache reset and returns complete output.
    cold_query: Callable[[np.ndarray, float], np.ndarray] | None = None


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def output_metrics(output, case: Case):
    a = np.asarray(output)
    if a.shape != case.truth.shape:
        raise ValueError(f"returned shape {a.shape} differs from truth {case.truth.shape}")
    if a.dtype != np.float64:
        raise ValueError(f"returned dtype {a.dtype} is not f64")
    if not np.isfinite(a).all():
        raise ValueError("nonfinite returned field")
    denom = float(np.linalg.norm(case.truth[0].ravel()))
    if not (math.isfinite(denom) and denom > 0):
        raise ValueError("invalid original truth initial norm")
    errors = np.linalg.norm((a - case.truth).reshape(a.shape[0], -1), axis=1) / denom
    return {
        "relative_errors": errors.tolist(),
        "initial_error": float(errors[0]),
        "final_error": float(errors[-1]),
        "evolved_max_error": float(np.max(errors[1:])),
        "output_sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
        "normalization": "original truth initial field L2 norm",
        "output_shape": list(a.shape),
        "output_dtype": str(a.dtype),
    }


def measure_one(subject: Subject, case: Case, *, cold=False):
    call = subject.cold_query if cold else subject.query
    if call is None:
        return {"subject": subject.name, "case_id": case.case_id,
                "status": "missing", "reason": "no directly timed cold deployment callback"}
    start = time.perf_counter_ns()
    try:
        # Adapter contract is host output: np.asarray must not hide an asynchronous
        # device completion beyond the timer. Host conversion belongs inside it.
        value = call(case.initial, case.viscosity)
        diagnostics = value.diagnostics if isinstance(value, QueryResult) else {}
        numerical_valid = value.numerical_valid if isinstance(value, QueryResult) else True
        output = np.asarray(value.fields if isinstance(value, QueryResult) else value)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if not numerical_valid:
            return {"subject": subject.name, "case_id": case.case_id, "status": "failed",
                    "elapsed_ms": elapsed, "reason": "adapter reported numerical solver failure",
                    "solver_diagnostics": diagnostics}
        metrics = output_metrics(output, case)
        return {"subject": subject.name, "case_id": case.case_id, "status": "ok",
                "elapsed_ms": elapsed, "solver_diagnostics": diagnostics, **metrics}
    except Exception as exc:
        return {"subject": subject.name, "case_id": case.case_id, "status": "failed",
                "elapsed_ms": (time.perf_counter_ns() - start) / 1e6,
                "reason": f"{type(exc).__name__}: {exc}"}


def block(subjects: list[Subject], cases: list[Case], repetitions: int, seed: int,
          label: str, output: Path, burn: Callable[[], None]):
    if not subjects or not cases or repetitions < 1:
        raise ValueError("empty timing block")
    rng = np.random.default_rng(seed)
    rows = []
    previous = None
    # Every round starts a separately burned-in block; the complete case/method
    # Cartesian product is randomly permuted, not measured one method at a time.
    for rep in range(repetitions):
        burn()
        pairs = [(s, c) for s in subjects for c in cases]
        for order, index in enumerate(rng.permutation(len(pairs))):
            subject, case = pairs[int(index)]
            row = measure_one(subject, case)
            row.update(block=label, repetition=rep, order=order, order_seed=seed,
                       preceded_by=previous)
            previous = {"subject": subject.name, "case_id": case.case_id}
            rows.append(row)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a") as stream:
                stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
                stream.flush()
    return rows


def summarize(rows, subjects, case_ids):
    result = {}
    case_ids = list(case_ids)
    for subject in subjects:
        selected = [r for r in rows if r["subject"] == subject.name]
        good = [r for r in selected if r["status"] == "ok"]
        failures = [r for r in selected if r["status"] != "ok"]
        per_case = {}
        for case_id in case_ids:
            rr = [r for r in good if r["case_id"] == case_id]
            if not rr:
                continue
            times = np.asarray([r["elapsed_ms"] for r in rr])
            q1, q3 = np.quantile(times, [0.25, 0.75])
            per_case[str(case_id)] = {
                "median_ms": float(np.median(times)),
                "repetitions_ms": times.tolist(),
                "outlier_count_tukey": int(np.sum((times < q1 - 1.5*(q3-q1)) |
                                                  (times > q3 + 1.5*(q3-q1)))),
                "evolved_max_error": max(r["evolved_max_error"] for r in rr),
                "initial_error": max(r["initial_error"] for r in rr),
                "final_error": max(r["final_error"] for r in rr),
                "repeated_output_hashes": sorted(set(r["output_sha256"] for r in rr)),
                "max_error_repetition_spread": max(r["evolved_max_error"] for r in rr)
                                               - min(r["evolved_max_error"] for r in rr),
            }
        complete = len(per_case) == len(case_ids) and not failures
        ee = [x["evolved_max_error"] for x in per_case.values()]
        tt = [x["median_ms"] for x in per_case.values()]
        result[subject.name] = {
            "family": subject.family, "mesh": subject.mesh, "complete": complete,
            "failure_count": len(failures), "invocation_count": len(selected),
            "case_ids": case_ids, "cases": per_case,
            "median_case_ms": float(np.median(tt)) if tt else None,
            "median_case_evolved_max_error": float(np.median(ee)) if ee else None,
            "worst_evolved_error": max(ee) if ee else None,
            "outlier_count_tukey": sum(x["outlier_count_tukey"] for x in per_case.values()),
        }
    return result


def eligible(candidate, target):
    return (candidate["complete"] and target["complete"] and
            candidate["case_ids"] == target["case_ids"] and
            candidate["median_case_evolved_max_error"] <= target["median_case_evolved_max_error"] and
            candidate["worst_evolved_error"] <= target["worst_evolved_error"])


def select_comparators(summary, target_names, scoring_mesh):
    selected = {}
    for name in target_names:
        target = summary[name]
        selections = {}
        for group in ("pod", "same_mesh_classical", "any_mesh_classical"):
            def in_group(c):
                if group == "pod":
                    return c["family"] == "pod"
                return c["family"] == "classical" and (
                    group == "any_mesh_classical" or c["mesh"] == scoring_mesh)
            candidates = {n: c for n, c in summary.items()
                          if n != name and in_group(c) and eligible(c, target)}
            best = min(candidates, key=lambda n: (candidates[n]["median_case_ms"], n)) if candidates else None
            selections[group] = {"selected": best, "eligible": sorted(candidates)}
        selected[name] = selections
    return selected


def confirmation_summary(summary, freeze, drift_gate_ratio, selection,
                         sentinels_before, sentinels_after):
    comparisons = {}
    for name, groups in freeze.items():
        comparisons[name] = {}
        target = summary[name]
        for group, choice in groups.items():
            chosen = choice["selected"]
            if chosen is None:
                comparisons[name][group] = {"qualified": False, "reason": "no eligible selection comparator"}
                continue
            candidate = summary[chosen]
            match = eligible(candidate, target)
            drifts = {}
            for arm in (name, chosen):
                ratios = []
                for case in summary[arm]["cases"].values():
                    x = case["repetitions_ms"]
                    if len(x) >= 4:
                        half = len(x) // 2
                        a, b = float(np.median(x[:half])), float(np.median(x[-half:]))
                        ratios.append(max(a / b, b / a))
                drifts[arm] = max(ratios) if ratios else None
            drift_ok = all(x is not None and x <= drift_gate_ratio for x in drifts.values())
            sentinel_drifts, selection_drifts = {}, {}
            for arm in (name, chosen):
                x, y = sentinels_before[arm], sentinels_after[arm]
                if x["complete"] and y["complete"]:
                    a, b = x["median_case_ms"], y["median_case_ms"]
                    sentinel_drifts[arm] = max(a / b, b / a)
                else:
                    sentinel_drifts[arm] = None
                x, y = selection[arm], summary[arm]
                if x["complete"] and y["complete"]:
                    a, b = x["median_case_ms"], y["median_case_ms"]
                    selection_drifts[arm] = max(a / b, b / a)
                else:
                    selection_drifts[arm] = None
            extra_drifts = list(sentinel_drifts.values()) + list(selection_drifts.values())
            extra_ok = all(x is not None and x <= drift_gate_ratio for x in extra_drifts)
            qualified = bool(match and drift_ok and extra_ok)
            comparisons[name][group] = {
                "comparator": chosen, "accuracy_eligible": bool(match),
                "within_confirmation_drift": drifts, "drift_pass": drift_ok,
                "sentinel_start_end_drift": sentinel_drifts,
                "selection_confirmation_drift": selection_drifts,
                "all_drift_gates_pass": bool(drift_ok and extra_ok),
                "qualified": qualified,
                "comparator_over_target": candidate["median_case_ms"] / target["median_case_ms"]
                    if qualified else None,
                "ratio_direction": "above one favors target; qualified same-allocation only",
            }
    return comparisons


def run_suite(subjects, cases, *, target_names, scoring_mesh, out, burn,
              selection_repetitions=3, confirmation_repetitions=5,
              seed=2026111803, drift_gate_ratio=1.1, metadata=None):
    """Measure already scientifically approved, compiled subjects on exposed cases.

    This does not compile or select a scientific configuration. The adapter must
    persist separate setup/compile measurements and numerical parity checks.
    """
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("one output directory per run; refusing to mix repetitions")
    out.mkdir(parents=True, exist_ok=True)
    names = [s.name for s in subjects]
    if len(names) != len(set(names)) or not set(target_names) <= set(names):
        raise ValueError("duplicate or missing target name")
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("duplicate cases")
    for case in cases:
        if case.initial.dtype != np.float64 or case.truth.dtype != np.float64:
            raise ValueError("all input and truth fields must be f64")
        if not np.array_equal(case.initial, case.truth[0]):
            raise ValueError("every method must receive the original physical truth initial field")
        case.initial.flags.writeable = False
    write_json(out / "contract.json", {
        "schema": "paired-complete-query-v1", "metadata": metadata or {},
        "subjects": [{"name": s.name, "family": s.family, "mesh": s.mesh,
                      "metadata": s.metadata} for s in subjects],
        "target_names": target_names, "case_ids": [c.case_id for c in cases],
        "scope": "development cases; fresh timing repetitions only",
        "timing_scope": "host physical input through host full fields; all native initialization recomputed",
        "selection_repetitions": selection_repetitions,
        "confirmation_repetitions": confirmation_repetitions,
        "order_seed": seed, "drift_gate_ratio": drift_gate_ratio,
    })
    sent0 = block(subjects, cases[:1], 3, seed + 100, "sentinel_before",
                  out / "invocations.jsonl", burn)
    sent0_summary = summarize(sent0, subjects, [cases[0].case_id])
    rows = block(subjects, cases, selection_repetitions, seed, "selection",
                 out / "invocations.jsonl", burn)
    selection = summarize(rows, subjects, [c.case_id for c in cases])
    freeze = select_comparators(selection, target_names, scoring_mesh)
    write_json(out / "selection.json", selection)
    write_json(out / "frozen_comparators.json", {
        "selection_sha256": hashlib.sha256((out / "selection.json").read_bytes()).hexdigest(),
        "comparators": freeze,
        "rule": "both median and maximum casewise evolved error no worse; select case-median elapsed time",
    })
    confirm_names = set(target_names)
    for groups in freeze.values():
        confirm_names.update(x["selected"] for x in groups.values() if x["selected"] is not None)
    confirmed = [s for s in subjects if s.name in confirm_names]
    rows2 = block(confirmed, cases, confirmation_repetitions, seed + 1, "confirmation",
                  out / "invocations.jsonl", burn)
    summary = summarize(rows2, confirmed, [c.case_id for c in cases])
    sent1 = block(confirmed, cases[:1], 3, seed + 101, "sentinel_after",
                  out / "invocations.jsonl", burn)
    sent1_summary = summarize(sent1, confirmed, [cases[0].case_id])
    result = {
        "schema": "paired-complete-query-result-v1", "confirmation": summary,
        "comparisons": confirmation_summary(summary, freeze, drift_gate_ratio, selection,
                                            sent0_summary, sent1_summary),
        "sentinels_before": sent0_summary, "sentinels_after": sent1_summary,
        "cold_deployment": "not yet measured by run_suite; adapter must provide separate direct invocation record",
        "selection_to_confirmation_median_ratios": {
            s.name: summary[s.name]["median_case_ms"] / selection[s.name]["median_case_ms"]
            if summary[s.name]["median_case_ms"] and selection[s.name]["median_case_ms"] else None
            for s in confirmed},
    }
    write_json(out / "result.json", result)
    return result
