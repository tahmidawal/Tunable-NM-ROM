"""CPU-only synthetic correctness checks for measurement and candidate selection.

These values are fixtures, never scientific results or optimistic projections.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

import paired_measurement as P


def main():
    truth = np.ones((6, 2, 2), dtype=np.float64)
    case = P.Case(0, truth[0].copy(), 0.1, truth)
    exact = P.Subject("exact", "target", 2, lambda u, nu: np.repeat(u[None], 6, axis=0))
    row = P.measure_one(exact, case)
    assert row["status"] == "ok" and row["evolved_max_error"] == 0
    bad = P.Subject("bad", "pod", 2, lambda u, nu: np.full_like(truth, np.nan))
    assert P.measure_one(bad, case)["status"] == "failed"
    f32 = P.Subject("f32", "pod", 2, lambda u, nu: truth.astype(np.float32))
    assert P.measure_one(f32, case)["status"] == "failed"
    finite_failure = P.Subject("finite_failure", "pod", 2,
        lambda u, nu: P.QueryResult(truth.copy(), False, {"termination": "budget"}))
    failed_row = P.measure_one(finite_failure, case)
    assert failed_row["status"] == "failed"
    assert failed_row["solver_diagnostics"]["termination"] == "budget"

    def data(family, median, worst, ms, complete=True, mesh=2):
        return {"family": family, "mesh": mesh, "case_ids": [0, 1, 2],
                "complete": complete, "median_case_evolved_max_error": median,
                "worst_evolved_error": worst, "median_case_ms": ms}
    summary = {
        "target": data("target", .02, .04, 5),
        "bad_worst": data("classical", .01, .05, 1),
        "bad_median": data("classical", .025, .03, 1),
        "failed": data("classical", .01, .02, .5, complete=False),
        "good_coarse": data("classical", .015, .03, 2, mesh=1),
        "good_same": data("classical", .015, .03, 3),
        "pod": data("pod", .015, .03, 4),
    }
    selection = P.select_comparators(summary, ["target"], 2)["target"]
    assert selection["any_mesh_classical"]["selected"] == "good_coarse"
    assert selection["same_mesh_classical"]["selected"] == "good_same"
    assert selection["pod"]["selected"] == "pod"
    assert not P.eligible(summary["bad_worst"], summary["target"])
    assert not P.eligible(summary["bad_median"], summary["target"])

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        out = Path(temporary) / "fixture"
        subjects = [exact, P.Subject("same", "classical", 2, exact.query)]
        result = P.run_suite(subjects, [case], target_names=["exact"], scoring_mesh=2,
                             out=out, burn=lambda: None,
                             metadata={"synthetic_fixture": True})
        records = [json.loads(line) for line in (out / "invocations.jsonl").read_text().splitlines()]
        assert len(records) == 2*(3 + 3 + 5 + 3)
        assert all(r["evolved_max_error"] == 0 for r in records)
        assert all(r["output_dtype"] == "float64" for r in records)
        assert (out / "frozen_comparators.json").is_file()
        assert result["confirmation"]["exact"]["invocation_count"] == 5
        assert not case.initial.flags.writeable
    print(json.dumps({"synthetic_fixture": True, "all_passed": True,
                      "checks": ["same-call accuracy", "f64/nonfinite failure preservation",
                                 "dual median and worst gate", "same/any mesh distinction",
                                 "selection freeze", "all repetition retention",
                                 "native-input immutable reset", "cold missing explicit"]}))


if __name__ == "__main__":
    main()
