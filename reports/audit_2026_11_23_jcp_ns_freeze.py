"""Verify the immutable source and metadata-only untouched-cohort registration."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory):
    directory = Path(directory).resolve()
    worktree = directory.parents[1]
    frozen_path = directory / "FROZEN_PROTOCOL.json"
    registration_path = directory / "FRESH_COHORT_REGISTRATION.json"
    frozen = json.loads(frozen_path.read_text())
    registration = json.loads(registration_path.read_text())
    files = []
    for relative, digest in frozen["files"].items():
        actual = sha(directory / relative)
        assert actual == digest, relative
        committed = subprocess.check_output([
            "git", "-C", str(worktree), "show",
            frozen["execution_source_commit"] + ":" + str((directory / relative).relative_to(worktree)),
        ])
        assert hashlib.sha256(committed).hexdigest() == digest, "execution commit: " + relative
        files.append({"path": relative, "sha256": actual})
    unchanged = ["shared_initializer.py", "arithmetic_audit.py", "support.py",
                 "check_arithmetic.py", "vendor/head_rom.py", "vendor/ns3d_fom.py"]
    for relative in unchanged:
        accepted = subprocess.check_output([
            "git", "-C", str(worktree), "show",
            frozen["accepted_solver_source_commit"] + ":" + str((directory / relative).relative_to(worktree)),
        ])
        assert accepted == (directory / relative).read_bytes(), relative
    assert registration["seed"] == frozen["eval_seed"] == 2026112301
    assert registration["count"] == frozen["eval_cases"] == 32
    assert registration["frozen_protocol_sha256"] == sha(frozen_path)
    assert registration["generator_sha256"] == sha(directory / "vendor/ns3d_fom.py")
    assert registration["source_freeze_commit"] == frozen["execution_source_commit"]
    assert registration["physical_fields_or_truth_generated"] is False
    assert registration["method_outcomes_inspected"] is False
    source = directory / "vendor/ns3d_fom.py"
    tree = ast.parse(source.read_text())
    generator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parameters")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[generator], type_ignores=[]), str(source), "exec"), namespace)
    parameters = namespace["parameters"]
    local_fresh = parameters(registration["seed"], registration["count"])
    fresh = np.asarray(registration["parameters"], dtype=np.float64)
    # Uniform coordinates reproduce exactly. Platform exp/log can differ in the
    # final viscosity bit; the registered cluster tuples remain authoritative.
    assert np.array_equal(local_fresh[:, :5], fresh[:, :5])
    np.testing.assert_array_max_ulp(local_fresh[:, 5], fresh[:, 5], maxulp=4)
    assert np.array_equal(np.round(local_fresh, 12), np.round(fresh, 12))
    assert hashlib.sha256(fresh.tobytes()).hexdigest() == registration["parameter_sha256"]
    assert len(set(map(tuple, fresh))) == registration["count"]
    checks = registration["known_cohort_overlap_checks"]
    assert {(r["seed"], r["count"]) for r in checks} == {(202609201, 512), (202609202, 64), (2026111901, 16)}
    local_checks = []
    for check in checks:
        known = parameters(check["seed"], check["count"])
        # The exactly reproduced uniform coordinates already distinguish every
        # new case, regardless of tiny platform viscosity differences.
        assert not (set(map(tuple, known[:, :5])) & set(map(tuple, fresh[:, :5])))
        local_checks.append({"seed": check["seed"], "local_parameter_sha256": hashlib.sha256(known.tobytes()).hexdigest(),
                             "registered_cluster_parameter_sha256": check["parameter_sha256"],
                             "overlap_in_first_five_exact_uniform_coordinates": 0})
        for digits, key in ((None, "exact_overlap_count"), (12, "rounded12_overlap_count")):
            old = known if digits is None else np.round(known, digits)
            new = fresh if digits is None else np.round(fresh, digits)
            overlap = len(set(map(tuple, old)) & set(map(tuple, new)))
            assert overlap == check[key] == 0
    assert registration["nonoverlap_pass"] is True
    assert registration["historical_sealed_cohorts_not_opened"] == frozen["historical_sealed_cohorts_closed"]
    for relative in ("configs/fresh-initializers.json", "configs/fresh-phase.json"):
        cfg = json.loads((directory / relative).read_text())
        assert cfg["eval_seed"] == registration["seed"] and cfg["eval_cases"] == registration["count"]
    accepted = directory / "runs/d02/output/records.json"
    assert sha(accepted) == frozen["accepted_development_records_sha256"]
    return {
        "integrity_passed": True,
        "scope": "source content hashes and parameter metadata only; no physical fields, JAX, model evaluation or fresh outcomes",
        "execution_source_commit": frozen["execution_source_commit"],
        "accepted_solver_source_commit": frozen["accepted_solver_source_commit"],
        "frozen_protocol_sha256": sha(frozen_path),
        "registration_sha256": sha(registration_path),
        "registered_at_utc": registration["registered_at_utc"],
        "case_count": registration["count"],
        "seed": registration["seed"],
        "parameter_sha256": registration["parameter_sha256"],
        "nonoverlap_passed": True,
        "duplicate_fresh_cases": 0,
        "known_cohort_overlap_checks": checks,
        "local_generator_crosscheck": {
            "numpy_version": np.__version__,
            "local_parameter_sha256": hashlib.sha256(local_fresh.tobytes()).hexdigest(),
            "exact_uniform_columns": True,
            "differing_viscosity_entries": int(np.sum(local_fresh[:, 5] != fresh[:, 5])),
            "maximum_viscosity_difference": float(np.max(np.abs(local_fresh[:, 5] - fresh[:, 5]))),
            "maximum_allowed_viscosity_ulps_for_cross_platform_audit_only": 4,
            "rounded12_identical": True,
            "frozen_cluster_execution_still_requires_exact_registered_bytes": True,
            "known_cohorts": local_checks,
        },
        "unchanged_accepted_core_files": unchanged,
        "files": files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.experiment)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "known_cohort_overlap_checks")}))
