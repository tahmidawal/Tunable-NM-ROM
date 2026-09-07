"""Native post-pull hash/numerical audit and common-observation-grid comparisons."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def restrict(a, fine, coarse):
    assert fine % coarse == 0
    full = np.pad(a, ((0, 0), (1, 1), (1, 1)))
    stride = fine//coarse
    return full[:, stride:fine:stride, stride:fine:stride]


def current_error(fields, truth):
    fields, truth = fields.reshape(len(fields), -1), truth.reshape(len(truth), -1)
    return np.linalg.norm(fields-truth, axis=1)/np.linalg.norm(truth, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive
    result = json.loads((archive/"outputs/results.json").read_text())
    cfg = result["config"]
    manifest = json.loads((archive/"SOURCE-MANIFEST.json").read_text())
    checked = 0
    for row in (archive/"ARCHIVE.sha256").read_text().splitlines():
        expected, path = row.split("  ", 1)
        assert hashlib.sha256((archive/path).read_bytes()).hexdigest() == expected, path
        checked += 1
    for path, digest in manifest["sha256"].items():
        committed = subprocess.check_output(["git", "show", f"{manifest['source_commit']}:{path}"])
        assert hashlib.sha256(committed).hexdigest() == digest, path
    assert hashlib.sha256((archive/"outputs/checkpoint.pkl").read_bytes()).hexdigest() == result["checkpoint_sha256"]
    assert result["complete"] and result["verification"]["passed"]
    assert result["metadata"]["backend"] == "gpu" and result["metadata"]["x64"]
    assert result["metadata"]["precision"] == "highest"
    log = (archive/f"job-{result['metadata']['job_id']}.log").read_text()
    assert "jax_backend=gpu" in log and "PILOT COMPLETE" in log
    assert not re.search(r"captured.*constant|Traceback|out of memory|RESOURCE_EXHAUSTED|No space left|failed call to cuInit", log, flags=re.I)
    observation = min(cfg["evaluation_intervals"])
    data = {n: np.load(archive/f"outputs/fields_N{n}.npz") for n in cfg["evaluation_intervals"]}
    groups = {}
    native_max_mismatch = 0.
    for row in result["rows"]:
        n, case, method = row["intervals"], row["case"], row["method"]
        assert row["frozen_checkpoint"] == result["checkpoint_sha256"]
        physical = data[n][f"n{n}_case{case}_physical_reference"]
        common_physical = restrict(physical, n, observation)
        baseline_common = data[observation][f"n{observation}_case{case}_physical_reference"]
        np.testing.assert_array_equal(common_physical, baseline_common)
        group = groups.setdefault((n, method), dict(intervals=n, method=method, cases=[]))
        records = []
        for rep in row["repetitions"]:
            fields = data[n][f"n{n}_case{case}_{method}_rep{rep['repetition']}"]
            assert fields.shape == (len(cfg["times"]), n-1, n-1)
            assert np.all(np.isfinite(fields))
            recomputed = current_error(fields, physical)
            saved = np.asarray(rep["vs_physical_reference"]["relative_current"])
            mismatch = float(max(abs(recomputed-saved)))
            native_max_mismatch = max(native_max_mismatch, mismatch)
            np.testing.assert_allclose(recomputed, saved, rtol=1e-12, atol=1e-14)
            phases = rep["phases"]
            assert all(value >= 0 for value in phases.values())
            assert sum(value for key, value in phases.items() if key != "query_seconds") <= phases["query_seconds"]+1e-8
            initial_nonstationary = step_nonstationary = ignored_initial_nonstationary = 0
            if rep["solver"]:
                initial = np.asarray(rep["solver"]["initial_fits"])
                best = int(np.argmin(initial[:, 3]))
                initial_nonstationary = int(initial[best, 4] > cfg["gradient_tolerance"])
                ignored_initial_nonstationary = int(np.sum(initial[:, 4] > cfg["gradient_tolerance"]))-initial_nonstationary
                step_info = np.asarray(rep["solver"]["steps"])
                step_nonstationary = int(np.sum(step_info[:, 4] > cfg["gradient_tolerance"]))
                assert int(initial[best, 2]) != 4 and np.all(step_info[:, 2] != 4)
            common_fields = restrict(fields, n, observation)
            common = current_error(common_fields, common_physical)
            records.append(dict(repetition=rep["repetition"], time_max_common_current_error=float(max(common)),
                                selected_initial_nonstationary=initial_nonstationary,
                                unused_initial_starts_nonstationary=ignored_initial_nonstationary,
                                rollout_steps_nonstationary=step_nonstationary,
                                valid=not (initial_nonstationary or step_nonstationary)))
        group["cases"].append(dict(case=case, repetitions=records))
    reference_uncertainty = max(result["verification"]["reference_spectral_refinement_current_errors"])
    for group in groups.values():
        cases = group["cases"]
        worst_by_case = [max(rep["time_max_common_current_error"] for rep in case["repetitions"]) for case in cases]
        group["common_current_median"] = float(np.median(worst_by_case))
        group["common_current_worst"] = max(worst_by_case)
        group["all_repetitions_valid"] = all(rep["valid"] for case in cases for rep in case["repetitions"])
        group["eligible_targets_with_reference_margin"] = [target for target in cfg["accuracy_targets"] if group["all_repetitions_valid"] and max(worst_by_case)+reference_uncertainty <= target and reference_uncertainty <= target/10]
    audit = dict(passed=True, source_commit=manifest["source_commit"], archive_files_verified=checked,
                 checkpoint_sha256=result["checkpoint_sha256"], observation_intervals=observation,
                 observation_contract="Nested restriction of every saved method field and common continuum reference to the same interior grid, with physical area-weighted L2; all requested times and repetitions retained.",
                 native_json_error_max_discrepancy=native_max_mismatch,
                 reference_refinement_uncertainty=reference_uncertainty, groups=list(groups.values()))
    args.out.write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({key: value for key, value in audit.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
