"""Second implementation: worst saved heat endpoints, without a JAX solve.

The experiment owner's full-cohort audit remains required. This independently
checks full output errors, sampled decoder values and every weak time-step
gradient on each model's worst trajectory at each mesh.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from audit_poisson_online_tuning import features, head_with_jacobian

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/accuracy10"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    out = RUN / "archive/outputs"
    result_path = out / "results.json"
    result = json.loads(result_path.read_text())
    assert result["complete"] and result["metadata"]["backend"] == "gpu"
    assert result["metadata"]["x64"] and result["metadata"]["precision"] == "highest"
    assert not result["final_cohort_opened"]
    refs = {(r["intervals"], r["case"]): r for r in result["case_fields"]}
    checkpoints = {}
    evidence = []

    def load(spec):
        field = np.load(out / spec["path"])["field"]
        assert field.dtype == np.float64 and np.isfinite(field).all()
        sha = hashlib.sha256(str((field.shape, field.dtype.str)).encode()
                             + np.ascontiguousarray(field).tobytes()).hexdigest()
        assert sha == spec["sha256_array"]
        return field

    for model in result["models"]:
        path = out / model["path"]
        assert digest(path) == model["sha256"]
        params = pickle.loads(path.read_bytes())["params"]
        checkpoints[model["name"]] = model["sha256"]
        for n in result["settings"]["requested_intervals"]:
            rows = [r for r in result["rows"] if r["intervals"] == n
                    and r["method"] == "nmrom_" + model["name"]]
            row, rep = max(((r, p) for r in rows for p in r["repetitions"]),
                           key=lambda pair: max(pair[1]["vs_physical"]["relative_current"]))
            field, truth = load(rep["field"]), load(refs[n, row["case"]]["physical"])
            differences = np.linalg.norm((field-truth).reshape(len(field), -1), axis=1)
            reference_norms = np.linalg.norm(truth.reshape(len(field), -1), axis=1)
            errors = differences/reference_norms
            np.testing.assert_allclose(errors, rep["vs_physical"]["relative_current"],
                                       atol=2e-13, rtol=2e-13)
            ij = np.random.default_rng(911101).integers(1, n, size=(131, 2))
            bank = features(params, ij/n)
            zs = np.array(rep["solver"]["latents"])
            predicted = np.array([bank @ head_with_jacobian(params, z)[0] for z in zs])
            saved = field[:, ij[:, 0]-1, ij[:, 1]-1]
            decode_error = float(np.linalg.norm(predicted-saved)/np.linalg.norm(saved))
            assert decode_error < 2e-10
            op = np.load(out / "assembly" / f"n{n}.npz")
            dt, nu = result["settings"]["dt"], result["config"]["diffusivity"]
            factor = (1-dt*nu*op["mode_lam"]/2)/(1+dt*nu*op["mode_lam"]/2)
            matrix = op["matrix"]
            internal = np.array(rep["solver"]["internal_latents"])
            gradients, deviations = [], []
            for before, after, saved_info in zip(internal[:-1], internal[1:], rep["solver"]["steps"]):
                target = factor*(matrix @ head_with_jacobian(params, before)[0])
                scale = max(np.linalg.norm(target), 1e-14)
                h, j = head_with_jacobian(params, after)
                residual, jac = (matrix@h-target)/scale, matrix@j.T/scale
                gradient = np.linalg.norm(jac.T@residual)/max(np.linalg.norm(jac), 1e-30)
                actual = np.array([np.linalg.norm(residual), gradient])
                expected = np.array(saved_info)[[3, 4]]
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-8)
                if saved_info[2] == 1:
                    assert gradient <= result["settings"]["gradient_tolerance"]+1e-11
                gradients.append(float(gradient))
                deviations.append(float(np.max(abs(actual-expected))))
            evidence.append(dict(model=model["name"], intervals=n, case=row["case"],
                                 cohort=row["cohort"], repetition=rep["repetition"],
                                 field_sha256=rep["field"]["sha256_array"],
                                 worst_current_relative_error=float(max(errors)),
                                 maximum_sampled_decoder_relative_error=decode_error,
                                 maximum_weak_diagnostic_difference=max(deviations),
                                 maximum_weak_stationarity=max(gradients)))
            print(model["name"], n, row["case"], max(errors), flush=True)
    return dict(passed=True, source_result_sha256=digest(result_path),
                source_script_sha256=digest(Path(__file__)),
                shared_numpy_decoder_sha256=digest(ROOT/"reports/audit_poisson_online_tuning.py"),
                source_result_path=str(result_path.relative_to(ROOT)), metadata=result["metadata"],
                scope="Worst trajectory for every model and mesh; second CPU implementation; no new solves or timings.",
                checkpoints=checkpoints, checked_trajectories=len(evidence), endpoints=evidence)


if __name__ == "__main__":
    result = audit()
    path = ROOT/"reports/2026-09-11-heat-accuracy.coordinator-audit.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
