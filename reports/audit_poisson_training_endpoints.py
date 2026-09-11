"""Independent CPU check of each trained Poisson model's worst mesh endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.fft import dstn

from audit_poisson_online_tuning import features, head_with_jacobian, source

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def audit(run):
    out = run/"cluster/out/pilot"
    path = out/"result.json"
    data = json.loads(path.read_text())
    assert data["complete"] and data["provenance"]["backend"] == "gpu"
    assert data["provenance"]["x64"] and data["provenance"]["matmul_precision"] == "highest"
    models = {m["id"]: m for m in data["checkpoints"]}
    evidence = []
    for model in data["config"]["trained_model_ids"]:
        checkpoint = out/models[model]["path"]
        assert sha(checkpoint) == models[model]["sha256"]
        params = pickle.loads(checkpoint.read_bytes())["params"]
        for n in data["config"]["intervals"]:
            rows = [r for r in data["rows"] if r["intervals"] == n and r["method"] == model]
            r = max(rows, key=lambda r: r["physical_error"])
            predicted = np.load(out/"fields"/(r["field_sha256"]+".npz"))["field"]
            assert predicted.dtype == np.float64 and np.isfinite(predicted).all()
            assert array_sha(predicted) == r["field_sha256"]
            refs = np.load(out/"references"/f"n{n}_case{r['case']}.npz")
            error = float(np.linalg.norm(predicted-refs["fine"])/np.linalg.norm(refs["fine"]))
            delta = float(np.linalg.norm(refs["coarse"]-refs["fine"])/np.linalg.norm(refs["fine"]))
            assert abs(error-r["physical_error"]) < 1e-12
            assert abs(delta-r["reference_delta"]) < 1e-12
            assert abs((error+delta)/(1-delta)-r["conservative_physical_error"]) < 1e-12
            setup = next(s for s in data["setup"] if s["model"] == model and s["intervals"] == n)
            modes = np.array(setup["mode_indices"])
            rhs = source(n, data["cohort"]["parameters"][r["case"]])
            eigenvalues = (4*n*n*np.sin(np.pi*modes/(2*n))**2).sum(axis=1)
            target = dstn(rhs[1:-1, 1:-1], type=1, norm="ortho")[modes[:, 0]-1, modes[:, 1]-1]/eigenvalues
            cache = np.load(out/f"cache_n{n}_{model}.npz")
            matrix = cache["B"]
            assert array_sha(matrix) == r["operator_sha256"]
            latent = np.array(r["latent"])
            h, j = head_with_jacobian(params, latent)
            residual, jac = matrix@h-target, matrix@j.T
            rn = float(np.linalg.norm(residual))
            gradient = float(np.linalg.norm(jac.T@residual)/(np.linalg.norm(jac)*rn+1e-300))
            assert abs(rn-r["residual"])/rn < 2e-10
            assert abs(gradient-r["stationarity"]) < 2e-10
            assert (gradient <= data["config"]["stationarity_tolerance"]) == r["stationary"]
            nearest = int(np.argmin(np.sum((cache["predictions"]-target)**2, axis=1)))
            assert nearest == r["selected_training_code_index"]
            np.testing.assert_array_equal(cache["codes"][nearest], r["initial_latent"])
            ij = np.random.default_rng(911102).integers(1, n, (137, 2))
            decoded = features(params, ij/n)@h
            saved = predicted[ij[:, 0], ij[:, 1]]
            parity = float(np.linalg.norm(decoded-saved)/np.linalg.norm(saved))
            assert parity < 2e-10
            assert not np.any(predicted[[0, -1], :]) and not np.any(predicted[:, [0, -1]])
            evidence.append(dict(model=model, intervals=n, case=r["case"], group=r["group"],
                                 physical_error=error, reference_delta=delta, stationarity=gradient,
                                 decoder_relative_difference=parity, checkpoint_sha256=sha(checkpoint),
                                 operator_sha256=r["operator_sha256"], field_sha256=r["field_sha256"]))
            print(model, n, r["case"], error, flush=True)
    return dict(passed=True, source_result_path=str(path.relative_to(ROOT)), source_result_sha256=sha(path),
                source_script_sha256=sha(Path(__file__)),
                shared_numpy_decoder_sha256=sha(ROOT/"reports/audit_poisson_online_tuning.py"),
                provenance=data["provenance"], checked_endpoints=len(evidence), endpoints=evidence,
                scope="Second CPU implementation of worst endpoint per model/mesh; full owner audit still required. No new solves or timings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=CELL/"runs/staged_accuracy08")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/2026-09-11-poisson-staged-accuracy.coordinator-audit.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.run), indent=2)+"\n")
