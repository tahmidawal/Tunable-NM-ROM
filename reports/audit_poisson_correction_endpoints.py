"""Independent NumPy/SciPy endpoint checks for the nested Poisson family."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.fft import dstn

from audit_poisson_online_tuning import features, head_with_jacobian, source

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT/"worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def audit(run):
    run = run.resolve()
    out = run/"cluster/out/pilot"
    path = out/"result.json"
    data = json.loads(path.read_text())
    owner_path = run/"audit.json"
    owner = json.loads(owner_path.read_text())
    assert data["complete"] and owner["passed"] and owner["result_sha256"] == sha(path)
    meta, cfg = data["provenance"], data["config"]
    assert meta["backend"] == "gpu" and meta["x64"] and meta["matmul_precision"] == "highest"
    models = {m["id"]: m for m in data["checkpoints"]}
    assert sha(out/"basis.npz") == cfg["basis_sha256"]
    basis = np.load(out/"basis.npz")
    evidence = []
    for method in cfg["trained_model_ids"]:
        base = cfg["method_base_model"][method]
        checkpoint = out/models[base]["path"]
        assert sha(checkpoint) == models[base]["sha256"]
        params = pickle.loads(checkpoint.read_bytes())["params"]
        for n in cfg["intervals"]:
            rows = [r for r in data["rows"] if r["intervals"] == n and r["method"] == method]
            row = max(rows, key=lambda r: r["physical_error"])
            field = np.load(out/"fields"/(row["field_sha256"]+".npz"))["field"]
            assert field.dtype == np.float64 and np.isfinite(field).all() and array_sha(field) == row["field_sha256"]
            refs = np.load(out/"references"/f"n{n}_case{row['case']}.npz")
            error = np.linalg.norm(field-refs["fine"])/np.linalg.norm(refs["fine"])
            delta = np.linalg.norm(refs["coarse"]-refs["fine"])/np.linalg.norm(refs["fine"])
            np.testing.assert_allclose([error, delta, (error+delta)/(1-delta)],
                                       [row["physical_error"], row["reference_delta"], row["conservative_physical_error"]], atol=1e-12, rtol=1e-12)
            setup = next(s for s in data["setup"] if s["method"] == method and s["intervals"] == n)
            modes = np.asarray(setup["mode_indices"])
            rhs = source(n, data["cohort"]["parameters"][row["case"]])
            eigen = (4*n*n*np.sin(np.pi*modes/(2*n))**2).sum(axis=1)
            target = dstn(rhs[1:-1, 1:-1], type=1, norm="ortho")[modes[:, 0]-1, modes[:, 1]-1]/eigen
            cache = np.load(out/f"cache_n{n}_{method}.npz")
            B, C = cache["B"], cache["C"]
            assert array_sha(B) == row["operator_sha256"] and array_sha(C) == row["correction_matrix_sha256"]
            count = cfg["correction_counts"][method]
            if count:
                np.testing.assert_array_equal(C, basis["coefficient_directions"][:, :count])
            h, j = head_with_jacobian(params, np.asarray(row["latent"]))
            J = B@j.T
            A = B@C
            if count:
                Q = np.linalg.qr(A, mode="reduced")[0]
                y = np.linalg.lstsq(A, target-B@h, rcond=None)[0]
                Bp, projected_target = B-Q@(Q.T@B), target-Q@(Q.T@target)
            else:
                y = np.zeros(0)
                Bp, projected_target = B, target
            np.testing.assert_allclose(y, row["correction_coefficients"], rtol=2e-10, atol=2e-11)
            coefficients = h+C@y
            residual = B@coefficients-target
            Jr, rr = Bp@j.T, Bp@h-projected_target
            reconstruction = np.linalg.norm(residual-rr)/(np.linalg.norm(target)+np.linalg.norm(B@h))
            assert reconstruction < cfg["residual_reconstruction_limit"]
            full_gradient = np.r_[J.T@residual, A.T@residual]
            full = np.linalg.norm(full_gradient)/(np.sqrt(np.sum(J*J)+np.sum(A*A))*np.linalg.norm(residual)+1e-300)
            reduced = np.linalg.norm(Jr.T@rr)/(np.linalg.norm(Jr)*np.linalg.norm(rr)+1e-300)
            np.testing.assert_allclose([full, reduced], [row["stationarity"], row["reduced_stationarity"]], rtol=1e-5, atol=2e-10)
            assert bool(full <= cfg["stationarity_tolerance"]) == row["stationary"]
            assert bool(reduced <= cfg["stationarity_tolerance"]) == row["reduced_stationary"]
            assert abs(np.linalg.norm(residual)-row["full_residual"])/max(np.linalg.norm(residual), 1e-300) < 2e-10
            sv = np.linalg.svd(Jr, compute_uv=False)
            rank = int(np.sum(sv > sv[0]*max(Jr.shape)*np.finfo(float).eps))
            assert rank == row["projected_jacobian_rank"]
            np.testing.assert_allclose(sv, row["projected_jacobian_singular_values"], rtol=2e-10, atol=1e-12)
            nearest = int(np.argmin(np.sum((cache["predictions"]-projected_target)**2, axis=1)))
            assert nearest == row["selected_training_code_index"]
            np.testing.assert_array_equal(cache["codes"][nearest], row["initial_latent"])
            h0, _ = head_with_jacobian(params, cache["codes"][nearest])
            np.testing.assert_allclose(cache["predictions"][nearest], Bp@h0, rtol=2e-10, atol=1e-11)
            ij = np.random.default_rng(911122).integers(1, n, (137, 2))
            decoded = features(params, ij/n)@coefficients
            parity = np.linalg.norm(decoded-field[ij[:, 0], ij[:, 1]])/np.linalg.norm(decoded)
            assert parity < 2e-10
            assert not np.any(field[[0, -1], :]) and not np.any(field[:, [0, -1]])
            evidence.append(dict(method=method, intervals=n, case=row["case"], physical_error=float(error),
                                 full_stationarity=float(full), reduced_stationarity=float(reduced),
                                 reconstruction_scaled=float(reconstruction), projected_rank=rank,
                                 decoder_relative_difference=float(parity), checkpoint_sha256=sha(checkpoint),
                                 field_sha256=row["field_sha256"], correction_count=count))
            print(method, n, row["case"], float(error), flush=True)
    return dict(passed=True, source_result_path=str(path.relative_to(ROOT)), source_result_sha256=sha(path),
                owner_audit_sha256=sha(owner_path), source_script_sha256=sha(Path(__file__)),
                shared_numpy_decoder_sha256=sha(ROOT/"reports/audit_poisson_online_tuning.py"), provenance=meta,
                checked_endpoints=len(evidence), endpoints=evidence,
                scope="Worst field per family prefix and mesh: full errors, independently eliminated linear coefficients, full/projected gradients and ranks, supplied-source initialization and sampled decoding. Full owner audit remains required. No new solve or timing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=CELL/"runs/correction_accuracy10")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/2026-09-11-poisson-correction.coordinator-audit.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.run), indent=2)+"\n")
