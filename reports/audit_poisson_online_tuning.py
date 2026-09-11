"""Independent NumPy/SciPy cross-check of reported Poisson tuning endpoints.

The owner audits every archived field. This second implementation checks each
confirmed ROM configuration's worst-error output on each mesh, its exact weak
residual/gradient, nearest-code initialization, multi-start selection, and the
decoder at deterministic spatial points. No JAX solver is imported or executed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np
from scipy.fft import dstn
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "worktrees/2026-09-07-mr-poisson2d"
CELL = TREE / "experiments/multiresolution-poisson"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def mlp(layers, x):
    for w, b in layers[:-1]:
        x = x @ w + b
        x = x * expit(x)
    return x @ layers[-1][0] + layers[-1][1]


def features(params, xy):
    angle = 2*np.pi*xy @ params["B"]
    ff = np.concatenate((np.sin(angle), np.cos(angle)), axis=-1)
    x, y = xy.T
    bc = params["out_scale"]*16*x*(1-x)*y*(1-y)
    return bc[:, None]*mlp(params["g"], ff)


def head_with_jacobian(params, z):
    x, jac = z.copy(), np.eye(len(z))
    if "hB" in params:
        angle = 2*np.pi*z @ params["hB"]
        x = np.concatenate((z, np.sin(angle), np.cos(angle)))
        jac = np.concatenate((jac, 2*np.pi*params["hB"]*np.cos(angle),
                              -2*np.pi*params["hB"]*np.sin(angle)), axis=1)
    for w, b in params["h"][:-1]:
        x, jac = x @ w+b, jac @ w
        sig = expit(x)
        jac *= sig*(1+x*(1-sig))
        x *= sig
    w, b = params["h"][-1]
    return x @ w+b+z @ params["h_lin"], jac @ w+params["h_lin"]


def source(n, param):
    cx, cy, width, amplitude = param
    x, y = np.meshgrid(np.linspace(0, 1, n+1)[1:-1], np.linspace(0, 1, n+1)[1:-1], indexing="ij")
    return np.pad(amplitude*np.exp(-((x-cx)**2+(y-cy)**2)/(2*width**2)), 1)


def check(run):
    native = run / "cluster/out/pilot"
    result_path = native / "result.json"
    result = json.loads(result_path.read_text())
    submission = json.loads((run / "submission.json").read_text())
    assert result["complete"]
    cfg = result["config"]
    checkpoint = TREE / cfg["checkpoint"]
    assert digest(checkpoint) == cfg["checkpoint_sha256"]
    params = pickle.loads(checkpoint.read_bytes())["params"]
    prov = result["provenance"]
    assert prov["backend"] == "gpu" and prov["x64"] and prov["matmul_precision"] == "highest"
    assert prov["commit"] == submission["source_commit"]
    # Staged names are unique in this job. Compare actual cluster bytes to
    # immutable git objects; do not use the cluster's ancestor git repository.
    source_paths = subprocess.check_output(["git", "-C", str(TREE), "ls-tree", "-r", "--name-only", prov["commit"]], text=True).splitlines()
    for name, expected in submission["source_hashes"].items():
        path = run / "cluster" / name
        assert digest(path) == expected
        candidates = [p for p in source_paths if Path(p).name == path.name]
        matches = [p for p in candidates if hashlib.sha256(subprocess.check_output(["git", "-C", str(TREE), "show", f"{prov['commit']}:{p}"])).hexdigest() == expected]
        assert matches, name
    assert digest(native / "SELECTION.json") == result["selection_sha256"]
    freeze = json.loads((native / "SELECTION.json").read_text())
    confirmed = freeze["confirmed_preset_ids"]
    # Verify the ordering encoded in the record: all screen rows precede any
    # confirmation rows and confirmation uses the identical frozen shortlist.
    phases = [r["phase"] for r in result["rows"]]
    first_confirmation = phases.index("confirmation")
    assert set(phases[:first_confirmation]) == {"screen"}
    assert set(phases[first_confirmation:]) == {"confirmation"}
    evidence, maxima = [], dict(field_error=0., adjusted_error=0., reference_delta=0.,
                               decoder_relative=0., weak_residual_relative=0., stationarity_absolute=0., source_operator_relative=0.)
    for n in cfg["intervals"]:
        for method in confirmed:
            rows = [r for r in result["rows"] if (r["phase"], r["intervals"], r["method"]) == ("confirmation", n, method)]
            assert len(rows) == len(result["cohort"]["parameters"])*cfg["repetitions"]
            row = max(rows, key=lambda r: r["physical_error"])
            case = row["case"]
            field = np.load(native / "fields" / (row["field_sha256"]+".npz"))["field"]
            assert array_digest(field) == row["field_sha256"] and np.isfinite(field).all()
            ref = np.load(native / "references" / f"n{n}_case{case}.npz")
            delta, error = relative(ref["coarse"], ref["fine"]), relative(field, ref["fine"])
            adjusted = (error+delta)/(1-delta)
            for key, difference in [("field_error", abs(error-row["physical_error"])),
                                    ("adjusted_error", abs(adjusted-row["conservative_physical_error"])),
                                    ("reference_delta", abs(delta-row["reference_delta"]))]:
                maxima[key] = max(maxima[key], difference)
                assert difference < 1e-12, (n, method, key, difference)
            rhs = source(n, result["cohort"]["parameters"][case])
            same_case = [r for r in result["rows"] if (r["intervals"], r["case"]) == (n, case)]
            assert {r["source_sha256"] for r in same_case} == {row["source_sha256"]}
            # ARM and x86 NumPy exp implementations can differ by a final bit.
            # Re-generated floating data cannot be required to hash identically
            # across those hosts. Archived bytes are still hash checked; verify
            # the actual source numerically through the paired direct solution.
            direct_row = next(r for r in same_case if r["method"] == "dst")
            direct = np.load(native / "fields" / (direct_row["field_sha256"]+".npz"))["field"]
            assert array_digest(direct) == direct_row["field_sha256"]
            applied = n*n*(4*direct[1:-1, 1:-1]-direct[:-2, 1:-1]-direct[2:, 1:-1]-direct[1:-1, :-2]-direct[1:-1, 2:])
            source_difference = relative(applied, rhs[1:-1, 1:-1])
            maxima["source_operator_relative"] = max(maxima["source_operator_relative"], source_difference)
            assert source_difference < 1e-9, (n, case, source_difference)
            setup = next(s for s in result["setup"] if (s["phase"], s["intervals"], s["requested_modes"]) == ("confirmation", n, row["requested_modes"]))
            modes = np.array(setup["mode_indices"])
            eigenvalues = 4*n*n*np.sin(np.pi*modes/(2*n))**2
            fm = dstn(rhs[1:-1, 1:-1], type=1, norm="ortho")[modes[:, 0]-1, modes[:, 1]-1]/eigenvalues.sum(axis=1)
            cache = np.load(native / f"cache_confirmation_n{n}_M{row['requested_modes']}.npz")
            B = cache["B"]
            assert array_digest(B) == row["operator_sha256"]
            assert len(B) == row["retained_modes"] == len(modes)
            distances = np.sum((cache["predictions"]-fm)**2, axis=1)
            nearest = np.argsort(distances)[:row["preset"]["starts"]]
            selected_residuals = []
            for j, start in enumerate(row["starts"]):
                assert start["selected_training_code_index"] == nearest[j]
                assert np.array_equal(np.asarray(start["initial_latent"]), cache["codes"][nearest[j]])
                z = np.asarray(start["latent"])
                h, dh = head_with_jacobian(params, z)
                residual, jac = B@h-fm, B@dh.T
                residual_norm = float(np.linalg.norm(residual))
                stationarity = float(np.linalg.norm(jac.T@residual)/(np.linalg.norm(jac)*residual_norm+1e-300))
                diffr = abs(residual_norm-start["residual"])/max(residual_norm, 1e-300)
                diffg = abs(stationarity-start["stationarity"])
                maxima["weak_residual_relative"] = max(maxima["weak_residual_relative"], diffr)
                maxima["stationarity_absolute"] = max(maxima["stationarity_absolute"], diffg)
                assert diffr < 2e-10 and diffg < 2e-10, (n, method, j, diffr, diffg)
                assert (stationarity <= cfg["stationarity_tolerance"]) == start["stationary"]
                if start["reason"] == 6:
                    assert stationarity <= row["preset"]["stationarity_stop"]+2e-10
                selected_residuals.append(residual_norm)
            assert int(np.argmin(selected_residuals)) == row["selected_start"]
            assert row["total_attempts"] == sum(s["attempts"] for s in row["starts"])
            assert row["total_jacobians"] == sum(s["jacobians"] for s in row["starts"])
            # Deterministic interior points and all boundary edges.
            ij = np.random.default_rng(70911).integers(1, n, size=(127, 2))
            decoded = features(params, ij/n) @ head_with_jacobian(params, np.asarray(row["latent"]))[0]
            decode_difference = relative(decoded, field[ij[:, 0], ij[:, 1]])
            maxima["decoder_relative"] = max(maxima["decoder_relative"], decode_difference)
            assert decode_difference < 2e-10 and not np.any(field[[0, -1], :]) and not np.any(field[:, [0, -1]])
            assert row["total_seconds"] >= row["fused_device_seconds"] > 0
            evidence.append(dict(intervals=n, method=method, case=case, field_sha256=row["field_sha256"],
                                 physical_error=error, adjusted_error=adjusted, retained_modes=row["retained_modes"],
                                 stationary=row["stationary"], reason=row["reason"], starts_checked=len(row["starts"])))
    return dict(passed=True, scope="Independent worst-case endpoint cross-check; complete-cohort audit remains with the experiment owner.",
                result_path=str(result_path.relative_to(ROOT)), result_sha256=digest(result_path),
                source_script_sha256=digest(Path(__file__)), checkpoint_sha256=digest(checkpoint),
                selection_sha256=digest(native / "SELECTION.json"), scientific_source=prov["commit"], job_id=prov["job_id"],
                checked_endpoints=len(evidence), maximum_differences=maxima, endpoints=evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=CELL / "runs/online_tuning07")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/2026-09-11-poisson-online-tuning.coordinator-audit.json")
    args = parser.parse_args()
    result = check(args.run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "endpoints"}, indent=2))
