"""Second CPU audit of worst reflective-wave trajectories, with spectral energy.

No runner or owner-audit imports. Full saved errors use the discrete sine basis;
sampled decoder velocities use complex-step differentiation of the head value.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def arrays(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def head_value(p, z):
    value = z
    for layer in ("l1", "l2"):
        value = value @ p[f"p/{layer}/w"] + p[f"p/{layer}/b"]
        value = value / (1 + np.exp(-value))
    return (p["p/bias"] + p["p/linear"] @ z
            + p["frozen/output_scale"] * (value @ p["p/out/w"] + p["p/out/b"]))


def bank_values(p, xy):
    angle = 2*np.pi * (xy @ p["frequency"].T)
    value = np.concatenate((2*xy-1, np.sin(angle), np.cos(angle)), axis=1)
    for layer in ("l1", "l2"):
        value = value @ p[f"p/{layer}/w"] + p[f"p/{layer}/b"]
        value = value / (1 + np.exp(-value))
    return (value @ p["p/out/w"] + p["p/out/b"]) * np.sin(np.pi*xy).prod(axis=1)[:, None]


def spectrum(x):
    return dstn(x, type=1, norm="ortho", axes=(-2, -1))


def audit(run):
    run = run.resolve()
    native = run/"cluster/out/pilot"
    inputs = run/"cluster/in/dirichlet"
    path = native/"result.json"
    data = json.loads(path.read_text())
    owner_path = run/"audit.json"
    owner = json.loads(owner_path.read_text())
    assert data["complete"] and not data["final_test_opened"] and owner["passed"]
    assert owner["result_sha256"] == sha(path)
    meta = data["provenance"]
    assert meta["jax_backend"] == "gpu" and meta["x64"] and meta["matmul_precision"] == "highest"
    bank = arrays(inputs/"bank_parameters.npz")
    qr = arrays(inputs/"coordinates.npz")["qr_r"]
    records = []
    for n in data["config"]["meshes"]:
        eigen = 4*n*n*np.sin(np.pi*np.arange(1, n)/(2*n))**2
        lam = eigen[:, None] + eigen[None, :]
        ij = np.random.default_rng(911115).integers(1, n, (137, 2))
        g = np.linalg.solve(qr.T, bank_values(bank, ij/n).T).T
        methods = [a["method"] for a in data["config"]["arms"]]
        for method in methods:
            choices = [r for r in data["invocations"] if r["intervals"] == n and r["method"] == method]
            row = max(choices, key=lambda r: r["same_grid_discrepancy"]["energy_state"]["max_initial_normalized"])
            field_path = native/row["field_artifact"]
            field = arrays(field_path)
            reference = arrays(native/f"reference_{n}_{row['case']}.npz")
            c = row["parameters"][5]
            u, v = field["u"], field["v"]
            ut, vt = reference["u"], reference["v"]
            for name, value in (("u", u), ("v", v)):
                assert value.dtype == np.float64 and np.isfinite(value).all()
                assert hashlib.sha256(value.tobytes()).hexdigest() == row["output_sha256"][name]
            initial2 = np.sum(vt[0]**2 + c*c*lam*spectrum(ut[0])**2)/(n*n)
            # Kinetic terms may be evaluated in physical coordinates by Parseval.
            scales = {"displacement": np.linalg.norm(ut[0])/n,
                      "velocity": np.sqrt(initial2), "energy_state": np.sqrt(initial2)}
            errors = {key: [] for key in scales}
            for uu, vv, tu, tv in zip(u, v, ut, vt):
                du, dv = uu-tu, vv-tv
                errors["displacement"].append(np.linalg.norm(du)/n/scales["displacement"])
                errors["velocity"].append(np.linalg.norm(dv)/n/scales["velocity"])
                energy2 = np.sum(dv*dv + c*c*lam*spectrum(du)**2)/(n*n)
                errors["energy_state"].append(np.sqrt(energy2)/scales["energy_state"])
            for key, value in errors.items():
                np.testing.assert_allclose(value, row["same_grid_discrepancy"][key]["initial_normalized"], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(scales[key], row["same_grid_discrepancy"][key]["initial_scale"], rtol=1e-12)
            omega = c*np.sqrt(lam)
            a0, b0 = spectrum(reference["u0"]), spectrum(reference["v0"])
            reference_max = 0.
            for idx in sorted({0, len(u)//2, len(u)-1, int(np.argmax(errors["energy_state"]))}):
                t = idx*data["config"]["observation_dt"]
                co, si = np.cos(omega*t), np.sin(omega*t)
                ru = idstn(a0*co+b0*si/omega, type=1, norm="ortho")
                rv = idstn(b0*co-a0*omega*si, type=1, norm="ortho")
                reference_max = max(reference_max, float(np.max(abs(ru-ut[idx]))), float(np.max(abs(rv-vt[idx]))))
            assert reference_max < 1e-10
            if method.startswith("trained_"):
                filename = data["config"].get("frozen_head_inputs", {}).get(method)
                head_path = inputs/filename if filename else native/f"head_{method}.npz"
            else:
                head_path = inputs/"head32.npz"
            head = arrays(head_path)
            coeff = np.asarray([head_value(head, z) for z in field["rollout_z"]])
            velocity_coeff = np.asarray([np.imag(head_value(head, z+1e-30j*w))/1e-30
                                         for z, w in zip(field["rollout_z"], field["rollout_w"])])
            saved_u, saved_v = u[:, ij[:, 0]-1, ij[:, 1]-1], v[:, ij[:, 0]-1, ij[:, 1]-1]
            decoder_max = max(float(np.max(abs(coeff@g.T-saved_u))), float(np.max(abs(velocity_coeff@g.T-saved_v))))
            assert decoder_max < 1e-9
            records.append(dict(intervals=n, method=method, case=row["case"], cohort=row["cohort"],
                                fixed_initial_worst={key: float(max(value)) for key, value in errors.items()},
                                complex_step_decoded_uv_max_absolute=decoder_max,
                                independent_modal_reference_max_absolute=reference_max,
                                field_artifact_sha256=sha(field_path), checkpoint_sha256=sha(head_path)))
            print(n, method, row["case"], records[-1]["fixed_initial_worst"], flush=True)
            del field, reference, u, v, ut, vt
    return dict(passed=True, source_result_sha256=sha(path), owner_audit_sha256=sha(owner_path),
                source_script_sha256=sha(Path(__file__)), provenance=meta,
                source_result_path=str(path.relative_to(ROOT)), checked_trajectories=len(records), endpoints=records,
                scope="Worst energy-state trajectory per ROM arm/mesh: all saved output errors with spectral energy; selected modal-reference times; sampled decoder values and complex-step tangent velocities. Full owner audit remains required. No new solves or timings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=CELL/"runs/accel12")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/2026-09-11-wave-accuracy.coordinator-audit.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.run), indent=2)+"\n")
