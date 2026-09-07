"""Follow-up to the retained failed compact-bump resolution trial.

Original analytic controls are extended; a pre-training smoother Gaussian-core
compact family is verified separately. This does not overwrite the failed trial.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.fft import dstn, idstn
import jax.numpy as jnp
from fresh_fom import Grid, integrate_balance, energy, localized_initial, damping_ratio, provenance
from fresh_verify import plane_pulse, fft_open_reference, norm


def sine_reference(u0, v0, c, time, n, semidiscrete=False):
    uu, vv = dstn(u0, type=1, norm="ortho"), dstn(v0, type=1, norm="ortho")
    modes = np.arange(1, n)
    one = 4*n*n*np.sin(np.pi*modes/(2*n))**2 if semidiscrete else (np.pi*modes)**2
    omega = c*np.sqrt(one[:, None]+one[None, :])
    co, si = np.cos(time*omega), np.sin(time*omega)
    return idstn(co*uu+si/omega*vv, type=1, norm="ortho"), idstn(-omega*si*uu+co*vv, type=1, norm="ortho")


def family_check(parameters, meshes, name, out):
    rows, arrays, gates = [], {}, {}
    for bc in ("dirichlet", "absorbing"):
        for pi, p in enumerate(parameters):
            arrays = {}
            solutions = {}
            for n in meshes:
                print("family", name, bc, pi, n, flush=True)
                grid = Grid(n, bc, bc)
                u0, v0 = localized_initial(grid, p)
                stride = int(np.ceil(.05/(.12*grid.h/p[5])))
                dt = .05/stride
                u, v, q = map(np.asarray, integrate_balance(u0, v0, p[5], dt, grid=grid, steps=48*stride, stride=stride))
                ee = np.asarray(energy(jnp.asarray(u), jnp.asarray(v), grid, p[5]))
                finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(q)))
                balance = float(np.max(abs(ee+q-ee[0]))/ee[0])
                if bc == "absorbing":
                    inv = np.sum(grid.mass()*(v+np.asarray(damping_ratio(grid, p[5]))*u), axis=(-2,-1))
                    inv_drift = float(np.max(abs(inv-inv[0])))
                else:
                    inv_drift = 0.
                key = f"{name}_{bc}_{pi}_{n}"
                gates[key+"_finite"] = finite
                gates[key+"_balance"] = balance < 1e-5
                gates[key+"_invariant"] = inv_drift < 1e-10
                row = {"family": name, "bc": bc, "parameter_index": pi, "parameters": list(p), "n": n, "dt": dt, "finite": finite, "balance_relative": balance, "invariant_drift": inv_drift, "initial_energy": float(ee[0]), "final_energy_fraction": float(ee[-1]/ee[0])}
                if bc == "dirichlet":
                    errors = []
                    for ti in (7, 24, 48):
                        su, sv = sine_reference(np.asarray(u0), np.asarray(v0), p[5], ti*.05, n, semidiscrete=True)
                        ce = np.sqrt(max(0., float(energy(jnp.asarray(u[ti]-su), jnp.asarray(v[ti]-sv), grid, p[5])))/ee[0])
                        tu, tv = sine_reference(np.asarray(u0), np.asarray(v0), p[5], ti*.05, n)
                        se = np.sqrt(max(0., float(energy(jnp.asarray(u[ti]-tu), jnp.asarray(v[ti]-tv), grid, p[5])))/ee[0])
                        errors.append({"time": ti*.05, "semidiscrete_state_error": float(ce), "continuum_sine_state_error": float(se)})
                    row["independent_sine"] = errors
                    gates[key+"_semidiscrete"] = max(x["semidiscrete_state_error"] for x in errors) < 2e-4
                rows.append(row)
                solutions[n] = (grid, u, v, ee)
                arrays[key+"_u"], arrays[key+"_v"] = u[[0,7,24,48]], v[[0,7,24,48]]
                arrays[key+"_energy"], arrays[key+"_flux"] = ee, q
            for coarse, fine in zip(meshes[:-1], meshes[1:]):
                g, u, v, ee = solutions[coarse]
                _, uf, vf, _ = solutions[fine]
                ratio = fine//coarse
                sl = slice(ratio-1, -1, ratio) if bc == "dirichlet" else slice(None, None, ratio)
                ur, vr = uf[:, sl, sl], vf[:, sl, sl]
                errors = np.sqrt(np.maximum(0., np.asarray(energy(jnp.asarray(u-ur), jnp.asarray(v-vr), g, p[5])))/ee[0])
                rows.append({"family": name, "bc": bc, "parameter_index": pi, "coarse_n": coarse, "fine_n": fine, "state_difference_array": errors.tolist(), "max_energy_state_difference": float(errors.max()), "final_energy_state_difference": float(errors[-1])})
                worst = int(np.argmax(errors))
                arrays[f"difference_{coarse}_{fine}_worst_index"] = worst
                arrays[f"difference_{coarse}_{fine}_coarse_u"] = u[worst]
                arrays[f"difference_{coarse}_{fine}_coarse_v"] = v[worst]
                arrays[f"difference_{coarse}_{fine}_fine_u"] = uf[worst]
                arrays[f"difference_{coarse}_{fine}_fine_v"] = vf[worst]
            np.savez_compressed(out/(f"{name}_{bc}_{pi}_arrays.npz"), **arrays)
    return rows, gates


def fft_self_check(parameters):
    rows = []
    for pi, p in enumerate(parameters):
        for time in (.35, 1.2, 2.4):
            u128, v128 = fft_open_reference(128, p, time, 4)
            u256, v256 = fft_open_reference(256, p, time, 4)
            u6, v6 = fft_open_reference(128, p, time, 6)
            grid = Grid(128, "absorbing", "absorbing")
            u0, v0 = localized_initial(grid, p)
            e0 = float(energy(u0, v0, grid, p[5]))
            def error(u, v):
                return float(np.sqrt(max(0., float(energy(jnp.asarray(u), jnp.asarray(v), grid, p[5])))/e0))
            uscale = norm(np.asarray(u0),grid)
            du_mesh,du_box=u128-u256[::2,::2],u128-u6
            rows.append({"parameter_index": pi, "time": time, "resolution_difference": error(du_mesh, v128-v256[::2, ::2]), "domain_difference": error(du_box, v128-v6),"resolution_l2_difference":norm(du_mesh,grid)/uscale,"domain_l2_difference":norm(du_box,grid)/uscale,"resolution_mean_difference":float(abs(np.sum(grid.mass()*du_mesh)))/uscale,"domain_mean_difference":float(abs(np.sum(grid.mass()*du_box)))/uscale})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = {"provenance": provenance(), "supersedes": "No data overwritten; initial compact-only family remains a failed coarse-grid trial."}
    if result["provenance"]["jax_backend"] != "gpu":
        raise RuntimeError("GPU required")
    gates = {}
    for kind in ("reflective", "absorbing"):
        print("extended plane", kind, flush=True)
        data = plane_pulse(kind, (128, 256, 512))
        result["plane_"+kind] = data
        gates["plane_"+kind+"_fine_order"] = data["joint_orders"][-1] >= 1.5
    original = [(.5, .49, .27, .27, 1., 1.15, .5, -.5)]
    result["original_family"], original_gates = family_check(original, (128, 256), "original_compact", out)
    gates.update(original_gates)
    # Proposed smoother family extremes, declared before any bank/head training.
    smooth = [(.5, .49, .36, .36, 1., 1.15, .5, -.5, .12, .12), (.48, .51, .42, .42, .9, .85, 0., 0., .16, .16)]
    result["smooth_family"], smooth_gates = family_check(smooth, (32, 64, 128, 256), "gaussian_compact", out)
    gates.update(smooth_gates)
    result["fft_self_check"] = fft_self_check(smooth)
    gates["fft_resolution"] = max(r["resolution_difference"] for r in result["fft_self_check"]) < 1e-3
    gates["fft_domain"] = max(r["domain_difference"] for r in result["fft_self_check"]) < 1e-3
    fine = [r for r in result["smooth_family"] if r.get("coarse_n") == 128]
    # Physical reference error target fixed before model training. Coarse-fine
    # difference <.015 implies an estimated coarse error <.02 at second order.
    gates["smooth_128_reference_difference"] = max(r["max_energy_state_difference"] for r in fine) <= .015
    result["gates"], result["passed"] = gates, all(gates.values())
    (out/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"gates": gates, "passed": result["passed"]}), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
