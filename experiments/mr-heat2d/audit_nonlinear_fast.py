"""Independent NumPy/SciPy audit of saved heat comparison fields and operators."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import pickle
from scipy.special import expit

import numpy as np
import scipy.linalg


def audit(record):
    archive = record/"archive"; out = archive/"outputs"
    result = json.loads((out/"results.json").read_text())
    assert result["complete"] and result["metadata"]["backend"] == "gpu"
    assert result["metadata"]["x64"] and result["metadata"]["precision"] == "highest"
    log = (archive/f'job-{result["metadata"]["job_id"]}.log').read_text()
    assert "jax_backend=gpu" in log and "NONLINEAR FAST HEAT COMPLETE" in log
    for forbidden in ("Captured constant", "captured constant", "large constant", "RESOURCE_EXHAUSTED", "Out of memory", "No space left", "Traceback"):
        assert forbidden not in log, forbidden
    source = result["source_manifest"]
    for path, expected in source["sha256"].items():
        blob = (archive/path).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == expected
        if path in source["provenance"]:
            origin = source["provenance"][path]
            saved = subprocess.check_output(["git", "show", origin["commit"]+":"+origin["git_path"]])
            assert saved == blob
    checkpoint = pickle.loads((archive/(result["settings"]["checkpoint"]+".pkl")).read_bytes())
    params = checkpoint["params"]
    def mlp(layers, x):
        for w, b in layers[:-1]:
            x = x@w+b; x = x*expit(x)
        w, b = layers[-1]
        return x@w+b
    def head(z): return mlp(params["h"], z)+z@params["h_lin"]
    def head_jacobian(z):
        x = z; jac = np.eye(len(z))
        for w, b in params["h"][:-1]:
            x = x@w+b; sig = expit(x)
            jac = (sig*(1+x*(1-sig)))[:, None]*(w.T@jac)
            x = x*sig
        w, b = params["h"][-1]
        return w.T@jac+params["h_lin"].T
    axis = np.arange(1, 64)/64
    xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    angle = 2*np.pi*(xy@params["B"])
    features = np.concatenate((np.sin(angle), np.cos(angle)), -1)
    mask = 16*xy[:,0]*(1-xy[:,0])*xy[:,1]*(1-xy[:,1])
    sampled_bank = (params["out_scale"]*mask)[:, None]*mlp(params["g"], features)
    manifold_checks = []; projected_residual_checks = []; adaptive_checks = []
    checked = set(); largest_delta = 0.; metric_count = 0
    @lru_cache(maxsize=8)
    def load(path):
        a = np.load(out/path)["field"]
        assert a.dtype == np.float64 and np.isfinite(a).all()
        return a
    def field(spec):
        a = load(spec["path"])
        if spec["path"] not in checked:
            digest = hashlib.sha256(str((a.shape, a.dtype.str)).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
            assert digest == spec["sha256_array"]
            assert list(a.shape) == spec["shape"] and a.dtype.str == spec["dtype"]
            checked.add(spec["path"])
        return a
    def check_metrics(predicted, truth, n, saved):
        nonlocal largest_delta, metric_count
        a = predicted.reshape(len(predicted), -1); b = truth.reshape(len(truth), -1)
        diff = np.linalg.norm(a-b, axis=1); norm = np.linalg.norm(b, axis=1)
        actual = dict(relative_current=diff/np.maximum(norm, 1e-300), relative_initial=diff/max(norm[0], 1e-300),
                      absolute_l2=diff/n, truth_norm_over_initial=norm/max(norm[0], 1e-300),
                      energy=np.sum(a*a, axis=1)/(2*n*n),
                      state_change_from_initial=np.linalg.norm(a-a[0], axis=1)/max(np.linalg.norm(a[0]), 1e-300))
        for key, value in actual.items():
            delta = float(np.max(np.abs(value-np.asarray(saved[key]))))
            largest_delta = max(largest_delta, delta); metric_count += len(value)
            np.testing.assert_allclose(value, saved[key], atol=2e-13, rtol=2e-13)
        assert np.array_equal(norm/max(norm[0], 1e-300)<1e-3, saved["vanished_below_1e-3_initial"])
    max_reference_delta = 0.
    for case in result["cases"]:
        coarse, fine = field(case["reference_coarse"]), field(case["reference_fine"])
        check_metrics(coarse, fine, max(result["settings"]["requested_intervals"]), case["reference_refinement"])
        max_reference_delta = max(max_reference_delta, max(case["reference_refinement"]["relative_current"]))
    refs = {(r["intervals"], r["case"]): r for r in result["case_fields"]}
    linear_rows = {(r["intervals"], r["case"]): r for r in result["rows"] if r["method"] == "linear_weak_exact"}
    counts = {}; summaries = []
    for row in result["rows"]:
        key = row["intervals"], row["case"], row["method"]
        assert key not in counts; counts[key] = len(row["repetitions"])
        assert counts[key] == result["settings"]["timing_repetitions"]
        refs_row = refs[key[:2]]; initial = field(refs_row["initial"])
        truth, discrete = field(refs_row["physical"]), field(refs_row["discrete"])
        for rep in row["repetitions"]:
            pred = field(rep["field"])
            if rep["solver"]:
                zs = np.asarray(rep["solver"]["latents"])
                assert zs.shape == (len(result["config"]["times"]), result["config"]["k"])
                predicted_samples = head(zs)@sampled_bank.T
                stride = row["intervals"]//64
                actual_samples = pred[:, stride-1::stride, stride-1::stride].reshape(pred.shape[0], -1)
                err = float(np.linalg.norm(predicted_samples-actual_samples)/np.linalg.norm(actual_samples))
                assert err < 1e-10
                manifold_checks.append(err)
                if row["method"].startswith("exp_project"):
                    op = np.load(out/"assembly"/f"n{row['intervals']}.npz")
                    coefficient_step = np.load(out/"assembly"/f"projected_n{row['intervals']}.npz")["coefficient_step"]
                    matrix = op["matrix"]
                    for i, saved_info in enumerate(rep["solver"]["steps"]):
                        target = matrix@(coefficient_step@head(zs[i]))
                        scale = max(np.linalg.norm(target), 1e-14)
                        residual = (matrix@head(zs[i+1])-target)/scale
                        jac = matrix@head_jacobian(zs[i+1])/scale
                        residual_norm = float(np.linalg.norm(residual))
                        gradient = float(np.linalg.norm(jac.T@residual)/max(np.linalg.norm(jac),1e-30))
                        np.testing.assert_allclose([residual_norm,gradient], np.asarray(saved_info)[[3,4]],rtol=1e-8,atol=1e-11)
                        projected_residual_checks.append(max(abs(residual_norm-saved_info[3]),abs(gradient-saved_info[4])))
                if "adaptive" in row["method"]:
                    linear = field(linear_rows[row["intervals"],row["case"]]["repetitions"][0]["field"])[0]
                    norm2 = float(np.sum(initial*initial)); projected2 = float(np.sum(linear*linear))
                    first = rep["solver"]["initial_fits"][0]; second = rep["solver"]["initial_fits"][1]
                    first_error2 = (max(norm2-projected2,0)+first[3]**2*projected2)/max(norm2,1e-28)
                    expected_second = first[2] != 1 or first_error2 > result["settings"]["initial_gate"]**2
                    assert expected_second == (second[2] != -1)
                    adaptive_checks.append(dict(intervals=row["intervals"],case=row["case"],method=row["method"],first_error=float(np.sqrt(first_error2)),second_attempted=bool(expected_second)))
            check_metrics(pred, truth, row["intervals"], rep["vs_physical"])
            check_metrics(pred, discrete, row["intervals"], rep["vs_same_grid"])
            phases = rep["phases"]
            assert min(phases.values()) > 0
            assert abs(phases["host_seconds"]-sum(phases[k] for k in ("input_seconds", "device_seconds", "output_seconds"))) < 1e-12
            if row["method"].startswith("fom_"): np.testing.assert_array_equal(pred[0], initial)
            if "refined_field" in rep:
                check_metrics(pred, field(rep["refined_field"]), row["intervals"], rep["vs_half_step"])
    ns = result["settings"]["requested_intervals"]
    assert len(counts) == len(ns)*len(result["cases"])*len(result["settings"].get("methods",list({r["method"] for r in result["rows"]})))
    assert sum(counts.values()) == result["timed_invocations"]
    operator_errors = []
    for n in ns:
        op = np.load(out/"assembly"/f"n{n}.npz")
        # QR least squares is independent of the production SVD-based pinverse.
        r = op["triangular"]; c = scipy.linalg.solve_triangular(r.T, op["matrix"].T, lower=True).T
        q, upper = np.linalg.qr(c, mode="reduced")
        nu = result["config"]["diffusivity"]; lam = op["mode_lam"]; dt = result["settings"]["dt"]
        generator = -nu*scipy.linalg.solve_triangular(upper, q.T@(lam[:, None]*c))
        cn = scipy.linalg.solve_triangular(upper, q.T@(((1-dt*nu*lam/2)/(1+dt*nu*lam/2))[:, None]*c))
        for name, a, b in (("generator", generator, op["generator"]), ("cn_step", cn, op["cn_step"])):
            relative = float(np.linalg.norm(a-b)/np.linalg.norm(b))
            assert relative < 1e-10
            operator_errors.append(dict(intervals=n, operator=name, relative_error=relative))
        assert np.max(scipy.linalg.eigvals(generator).real) < 0
        observation_dt = result["config"]["times"][1]-result["config"]["times"][0]
        raw_step = scipy.linalg.solve_triangular(r, scipy.linalg.expm(observation_dt*generator)@r)
        saved_step = np.load(out/"assembly"/f"projected_n{n}.npz")["coefficient_step"]
        raw_error = float(np.linalg.norm(raw_step-saved_step)/np.linalg.norm(saved_step))
        assert raw_error < 1e-9
        operator_errors.append(dict(intervals=n,operator="raw_exponential_predictor",relative_error=raw_error))
        for name in sorted({r["method"] for r in result["rows"]}):
            rows = [r for r in result["rows"] if r["intervals"] == n and r["method"] == name]
            reps = [rep for row in rows for rep in row["repetitions"]]
            summary = dict(intervals=n, method=name, cases=len(rows), invocations=len(reps),
                worst_physical_error=max(max(r["vs_physical"]["relative_current"]) for r in reps),
                worst_initial_error=max(r["vs_physical"]["relative_current"][0] for r in reps),
                worst_same_grid_error=max(max(r["vs_same_grid"]["relative_current"]) for r in reps),
                largest_energy_increase=max(float(np.max(np.diff(r["vs_physical"]["energy"]))) for r in reps))
            for contract in ("device", "host"):
                key = contract+"_seconds"
                values = [rep["phases"][key] for rep in reps]
                summary[contract+"_median_ms"] = float(np.median(values)*1000)
                summary[contract+"_outliers"] = int(sum(sum(rep["phases"][key]>1.5*np.median([r["phases"][key] for r in row["repetitions"]]) for rep in row["repetitions"]) for row in rows))
            summary["nonstationary_fit_count"] = sum(int(np.any((np.asarray(rep["solver"]["initial_fits"])[:, 2] != 1) & (np.asarray(rep["solver"]["initial_fits"])[:, 2] != -1))) for rep in reps if rep["solver"])
            summary["nonstationary_step_count"] = sum(int(np.sum(np.asarray(rep["solver"]["steps"])[:, 2] != 1)) for rep in reps if rep["solver"])
            summary["total_initial_attempts"] = sum(sum(int(x[0]) for x in rep["solver"]["initial_fits"]) for rep in reps if rep["solver"])
            summary["total_step_attempts"] = sum(sum(int(x[0]) for x in rep["solver"]["steps"]) for rep in reps if rep["solver"])
            summary["skipped_second_initializations"] = sum(int(rep["solver"]["initial_fits"][1][2] == -1) for rep in reps if rep["solver"])
            if name == "linear_weak_cn":
                summary["worst_half_step_current_delta"] = max(max(r["vs_half_step"]["relative_current"]) for r in reps)
            summaries.append(summary)
    prior_out = record.parent/"transfer04/archive/outputs"
    prior = json.loads((prior_out/"results.json").read_text())
    prior_rows = {(r["intervals"], r["case"]): r for r in prior["rows"] if r["method"] == result["settings"]["checkpoint"]}
    nmrom_parity = []
    for row in result["rows"]:
        if row["method"] != "nmrom": continue
        old_rep = prior_rows[row["intervals"], row["case"]]["repetitions"][0]
        b = np.load(prior_out/old_rep["field"]["path"])["field"]
        a = field(row["repetitions"][0]["field"])
        relative = float(np.linalg.norm(a.reshape(-1)-b.reshape(-1))/np.linalg.norm(b))
        assert relative < 1e-9
        nmrom_parity.append(dict(intervals=row["intervals"], case=row["case"], relative_difference=relative))
    output = dict(passed=True, result_sha256=hashlib.sha256((out/"results.json").read_bytes()).hexdigest(),
        metadata=result["metadata"], source_commit=source["source_commit"], settings=result["settings"],
        unique_fields_checked=len(checked), timed_invocations_checked=sum(counts.values()),
        metric_entries_checked=metric_count, maximum_metric_difference=largest_delta,
        maximum_reference_refinement=max_reference_delta, operator_errors=operator_errors,
        outlier_rule="Above 1.5 times the median of the same case/method/mesh repetition group; none excluded.",
        summaries=summaries, prior_nmrom_field_parity=nmrom_parity,
        nonlinear_decoder_checks=len(manifold_checks), maximum_sampled_decoder_error=max(manifold_checks),
        projected_weak_checks=len(projected_residual_checks), maximum_projected_weak_difference=max(projected_residual_checks),
        adaptive_checks=adaptive_checks)
    analysis = record/"analysis"; analysis.mkdir(exist_ok=True)
    (analysis/"audit.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("record", type=Path)
    audit(parser.parse_args().record)
