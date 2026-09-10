"""Recompute the native replay summaries and audit saved timed fields."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HISTORICAL = ROOT / "worktrees/2026-08-29-b2d-tensor/experiments/separable-decoder/runs/b2dtensor"


def check_close(a, b, label, checks, atol=2e-12):
    diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert diff <= atol, (label, diff, atol)
    checks.append(dict(label=label, absolute_difference=diff, tolerance=atol))


def outliers(raw):
    raw = np.asarray(raw, dtype=np.float64)
    return int(np.count_nonzero(raw > 2 * np.median(raw)))


def main():
    run = Path(sys.argv[1]).resolve()
    checks, rows, files = [], [], []
    jobs, gpus = set(), set()
    for n in (64, 256, 512, 1024):
        file = run / "out" / f"n{n}" / f"sep_b2d_tensor_n{n}.json"
        data = json.loads(file.read_text())
        old = json.loads((HISTORICAL / f"n{n}/out/sep_b2d_tensor_n{n}.json").read_text())
        files.append(dict(path=str(file.relative_to(run)), sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        config = data["config"]
        jobs.add(config["slurm_job"])
        gpus.add(config["gpu"])
        assert data["complete"] and config["backend"] == "gpu" and config["x64"]
        assert config["matmul_precision"] == "highest"
        for key in ("N", "k", "r", "M", "m_nnls", "n_test", "seed", "data_seed", "test_seed",
                    "n_train_traj", "n_val_traj", "step_tol", "stall", "extrap", "tr_factor",
                    "gn_budget", "ic_enc_budget", "enc_steps", "num_steps", "dt", "newton_tols", "lin_fracs"):
            assert config[key] == old["config"][key], (n, key, config[key], old["config"][key])
        for arm, variant in data["variants"].items():
            per_time = np.asarray([r["per_time"] for r in variant["per_traj"]])
            check_close(per_time.mean(), variant["err_traj_rel_mean"], f"N{n}:{arm}:mean", checks)
            pair = data["matched"]["arms"][arm]["paired"]
            median_a = np.median([np.median(p["a_raw_ms"]) for p in pair["per_traj"]])
            median_b = np.median([np.median(p["b_raw_ms"]) for p in pair["per_traj"]])
            check_close(median_a, pair["rom_ms"], f"N{n}:{arm}:paired ROM", checks)
            check_close(median_b, pair["fom_ms"], f"N{n}:{arm}:paired FOM", checks)
            check_close(median_b / median_a, pair["speedup"], f"N{n}:{arm}:ratio", checks)
            paired_rom = np.asarray([p["timed_output_errors"]["a"]["per_time"] for p in pair["per_traj"]])
            paired_fom = np.asarray([p["timed_output_errors"]["b"]["per_time"] for p in pair["per_traj"]])
            check_close(paired_rom, per_time, f"N{n}:{arm}:paired/main errors", checks)
            old_per_time = np.asarray([r["per_time"] for r in old["variants"][arm]["per_traj"]])
            rows.append(dict(N=n, arm=arm, paired_rom_ms=float(median_a), paired_fom_ms=float(median_b),
                             paired_fom_over_rom=float(median_b / median_a),
                             mean_error=float(paired_rom.mean()), median_error=float(np.median(paired_rom)),
                             worst_error=float(paired_rom.max()), paired_fom_mean_error=float(paired_fom.mean()),
                             paired_fom_worst_error=float(paired_fom.max()),
                             latent_ms=float(np.median([t * 1e3 for p in variant["per_traj"] for t in p["raw_s"]["roll"]])),
                             rom_outliers=sum(outliers(p["a_raw_ms"]) for p in pair["per_traj"]),
                             fom_outliers=sum(outliers(p["b_raw_ms"]) for p in pair["per_traj"]),
                             historical_per_time_max_difference=float(np.max(np.abs(per_time - old_per_time))),
                             blowups=variant["n_blowups"], stop_reasons=variant["stop_reasons"],
                             matched=data["matched"]["arms"][arm]["matched"]))
        for rung in data["fom"]:
            check_close(np.mean([np.mean(p["per_time"]) for p in rung["per_traj"]]),
                        rung["err_traj_rel_mean"], f"N{n}:FOM:{rung['newton_tol']}:{rung['lin_tol']}", checks)
        fields_file = file.with_name(file.stem + "_tensor_full_case0.npz")
        with np.load(fields_file) as fields:
            assert fields["case"] == 0 and fields["N"] == n
            truth = fields["truth"]
            assert truth.dtype == np.float64 and truth.shape == (51, n * n)
            reference_norm = np.linalg.norm(truth, axis=1)
            for side, name in (("a", "rom"), ("b", "fom")):
                field = fields[name]
                assert field.dtype == np.float64 and np.all(np.isfinite(field))
                error = np.linalg.norm(field - truth, axis=1) / reference_norm
                reported = data["matched"]["arms"]["tensor"]["paired"]["per_traj"][0]["timed_output_errors"][side]["per_time"]
                check_close(error, reported, f"N{n}:full timed case0:{name}", checks)
        files.append(dict(path=str(fields_file.relative_to(run)), sha256=hashlib.sha256(fields_file.read_bytes()).hexdigest()))
    assert len(jobs) == 1 and len(gpus) == 1, (jobs, gpus)
    job_log = (run / "logs" / f"{next(iter(jobs))}.out").read_text()
    assert "jax_backend=gpu" in job_log and "ALL-DONE" in job_log
    for log in (run / "logs").glob("*.err"):
        text = log.read_text().lower()
        assert not any(term in text for term in (
            "captured", "out of memory", "resource_exhausted", "traceback", "no space left",
            "cuda_error", "cuinit")), (log, text)
    report = dict(job=next(iter(jobs)), gpu=next(iter(gpus)), rows=rows, checks=checks,
                  source_files=files, outlier_rule="retained repetition >2 times its own case median",
                  full_field_audit="predeclared case0 only; all other cases retain native timed-output errors")
    (run / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(job=report["job"], checks=len(checks), rows=rows), indent=2))


if __name__ == "__main__":
    main()
