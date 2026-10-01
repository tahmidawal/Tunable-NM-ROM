"""Write the mesh-job configs (development or test) for one frozen coordnet bank.

The POD reference arm is the paper's frozen head per mesh; its development errors in the
parent jobs are copied in as the reproduction reference (development mode only).

Usage:
  make_configs.py --bank <bank_selected.npz, repo-relative> --mode dev  --out configs/
  make_configs.py --bank <...> --mode test --out configs/
  make_configs.py --bank <...> --mode smoke --out configs/
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

POD_FROZEN = {32: "experiments/ns3d-operators/frozen/h32",
              64: "experiments/ns3d-operators/frozen/h64",
              96: "experiments/ns3d-shift-head/frozen"}
PARENT_JOB = {32: ("a2_h32", "4198840"), 64: ("a2_h64", "4198090"), 96: ("a3_h96", "4198101")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base(n):
    return {
        "n": n, "dt_truth": 0.001, "horizon": 0.2,
        "train_seed": 202609201, "train_cases": 128, "head_train_cases": 512,
        "head_frame_every": 10, "val_every": 8,
        "modes": 292, "check_modes": 8, "gram_block": 64,
        "k": 8, "width": 256, "head_seed": 20260923, "head_steps": 100000,
        "head_log_every": 5000, "head_lr": 0.001, "head_floor_lr": 1e-05, "code_reg": 1e-06,
        "ic_iters": 12, "ladder_dt": 0.02, "ladder_iters": 3, "damping": 1e-06,
        "lm_budget": 60, "lm_gtol": 1e-07,
        "cnab2_steps": [200, 100, 80, 70, 60, 50, 40, 20, 10],
        "timing_case": 0, "timing_repetitions": 7, "neighbour_gate_ratio": 1.10,
        "target_relative": 0.05, "one_bank_orthonormality_gate": 0.05,
        "span_ladder": [256, 128, 64, 48, 32, 16, 8],
        "sample_chunk_eval": 32768,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="repo-relative path of bank_selected.npz")
    ap.add_argument("--mode", choices=("dev", "test", "smoke"), required=True)
    ap.add_argument("--out", type=Path, default=HERE / "configs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    bank_sha = sha(ROOT / args.bank)
    meshes = (32, 64, 96) if args.mode != "smoke" else (32,)
    for n in meshes:
        cfg = base(n)
        cfg["coordnet_bank_file"] = args.bank
        cfg["coordnet_bank_sha256"] = bank_sha
        cfg["pod_frozen_dir"] = POD_FROZEN[n]
        cfg["pod_frozen_sha256"] = {f: sha(ROOT / POD_FROZEN[n] / f)
                                    for f in ("head_k8.npz", "bank_probe.npz")}
        if args.mode in ("dev", "smoke"):
            cfg["eval_seed"], cfg["eval_cases"] = 202609202, 16
            cfg["closed_seeds"] = [202609203, 202609211, 202609221]
            job, jid = PARENT_JOB[n]
            src = ROOT / f"experiments/ns3d-shift-head/runs/{job}/output/summary.json"
            s = json.loads(src.read_text())
            cfg["reference_source"] = dict(job=job, slurm_id=jid, summary=str(src.relative_to(ROOT)),
                                           sha256=sha(src))
            cfg["reference_errors"] = {
                "pod_head_k8": s["frontier"]["k8_q0_dt0.02_it3"]["errors"],
                "pod_linear_R64": s["linear"]["lin_dt0.02_it3"]["errors"]}
        else:
            cfg["eval_seed"], cfg["eval_cases"] = 202609221, 32
            cfg["closed_seeds"] = [202609203, 202609211]
            cfg["disjoint_against"] = [[202609202, 16], [202609203, 32], [202609211, 32]]
            # the paper's POD head on this cohort, from the jobs that opened it before
            if n == 96:
                src = ROOT / "experiments/ns3d-shift-head/runs/b2_heldout96/output/summary.json"
                errs = json.loads(src.read_text())["frontier"]["k8_q0_dt0.02_it3"]["errors"]
                cfg["reference_source"] = dict(job="b2_heldout96", slurm_id="4202872",
                                               summary=str(src.relative_to(ROOT)), sha256=sha(src))
            else:
                src = ROOT / f"experiments/t2-ns3d-test/configs/t2test{n}.json"
                c2 = json.loads(src.read_text())
                errs = c2["reference_values"]["nmrom_accurate_head_k8"]
                cfg["reference_source"] = dict(job=f"ns3d-test t{n}a via t2-ns3d-test config",
                                               summary=c2["reference_summary"],
                                               summary_sha256=c2["reference_summary_sha256"],
                                               config=str(src.relative_to(ROOT)), sha256=sha(src))
            cfg["reference_errors"] = {"pod_head_k8": errs}
        if args.mode == "smoke":
            cfg.update({"eval_cases": 2, "head_train_cases": 16, "head_steps": 200,
                        "head_log_every": 100, "cnab2_steps": [50, 20], "timing_repetitions": 2,
                        "reference_errors": {}, "span_ladder": [4, 2], "k": 2})
            cfg["reference_errors"] = {}
        name = f"{args.mode}{n}{args.tag}"
        cfg["name"] = name
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=1) + "\n")
        print(path, sha(path))


if __name__ == "__main__":
    main()
