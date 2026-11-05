"""Write configs/q{32,64,96}.json (DESIGN.md sections 2-7). Hashes are recomputed from the files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LANE = "experiments/ns3d-offmesh-quadrature"
PM = "experiments/ns3d-pod-match"
DEVDIR = {32: "dev32b", 64: "dev64", 96: "dev96"}
WALL = {32: "00:40:00", 64: "01:00:00", 96: "01:30:00"}


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    rules = json.loads((HERE / "rules" / "rules.json").read_text())["rules"]
    fams = ["lat", "kor", "sob", "gl", "trap"]
    ladder = {f: sorted([k for k, v in rules.items() if v["family"] == f and v["role"] != "must-fail control"],
                        key=lambda k: rules[k]["m"]) for f in fams}
    for f in fams:
        assert len(ladder[f]) == 8, (f, ladder[f])
    for n in (32, 64, 96):
        bank = f"{PM}/banks/a2_mu001/bank_selected.npz"
        head = f"{PM}/runs/{DEVDIR[n]}/output/mesh/coordnet_head_k8_n{n}.npz"
        refs = f"{PM}/runs/{DEVDIR[n]}/output/mesh/summary.json"
        cfg = dict(
            name=f"q{n}", n=n, dt_truth=0.001, horizon=0.2, modes=292, check_modes=8, k=8,
            ic_iters=12, ladder_dt=0.02, ladder_iters=3, damping=1e-6,
            cnab2_steps=[200, 100, 80, 70, 60, 50, 40, 20, 10], timing_case=0, timing_repetitions=7,
            neighbour_gate_ratio=1.1, target_relative=0.05, eval_seed=202609202, eval_cases=16,
            closed_seeds=[202609203, 202609211, 202609221], sample_chunk_eval=32768,
            orthonormality_abort=1.0 if n == 32 else 0.05,
            bank_file=bank, bank_sha256=sha(bank), head_file=head, head_sha256=sha(head),
            reference_summary=refs, reference_summary_sha256=sha(refs),
            rules_file=f"{LANE}/rules/rules.npz", rules_sha256=sha(f"{LANE}/rules/rules.npz"),
            families=fams, ladder=ladder, must_fail="lat64", integrands=["net", "interp"],
            continuum_p=[128, 96], continuum_seeds=[1, 2], control_shift_seed=3,
            derivative_points=4096, sentinel_rule="lat4096", wall=WALL[n],
            smoke=dict(eval_cases=2, families=fams,
                       ladder=dict(lat=["lat512", "lat4096"], kor=["kor509"], sob=["sob512"],
                                   gl=["gl8"], trap=["trap8"]),
                       continuum_p=[48, 40], cnab2_steps=[200, 20], timing_repetitions=2,
                       derivative_points=512))
        (HERE / "configs" / f"q{n}.json").write_text(json.dumps(cfg, indent=1) + "\n")
        print(n, cfg["head_sha256"][:12], cfg["reference_summary_sha256"][:12], cfg["rules_sha256"][:12])


if __name__ == "__main__":
    main()
