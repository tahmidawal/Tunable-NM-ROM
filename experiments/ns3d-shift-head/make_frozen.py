"""Freeze every held-out setting from one development job, by script, before opening.

Usage: make_frozen.py <dev attempt_job>
Writes frozen/frozen_settings.json (+ copies of the head checkpoint, the rotation and
the bank probe) and configs/heldout<n>.json. Commit both before submitting the held-out
job; the job refuses to start without them and records their sha256.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELDOUT_SEED = 202609221


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    job = sys.argv[1]
    src = HERE / "runs" / job / "output"
    s = json.loads((src / "summary.json").read_text())
    if s.get("heldout_opened"):
        raise RuntimeError("that job is itself a held-out job")
    cfg = s["config"]
    k = int(s["k_selected"])
    frozen_dir = HERE / "frozen"
    if (frozen_dir / "frozen_settings.json").exists():
        raise RuntimeError("frozen settings already exist; they are written once")
    frozen_dir.mkdir()
    for name in (f"head_k{k}.npz", "rotation.npz", "bank_probe.npz"):
        shutil.copy2(src / name, frozen_dir / name)
    frozen = dict(
        schema="ns3d-shift-head-frozen-v1",
        source_job=s["job_id"], source_commit=s["source_commit"], source_attempt=job,
        source_summary_sha256=sha(src / "summary.json"),
        n=cfg["n"], rank=cfg["rank"], modes=cfg["modes"], k=k, q_ladder=[0],
        span_ladder=[int(x) for x in cfg["span_ladder"]],
        dt=float(cfg["ladder_dt"]), iters=int(cfg["ladder_iters"]),
        damping=float(cfg["damping"]), ic_iters=int(cfg["ic_iters"]), width=cfg["width"],
        head_file=f"head_k{k}.npz", rotation_file="rotation.npz",
        bank_probe_file="bank_probe.npz",
        file_sha256={name: sha(frozen_dir / name) for name in
                     (f"head_k{k}.npz", "rotation.npz", "bank_probe.npz")},
        cnab2_steps=cfg["cnab2_steps"], fd_steps=cfg["fd_steps"], fd_rtols=cfg["fd_rtols"],
        fd_fine=cfg.get("fd_fine", []),
        selection_rule=("k: smallest development evolved worst at q=0 on the ladder setting, "
                        "within 1 % relative the smaller k; ladder setting and R' ladder "
                        "fixed in DESIGN.md before any job"),
        heldout_seed=HELDOUT_SEED, heldout_cases=32)
    (frozen_dir / "frozen_settings.json").write_text(json.dumps(frozen, indent=2) + "\n")
    held = dict(cfg)
    held.pop("smoke", None)
    held.update(name=f"heldout{cfg['n']}", heldout=True, dev_seed=HELDOUT_SEED, dev_cases=32,
                closed_seeds=[202609203, 202609211],
                disjoint_against=[[202609202, 16], [202609203, 32], [202609211, 32]],
                stream_audit=True,
                question="single held-out evaluation of the frozen development setting")
    (HERE / "configs" / f"heldout{cfg['n']}.json").write_text(json.dumps(held, indent=2) + "\n")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
