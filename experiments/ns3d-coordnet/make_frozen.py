"""Freeze the test-cohort settings from the three development mesh jobs.

Copies each mesh's trained coordnet head into frozen/, records sha256 of every frozen
input, and checks that the three development jobs used the same bank file, prefix and
solver settings. Usage: make_frozen.py --jobs dev32=<run> dev64=<run> dev96=<run>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs=3, required=True, help="n=<run dir name> for n in 32 64 96")
    ap.add_argument("--out", type=Path, default=HERE / "frozen")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    keys = ("coordnet_bank_file", "coordnet_bank_sha256", "bank_prefix", "k", "ladder_dt",
            "ladder_iters", "damping", "ic_iters", "modes", "span_ladder")
    heads, common, sources = {}, None, {}
    for item in args.jobs:
        n, run = item.split("=")
        out = HERE / "runs" / run / "output" / "mesh"
        s = json.loads((out / "summary.json").read_text())
        if s.get("status") != "final":
            raise SystemExit(f"{run}: status {s.get('status')} is not final")
        v = json.loads((out / "verify.json").read_text())
        if not v["passed"]:
            raise SystemExit(f"{run}: audit did not pass")
        cfg = s["config"]
        these = {k: cfg.get(k) for k in keys}
        if common is None:
            common = these
        elif these != common:
            raise SystemExit(f"{run}: settings differ from the other meshes: {these} vs {common}")
        src = out / f"coordnet_head_k{cfg['k']}_n{n}.npz"
        dst = args.out / src.name
        shutil.copyfile(src, dst)
        heads[n] = dict(file=dst.name, sha256=sha(dst), source_job=s["job_id"], source_run=run)
        sources[n] = dict(summary_sha256=sha(out / "summary.json"), job=s["job_id"],
                          commit=s["source_commit"])
    frozen = dict(schema="ns3d-coordnet-frozen-v1", heads=heads, sources=sources, **common)
    (args.out / "frozen_settings.json").write_text(json.dumps(frozen, indent=1) + "\n")
    print(json.dumps(frozen, indent=1))


if __name__ == "__main__":
    main()
