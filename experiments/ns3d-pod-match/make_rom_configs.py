"""DESIGN §6: freeze a passing bank and write the development ROM configs.

Copies runs/<job>/output/bank_final_ordered.npz to banks/<arm>/bank_selected.npz, and writes
configs/dev{32,64,96}.json = the parent's ns3d-coordnet/configs/dev{n}.json with ONLY the bank
file and its sha256 changed. Refuses if the bank job's gates did not all pass.
Usage: make_rom_configs.py <job-name, e.g. bank_b> <arm, e.g. b>
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
job, arm = sys.argv[1], sys.argv[2]
summary = json.loads((HERE / "runs" / job / "output" / "summary.json").read_text())
if summary.get("status") != "final" or summary["gates"].get("verdict") != "pass":
    raise SystemExit(f"{job}: status {summary.get('status')}, gates {summary.get('gates')} -- no ROM")
src = HERE / "runs" / job / "output" / "bank_final_ordered.npz"
dst = HERE / "banks" / arm / "bank_selected.npz"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst)
sha = hashlib.sha256(dst.read_bytes()).hexdigest()
assert sha == hashlib.sha256(src.read_bytes()).hexdigest()
for n in (32, 64, 96):
    cfg = json.loads((HERE.parent / f"ns3d-coordnet/configs/dev{n}.json").read_text())
    cfg["coordnet_bank_file"] = f"experiments/ns3d-pod-match/banks/{arm}/bank_selected.npz"
    cfg["coordnet_bank_sha256"] = sha
    cfg["source_bank_job"] = job
    (HERE / "configs" / f"dev{n}.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"wrote configs/dev{n}.json  bank sha256 {sha}")
