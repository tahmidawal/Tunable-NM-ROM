#!/usr/bin/env python
"""Run the excluded synthetic retraction audit and require corruption rejection."""
import json
import os
import subprocess
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: test_phase9_terminal_retraction_audit.py SMOKE_DIR PREREG")
smoke = os.path.abspath(sys.argv[1]); prereg = os.path.abspath(sys.argv[2])
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prior = os.path.join(smoke, "PRIOR-FAILED-AUDIT.json")
manifest = os.path.join(smoke, "MANIFEST.sha256")
stdout = os.path.join(smoke, "audit-only.out"); stderr = os.path.join(smoke, "audit-only.err")
with open(prior, "w", encoding="utf-8") as handle:
    json.dump({"status":"fail","checks":{"capacity_independent":False,"full_field_independent":True}}, handle)
open(manifest, "w", encoding="utf-8").close()
with open(stdout, "w", encoding="utf-8") as handle:
    handle.write("jax_backend=gpu\naudit_only=true optimizer_updates=0\n")
open(stderr, "w", encoding="utf-8").close()
report = json.load(open(os.path.join(smoke, "phase9_terminal_recovery.json"), encoding="utf-8"))
command = [sys.executable, os.path.join(root, "b10_audit_phase9_terminal_retraction.py"),
    "--source-json", os.path.join(smoke,"phase9_terminal_recovery.json"),
    "--source-npz", os.path.join(smoke,"phase9_terminal_recovery.npz"),
    "--checkpoint", os.path.join(smoke,"terminal_checkpoint.pkl"),
    "--prior-failed-audit", prior, "--manifest", manifest, "--prereg", prereg,
    "--expected-commit", str(report["provenance"]["commit"]),
    "--expected-job", str(report["provenance"]["slurm_job_id"]),
    "--slurm-out", stdout, "--slurm-err", stderr,
    "--output", os.path.join(smoke,"RETRACTION-AUDIT.json"),
    "--audit-work-npz", os.path.join(smoke,"RETRACTION-AUDIT-WORK.npz"), "--smoke"]
subprocess.run(command, check=True)
audit = json.load(open(os.path.join(smoke,"RETRACTION-AUDIT.json"), encoding="utf-8"))
assert audit["status"] == "pass"
assert audit["checks"]["retraction_negative_test"] == {
    "positive_contract_pass": True, "corruption_count": 11,
    "corruptions_rejected": True, "pass": True}
assert audit["checks"]["base_corruption_test"]["pass"] is True
assert audit["decision"]["terminal_full_field_accepted"] is True
assert audit["decision"]["capacity_retracted"] is True
assert audit["decision"]["capacity_reproducible"] is False
assert audit["decision"]["capacity_accepted"] is False
assert audit["decision"]["capacity_license_complete"] is False
assert audit["decision"]["g2_licensed"] is False
print("phase9_terminal_retraction_synthetic_audit=pass corruptions=20")
