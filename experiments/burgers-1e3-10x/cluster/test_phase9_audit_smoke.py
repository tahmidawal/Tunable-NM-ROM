#!/usr/bin/env python
"""Build an excluded synthetic audit fixture and require full audit PASS."""
import hashlib
import json
import os
import subprocess
import sys

if len(sys.argv)!=3:
    raise SystemExit("usage: test_phase9_audit_smoke.py SMOKE_DIR PREREG")
smoke=os.path.abspath(sys.argv[1]); prereg=os.path.abspath(sys.argv[2])
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()
report=json.load(open(os.path.join(smoke,"result.json"),encoding="utf-8"))
manifest=os.path.join(smoke,"MANIFEST.sha256"); stdout=os.path.join(smoke,"job.out"); stderr=os.path.join(smoke,"job.err")
with open(manifest,"w",encoding="utf-8") as handle:
    handle.write(f"{sha(os.path.join(root,'b10_phase9_train.py'))}  ./code/b10_phase9_train.py\n")
    handle.write(f"{sha(prereg)}  ./code/PHASE-9-PRE-REGISTRATION.md\n")
with open(stdout,"w",encoding="utf-8") as handle: handle.write("jax_backend=gpu\nALL-DONE\n")
open(stderr,"w",encoding="utf-8").close()
command=[sys.executable,os.path.join(root,"b10_audit_phase9_train.py"),"--source-json",os.path.join(smoke,"result.json"),
         "--source-npz",os.path.join(smoke,"result.npz"),"--checkpoint",os.path.join(smoke,"checkpoint.pkl"),
         "--manifest",manifest,"--prereg",prereg,"--expected-commit",str(report["provenance"].get("commit")),
         "--expected-job",str(report["provenance"].get("slurm_job_id")),"--slurm-out",stdout,"--slurm-err",stderr,
         "--output",os.path.join(smoke,"AUDIT.json"),"--smoke"]
subprocess.run(command,check=True)
audit=json.load(open(os.path.join(smoke,"AUDIT.json"),encoding="utf-8"))
assert audit["status"]=="pass" and audit["negative_aware"] is True and audit["checks"]["negative_self_test"]["pass"] is True
print("phase9_smoke_audit=pass")
