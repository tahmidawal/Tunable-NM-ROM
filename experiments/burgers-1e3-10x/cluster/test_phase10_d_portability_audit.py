#!/usr/bin/env python
"""Check the excluded synthetic Phase10 portability audit."""
import json,os,subprocess,sys
if len(sys.argv)!=4: raise SystemExit("usage: test_phase10_d_portability_audit.py SMOKE_DIR MANIFEST PREREG")
smoke=os.path.abspath(sys.argv[1]); manifest=os.path.abspath(sys.argv[2]); prereg=os.path.abspath(sys.argv[3])
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report=json.load(open(os.path.join(smoke,"phase10_d.json"),encoding="utf-8"))
audit_commit=subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip()
stdout=os.path.join(smoke,"portability-audit.out"); stderr=os.path.join(smoke,"portability-audit.err")
with open(stdout,"w",encoding="utf-8") as handle: handle.write("jax_backend=gpu\naudit_only=true optimizer_updates=0\n")
open(stderr,"w",encoding="utf-8").close()
command=[sys.executable,os.path.join(root,"b10_audit_phase10_d_portability.py"),
    "--r2-bundle",smoke,"--manifest",manifest,"--prereg",prereg,
    "--expected-source-commit",str(report["provenance"]["commit"]),
    "--expected-source-job",str(report["provenance"]["slurm_job_id"]),
    "--expected-audit-commit",audit_commit,"--expected-audit-job","local",
    "--slurm-out",stdout,"--slurm-err",stderr,"--output",os.path.join(smoke,"PORTABILITY-AUDIT.json"),"--smoke"]
subprocess.run(command,check=True)
audit=json.load(open(os.path.join(smoke,"PORTABILITY-AUDIT.json"),encoding="utf-8"))
assert audit["status"]=="pass" and audit["audit_only"] is True
assert audit["checks"]["accepted_initial_control"] is True
assert audit["checks"]["repeated_full_field"] is True
assert audit["checks"]["trust_trace"]["termination_portable"] is True
assert audit["checks"]["negative_self_test"]["pass"] is True
assert audit["checks"]["negative_self_test"]["termination_beyond_band_rejected"] is True
assert audit["decision"]["optimizer_updates"]==0 and audit["decision"]["g2_licensed"] is False
assert audit["checks"]["negative_self_test"]["normalization_corruption_count"]==4
assert audit["checks"]["negative_self_test"]["normalization_corruptions_rejected"] is True
print("phase10_portability_synthetic_audit=pass corruptions=13")
