#!/usr/bin/env python
"""Build an excluded synthetic audit fixture and require full audit PASS."""
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys

import jax
import numpy as np

if len(sys.argv) not in (3,4):
    raise SystemExit("usage: test_phase9_audit_smoke.py SMOKE_DIR PREREG [--corruptions]")
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
def command(directory):
    return [sys.executable,os.path.join(root,"b10_audit_phase9_train.py"),"--source-json",os.path.join(directory,"result.json"),
         "--source-npz",os.path.join(directory,"result.npz"),"--checkpoint",os.path.join(directory,"checkpoint.pkl"),
         "--manifest",os.path.join(directory,"MANIFEST.sha256"),"--prereg",prereg,"--expected-commit",str(report["provenance"].get("commit")),
         "--expected-job",str(report["provenance"].get("slurm_job_id")),"--slurm-out",os.path.join(directory,"job.out"),"--slurm-err",os.path.join(directory,"job.err"),
         "--output",os.path.join(directory,"AUDIT.json"),"--smoke"]
subprocess.run(command(smoke),check=True)
audit=json.load(open(os.path.join(smoke,"AUDIT.json"),encoding="utf-8"))
assert audit["status"]=="pass" and audit["negative_aware"] is True and audit["checks"]["negative_self_test"]["pass"] is True
if len(sys.argv)==4:
    assert sys.argv[3]=="--corruptions"
    for index,kind in enumerate(("optimizer","feature_normalization","preflight","update_order","history_marker","work_checkpoint")):
        directory=os.path.join(smoke,f"corrupt_{index}"); os.makedirs(directory,exist_ok=True)
        for name in ("result.json","result.npz","checkpoint.pkl","work.pkl","MANIFEST.sha256","job.out","job.err"):
            shutil.copy2(os.path.join(smoke,name),os.path.join(directory,name))
        one=json.load(open(os.path.join(directory,"result.json"),encoding="utf-8"))
        if kind=="optimizer":
            value=pickle.load(open(os.path.join(directory,"checkpoint.pkl"),"rb"))
            def corrupt_leaf(leaf):
                array=np.asarray(leaf).copy()
                if np.issubdtype(array.dtype,np.floating) and array.size: array.reshape(-1)[0]=np.nan
                return array
            value["optimizer_states"]=jax.tree_util.tree_map(corrupt_leaf,value["optimizer_states"])
            pickle.dump(value,open(os.path.join(directory,"checkpoint.pkl"),"wb")); one["checkpoint"]["sha256"]=sha(os.path.join(directory,"checkpoint.pkl"))
        elif kind=="feature_normalization":
            with np.load(os.path.join(directory,"result.npz"),allow_pickle=False) as data: values={name:np.asarray(data[name]) for name in data.files}
            values["predictor_feature_mean"]=values["predictor_feature_mean"].copy(); values["predictor_feature_mean"][0]+=1e-3
            np.savez_compressed(os.path.join(directory,"result.npz"),**values); one["npz"]["sha256"]=sha(os.path.join(directory,"result.npz"))
        elif kind=="preflight": one["preflight"]["projected_terminal_seconds"]+=1.0
        elif kind in ("update_order","history_marker"):
            with np.load(os.path.join(directory,"result.npz"),allow_pickle=False) as data: values={name:np.asarray(data[name]) for name in data.files}
            key="update_resolution_order" if kind=="update_order" else "history_marker"
            values[key]=values[key].copy(); values[key].reshape(-1)[0]+=1
            np.savez_compressed(os.path.join(directory,"result.npz"),**values); one["npz"]["sha256"]=sha(os.path.join(directory,"result.npz"))
        else:
            with open(os.path.join(directory,"work.pkl"),"ab") as handle: handle.write(b"corrupt")
        with open(os.path.join(directory,"result.json"),"w",encoding="utf-8") as handle: json.dump(one,handle)
        assert subprocess.run(command(directory),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0,kind
        failed=json.load(open(os.path.join(directory,"AUDIT.json"),encoding="utf-8"))["checks"]
        expected_check=("preflight" if kind=="preflight" else "schedule" if kind in ("update_order","history_marker")
                        else "work_checkpoint_binding" if kind=="work_checkpoint" else "data_normalization")
        observed=failed[expected_check]
        assert not (observed.get("pass") if isinstance(observed,dict) else observed),(kind,expected_check)
    print("phase9_smoke_corruptions=pass count=6")
print("phase9_smoke_audit=pass")
