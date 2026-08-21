#!/usr/bin/env python
"""Exercise the immutable recovery checkpoint's real normalization schema."""
import copy
import json
import os
import pickle
import sys

import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
original_argv=sys.argv
sys.argv=["schema-import","synthetic","G1","unused.json","unused.npz","unused.pkl"]
import b10_audit_phase10_d_portability as audit
sys.argv=original_argv

if len(sys.argv)!=5:
    raise SystemExit("usage: test_phase10_d_normalization_schema.py CHECKPOINT RECOVERY_NPZ RECOVERY_JSON SOURCE_NPZ")
checkpoint=pickle.load(open(sys.argv[1],"rb"))
report=json.load(open(sys.argv[3],encoding="utf-8"))
names=("coefficient_mean","head_scales","predictor_feature_mean","predictor_feature_scale")
with np.load(sys.argv[2],allow_pickle=False) as data:
    recovery_arrays={name:np.asarray(data[name]) for name in names}
with np.load(sys.argv[4],allow_pickle=False) as data:
    source_arrays={name:np.asarray(data[name]) for name in ("coefficient_mean","head_scales")}

assert "coefficient_mean" not in checkpoint and "head_scales" not in checkpoint
positive=audit.normalization_binding(checkpoint,recovery_arrays,report,source_arrays)
assert positive["pass"] is True and positive["schema_exact"] is True
base={"normalization":{name:np.asarray(checkpoint["normalization"][name]).copy()
                       for name in ("mean","scales","feature_mean","feature_scale")}}
corruptions=[]
one=copy.deepcopy(base); one["normalization"]["mean"][0]+=1e-8
corruptions.append(not audit.normalization_binding(one,recovery_arrays,report,source_arrays)["pass"])
one=copy.deepcopy(base); del one["normalization"]["feature_scale"]
corruptions.append(not audit.normalization_binding(one,recovery_arrays,report,source_arrays)["pass"])
one=copy.deepcopy(report); one["normalization"]["mean_sha256"]="bad"
corruptions.append(not audit.normalization_binding(base,recovery_arrays,one,source_arrays)["pass"])
one={name:value.copy() for name,value in source_arrays.items()}; one["head_scales"][0]+=1e-8
corruptions.append(not audit.normalization_binding(base,recovery_arrays,report,one)["pass"])
assert all(corruptions)
print("phase10_real_normalization_schema=pass corruptions=4")
