#!/usr/bin/env python
"""Positive and negative Phase-12 route/training contracts."""
from __future__ import annotations

import inspect
import json
import os
import pickle
import sys

import numpy as np

assert len(sys.argv) == 3
BUNDLE, STRUCTURAL = map(os.path.abspath, sys.argv[1:])
if len(sys.argv) < 6:
    sys.argv.extend(["synthetic"] * (6 - len(sys.argv)))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import b10_audit_phase12_g2 as audit
import b10_phase12_g2 as p12
import b10_phase11_g2 as p11
import b10_phase9_train as p9


actual_source = inspect.getsource(p12.make_timed_actual_route)
identity_source = inspect.getsource(p12.make_untimed_identity_route)
assert "decode_states_cox_sequential" not in actual_source
assert "make_untimed_identity_route" not in actual_source
assert "decode_states_cox_sequential" in identity_source
assert p12.expected_output_contract(1024, 50)["logical_output_bytes"] == 429230728
assert p12.expected_output_contract(1024, 50)["leaf_count"] == 5
assert audit.structural_corruption_self_test()["pass"]

with open(STRUCTURAL, encoding="utf-8") as handle:
    structural = json.load(handle)
assert audit.structural_check({"structural_preflight": structural}, True)

with open(os.path.join(BUNDLE, "out", "phase12_g2.json"), encoding="utf-8") as handle:
    report = json.load(handle)
arrays = {key: value.copy() for key, value in np.load(
    os.path.join(BUNDLE, "out", "phase12_g2.npz"), allow_pickle=False).items()}
with open(os.path.join(BUNDLE, "out", "checkpoint.pkl"), "rb") as handle:
    checkpoint = pickle.load(handle)
assert report["phase12_execution"]["phase11_training_path_reused_unchanged"]
assert report["decision"]["phase12_pass"] == report["decision"]["phase11_pass"]
assert report["decision"]["phase12_cell_cap"] == 1
assert audit.a11.schedule_check(report, arrays, True)
assert audit.a11.trace_check(arrays, "train_trust", False)["pass"]
assert audit.a11.preflight_check(report, True)
assert audit.a11.decision_check(report, True)["pass"]
assert p9.tree_count(checkpoint["generator"]) == 165954
assert p9.tree_count(checkpoint["encoder"]) == 164384
assert p9.tree_count(checkpoint["predictor"]) == 2533

bad = dict(arrays)
bad["train_trust_jvp_count"] = arrays["train_trust_jvp_count"].copy()
bad["train_trust_jvp_count"][0, 0, 0] += 1
assert not audit.a11.trace_check(bad, "train_trust", False)["pass"]
print("phase12_contracts=pass hidden_control_and_leaf_corruptions=10 training_corruption=1")
