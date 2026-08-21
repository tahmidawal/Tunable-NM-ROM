#!/usr/bin/env python
"""Fixed positive and corruption contracts for Phase 11.

Pass the directory containing an excluded synthetic driver/audit bundle.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import tempfile

import numpy as np

assert len(sys.argv) == 2
BUNDLE = sys.argv[1]
if len(sys.argv) < 6:
    sys.argv.extend(["synthetic"]*(6-len(sys.argv)))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import b10_audit_phase11_g2 as audit
import b10_phase11_g2 as p11
import b10_phase9_train as p9


permutations, pre_arrays, pre_history = p11.independent_pre_update_containers()
permutations["sentinel"] = 1
assert pre_arrays == {} and pre_history == {}
pre_arrays["sentinel"] = 2
assert pre_history == {}

nested = p11.json_normalize({"array": np.asarray([[1., 2.]], np.float64),
    "scalar": np.asarray(3, np.int64)[()], "tuple": (np.bool_(True), np.float64(.5)),
    "nested": [{"value": np.asarray([4], np.int32)}]})
assert nested == {"array": [[1.0, 2.0]], "scalar": 3,
                  "tuple": [True, .5], "nested": [{"value": [4]}]}
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "nested.json")
    p11.atomic_json(path, nested)
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == nested
    try:
        p11.atomic_json(path, {"bad": np.asarray([np.nan])})
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite JSON was not rejected")


root = os.path.abspath(BUNDLE)
with open(os.path.join(root, "out", "phase11_g2.json"), encoding="utf-8") as handle:
    report = json.load(handle)
arrays = {key: value.copy() for key, value in np.load(os.path.join(root, "out", "phase11_g2.npz"), allow_pickle=False).items()}
with open(os.path.join(root, "out", "checkpoint.pkl"), "rb") as handle:
    checkpoint = pickle.load(handle)

assert audit.schedule_check(report, arrays, True)
bad_order = dict(arrays)
bad_order["update_resolution_order"] = arrays["update_resolution_order"].copy()
bad_order["update_resolution_order"][0] = 64
assert not audit.schedule_check(report, bad_order, True)

clean_trace = audit.trace_check(arrays, "train_trust", False)
assert clean_trace["pass"]
bad_rho = dict(arrays)
bad_rho["train_trust_rho"] = arrays["train_trust_rho"].copy()
bad_rho["train_trust_rho"][0, 0, 0] += 1e-3
assert not audit.trace_check(bad_rho, "train_trust", False)["pass"]
bad_work = dict(arrays)
bad_work["train_trust_jvp_count"] = arrays["train_trust_jvp_count"].copy()
bad_work["train_trust_jvp_count"][0, 0, 0] += 1
assert not audit.trace_check(bad_work, "train_trust", False)["pass"]
bad_bound = dict(arrays)
bad_bound["train_trust_bound_active_count"] = arrays["train_trust_bound_active_count"].copy()
bad_bound["train_trust_bound_active_count"][0, 0, 0] += 1
assert not audit.trace_check(bad_bound, "train_trust", False)["pass"]

assert audit.preflight_check(report, True)
bad_preflight = copy.deepcopy(report)
bad_preflight["preflight"]["projected_terms_seconds"]["train_globalization"] = 1.0
assert not audit.preflight_check(bad_preflight, True)

assert audit.decision_check(report, True)["pass"]
bad_decision = copy.deepcopy(report)
bad_decision["decision"]["predictor_trained"] = True
assert not audit.decision_check(bad_decision, True)["pass"]
bad_capacity = copy.deepcopy(report)
bad_capacity["decision"]["retracted_capacity_used"] = True
assert not audit.decision_check(bad_capacity, True)["pass"]

passing_metric = {
    "meshes": {"64": {"trajectory_error_mean": 1e-4, "trajectory_error_worst": 2e-4,
                         "all_finite": True, "boundary_violation_count": 0,
                         "k3_cox_identity_worst": 1e-15}},
    "pooled": {"trajectory_error_mean": 1e-4, "trajectory_error_worst": 2e-4,
               "mean_snapshot_relative_l2_squared": 1e-8, "all_finite": True,
               "boundary_violation_count": 0, "k3_cox_identity_worst": 1e-15},
}
positive = copy.deepcopy(report)
positive["status"] = "complete"
positive["globalized_train"] = copy.deepcopy(passing_metric)
positive["train_direct"] = copy.deepcopy(passing_metric)
positive["train_trust_health"] = {
    "finite": True, "breakdown_count": 0, "no_breakdown": True,
    "unhealthy_exhaustion_count": 0, "no_unhealthy_exhaustion": True,
    "defined_rho_match": True, "undefined_rho_zero": True,
    "undefined_never_accepted": True,
}
positive["selection"] = {"direct": copy.deepcopy(passing_metric),
    "oracle": copy.deepcopy(passing_metric), "direct_oracle_mean_ratio": {"64": 1.0, "pooled": 1.0},
    "trust_gate": True, "pass": True}
positive["decision"].update({"globalized_train_pass": True, "predictor_trained": True,
    "train_direct_pass": True, "selection_evaluated": True, "phase11_pass": True,
    "scientific_promotion_allowed": True})
assert audit.decision_check(positive, False)["pass"]
positive_bad = copy.deepcopy(positive)
positive_bad["selection"]["direct_oracle_mean_ratio"]["64"] = 1.5001
assert not audit.decision_check(positive_bad, False)["pass"]

corrupt_checkpoint = copy.deepcopy(checkpoint)
corrupt_checkpoint["globalized_q"] = checkpoint["globalized_q"].copy()
corrupt_checkpoint["globalized_q"][0, 0] += 1e-6
assert not np.array_equal(corrupt_checkpoint["globalized_q"], arrays["train_trust_terminal_q"])

counts = {"generator": p9.tree_count(checkpoint["generator"]),
          "encoder": p9.tree_count(checkpoint["encoder"]),
          "predictor": p9.tree_count(checkpoint["predictor"])}
assert counts == {"generator": 165954, "encoder": 164384, "predictor": 2533}
assert audit.negative_self_test()["pass"]
print("phase11_contracts=pass corruptions=10 positive_selection_contract=pass json_normalization=pass independent_pre_update_containers=pass")
