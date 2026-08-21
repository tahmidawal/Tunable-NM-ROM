#!/usr/bin/env python
"""Negative self-test for Phase-8 attempt-value audit semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


AUDITOR = Path(__file__).resolve().parents[1] / "b10_audit_phase8_d.py"
SPEC = importlib.util.spec_from_file_location("b10_audit_phase8_d", AUDITOR)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_values():
    return {
        "trial": np.array([0.5, 1.2]),
        "predicted": np.array([1.0, -0.1]),
        "actual": np.array([0.5, -0.2]),
        "rho": np.array([0.5, -np.inf]),
        "finite": np.array([True, False]),
        "attempted": np.array([True, True]),
    }


def require_rejected(values, label):
    try:
        MODULE.audit_attempt_values(**values)
    except SystemExit:
        return
    raise AssertionError(f"corruption accepted: {label}")


MODULE.audit_attempt_values(**valid_values())

corrupt = valid_values(); corrupt["rho"][1] = np.inf
require_rejected(corrupt, "positive-infinity sentinel")
corrupt = valid_values(); corrupt["finite"][1] = True
require_rejected(corrupt, "sentinel marked finite")
corrupt = valid_values(); corrupt["trial"][1] = np.nan
require_rejected(corrupt, "nonfinite trial work")
corrupt = valid_values(); corrupt["rho"][0] = np.nan
require_rejected(corrupt, "nonfinite positive-prediction rho")

print("phase8_negative_trace_self_test=PASS valid_sentinel=accepted corruptions=4_rejected")
