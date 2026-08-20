#!/usr/bin/env python
"""Regression check for immutable S0/P3 audit-decision schema normalization."""
from __future__ import annotations

import argparse

import b10_phase4_d as d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0-json", required=True)
    parser.add_argument("--s0-npz", required=True)
    parser.add_argument("--s0-audit", required=True)
    parser.add_argument("--p3-json", required=True)
    parser.add_argument("--p3-npz", required=True)
    parser.add_argument("--p3-audit", required=True)
    args = parser.parse_args()
    s0_report, s0_audit = d.validate_bound_artifact(
        args.s0_json, args.s0_npz, args.s0_audit, "S0"
    )
    p3_report, p3_audit = d.validate_bound_artifact(
        args.p3_json, args.p3_npz, args.p3_audit, "P3"
    )
    assert d.normalized_audit_decision(s0_audit, "S0") == s0_report["decision"]
    assert d.normalized_audit_decision(p3_audit, "P3") == p3_report["decision"]
    print("phase4_chain_schema_regression=PASS s0=nested p3=legacy-flat")


if __name__ == "__main__":
    main()
