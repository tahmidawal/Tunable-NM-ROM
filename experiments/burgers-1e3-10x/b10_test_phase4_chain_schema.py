#!/usr/bin/env python
"""Regression check for immutable S0/P3 audit-decision schema normalization."""
from __future__ import annotations

import argparse
import copy

import b10_audit_phase4_d as audit
import b10_phase4_d as d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0-json", required=True)
    parser.add_argument("--s0-npz", required=True)
    parser.add_argument("--s0-audit", required=True)
    parser.add_argument("--p3-json", required=True)
    parser.add_argument("--p3-npz", required=True)
    parser.add_argument("--p3-audit", required=True)
    parser.add_argument("--p3-manifest", required=True)
    parser.add_argument("--stage-manifest", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--worktree", required=True)
    args = parser.parse_args()
    s0_report, s0_audit = d.validate_bound_artifact(
        args.s0_json, args.s0_npz, args.s0_audit, "S0"
    )
    p3_report, p3_audit = d.validate_bound_artifact(
        args.p3_json, args.p3_npz, args.p3_audit, "P3"
    )
    assert d.normalized_audit_decision(s0_audit, "S0") == s0_report["decision"]
    assert d.normalized_audit_decision(p3_audit, "P3") == p3_report["decision"]
    prior = d.manifest_rows(args.p3_manifest)
    stage = d.manifest_rows(args.stage_manifest)
    report = {
        "bindings": {
            "P3": {"staged_manifest_sha256": d.c.sha256(args.p3_manifest)},
            "runtime_dependencies": {
                "burgers2d_film_sha256": prior[
                    "code/deps/burgers2d-coord-rom/burgers2d_film.py"
                ],
                "bh_common_sha256": prior["code/bh_common.py"],
                "source_manifest": "P3",
            },
        },
    }
    assert d.c.sha256(args.p3_manifest) == stage[
        "code/deps/p3/MANIFEST.sha256"
    ]
    assert audit.verify_runtime_manifest(
        report, stage, prior, args.expected_commit, args.worktree
    )
    corrupted = copy.deepcopy(stage)
    corrupted["code/deps/p3/MANIFEST.sha256"] = "0" * 64
    try:
        audit.verify_runtime_manifest(
            report, corrupted, prior, args.expected_commit, args.worktree
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("missing nested-manifest corruption was accepted")
    print(
        "phase4_chain_schema_regression=PASS s0=nested p3=legacy-flat "
        "nested_manifest=bound corruption=rejected"
    )


if __name__ == "__main__":
    main()
