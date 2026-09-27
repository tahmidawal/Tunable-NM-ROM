"""Licensed Phase-4 H1/k24 seed-11 hyperdecoder/autolatent trainer.

Usage: b10_phase4_train.py P4_D.json H1 OUTPUT.json OUTPUT.npz CHECKPOINT.pkl

This is a thin architecture and provenance adapter around the locked Phase-2
optimizer/data protocol.  It cannot open model-validation or confirmation data.
"""
from __future__ import annotations

import json
import os
import sys

import b10_phase4 as p4
import b10_spline as spline
import b10_spline_train as train


class HierarchicalArchitecture:
    CANDIDATES = p4.CANDIDATES
    HYPER_WIDTH = spline.HYPER_WIDTH
    affine_state_from_field = staticmethod(spline.affine_state_from_field)
    normalized_state_from_affine = staticmethod(spline.normalized_state_from_affine)
    apply_mlp = staticmethod(spline.apply_mlp)
    init_mlp = staticmethod(spline.init_mlp)

    @staticmethod
    def coefficient_count(candidate):
        return p4.coefficient_count(candidate)

    @staticmethod
    def decode_candidate_jax(state, coefficients, coords, boundary_mask, candidate):
        return p4.decode_one_cox_jax(
            state, coefficients, coords, boundary_mask, candidate
        )

    @staticmethod
    def hyperdecoder_parameter_count(_r, k):
        candidate = next(
            item for item in p4.CANDIDATES
            if item["arm"] == train.ARM and item["k"] == int(k)
        )
        return int(candidate["hyperdecoder_parameters"])

    predictor_parameter_count = staticmethod(spline.predictor_parameter_count)


def _resolve(path, basename):
    directory = os.path.dirname(os.path.abspath(path))
    candidates = (os.path.join(directory, basename), os.path.join(directory, "..", basename))
    matches = [os.path.abspath(item) for item in candidates if os.path.isfile(item)]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {basename} beside P4 artifact")
    return matches[0]


def load_phase4_gate(path, arm):
    with open(path) as handle:
        artifact = json.load(handle)
    json_sha = train.c.sha256(path)
    npz_record = artifact.get("npz") or {}
    npz_path = _resolve(path, npz_record.get("basename", ""))
    audit_path = _resolve(path, "AUDIT.json")
    manifest_path = _resolve(path, "MANIFEST.sha256")
    with open(audit_path) as handle:
        audit = json.load(handle)
    decision = {
        "selected_spatial_arm": "H1",
        "training_seed11_k24_licensed": True,
        "correction_classification": "conditional-zero-or-occasional-attempt",
        "phase4_hard_stop": False,
        "scientific_promotion_allowed": True,
    }
    provenance = artifact.get("provenance") or {}
    valid = (
        artifact.get("status") == "complete"
        and artifact.get("decision") == decision
        and arm == "H1"
        and train.TRAIN_SEED == 11
        and npz_record.get("sha256") == train.c.sha256(npz_path)
        and audit.get("status") == "pass"
        and audit.get("source_json_sha256") == json_sha
        and audit.get("source_npz_sha256") == npz_record.get("sha256")
        and audit.get("manifest_sha256") == train.c.sha256(manifest_path)
        and audit.get("expected_commit") == provenance.get("commit")
        and str(audit.get("expected_job")) == str(provenance.get("slurm_job_id"))
        and audit.get("decision") == decision
        and provenance.get("jax_backend") == "gpu"
        and provenance.get("x64") is True
        and provenance.get("matmul_precision") == "highest"
        and artifact.get("config", {}).get("model_validation_touched") is False
        and artifact.get("config", {}).get("confirmation_touched") is False
    )
    if not valid:
        raise SystemExit("P4-D artifact/audit does not license H1 seed-11 k24 training")
    if train.PRIOR_TRAIN_JSONS:
        raise SystemExit("the first H1 seed-11 cell accepts no prior training artifacts")
    gate = {
        "path": os.path.abspath(path), "sha256": json_sha,
        "npz_path": npz_path, "npz_sha256": npz_record["sha256"],
        "audit_path": audit_path, "audit_sha256": train.c.sha256(audit_path),
        "manifest_path": manifest_path,
        "manifest_sha256": train.c.sha256(manifest_path),
        "source_commit": provenance["commit"],
        "source_job_id": str(provenance["slurm_job_id"]),
        "decision": decision,
        "artifact_chain": {
            "decision": "audited P4-D selects H1 and licenses seed-11 k24",
            "required_seed": 11, "required_arm": "H1", "prior_artifacts": [],
        },
        "smoke": bool(train.SMOKE),
    }
    return artifact, gate


def phase4_extra_gates(report):
    oracle = report["selection_oracle"]["metrics"]
    rows = list(oracle["meshes"].values()) + [oracle["pooled"]]
    oracle_pass = bool(report["selection_oracle"]["gate_pass"])
    near_miss = bool(
        not oracle_pass and all(
            row["trajectory_error_mean"] <= 4e-4
            and row["trajectory_error_worst"] <= 1.4e-3
            and row["all_finite"] and row["exact_binary_boundary"]
            for row in rows
        )
    )
    return {
        "k32_retraining_near_miss_condition": near_miss,
        "k32_retraining_licensed": near_miss,
        "phase4_next_decision": (
            "promote H1/k24 seed11" if report["gates"]["promote_seed"]
            else "license same-H1 k32 complete retraining" if near_miss
            else "hard stop: direct predictor misses locked gate" if oracle_pass
            else "hard stop: learned manifold misses the 2x bracket"
        ),
    }


def main():
    if train.ARM != "H1" or train.TRAIN_SEED != 11:
        raise SystemExit("Phase-4 first training cell is locked to H1 seed 11")
    train.s = HierarchicalArchitecture
    train.load_s0_gate = load_phase4_gate
    train.PARENT_GATE_NAME = "phase4_gate"
    train.STAGE_NAME = "Phase-4 H1/k24 seed-11 complete training"
    train.EXTRA_GATE_FN = phase4_extra_gates
    train.main()


if __name__ == "__main__":
    main()
