"""Generate final archival and independent-review records from retained evidence.

This is CPU-only metadata processing; it does not recompute an experiment.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREES = HERE.parents[2]
COST = TREES / "2026-11-18-jcp-paired-cost/experiments/jcp-paired-cost"
WEAK = TREES / "2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def evidence(root, names):
    return [{"path": str(root / name), "sha256": sha(root / name)} for name in names]


ledger = load(HERE / "RUN_LEDGER.json")
closure = {
    "schema": "P1-archive-closure-v1",
    "date": "2026-11-18",
    "experiment_complete": True,
    "further_P1_GPU_jobs": False,
    "data_archive_commit": "f80b9855c85283ad823a36ea9903446494e4464c",
    "data_archive_codeonly_mirror": "11b615595cbb25ea53caf676a2f6d127a5881af6",
    "codeonly_ref": "origin/codeonly/exp/2026-11-18-jcp-head-mechanism",
    "cluster_namespace": "/cluster/tufts/paralab/tawal01/jcp-head-mechanism-20261118-488b0837",
    "remote_cleanup": "All five verified job duplicates removed under exclusive OWNER guard; namespace contains OWNER.txt only.",
    "ledger": ledger,
    "checksum_manifests": evidence(HERE, [f"runs/{j['tag']}/{m}" for j in ledger["jobs"] for m in ("INPUTS.sha256", "OUTPUTS.sha256")]),
    "large_artifacts": load(HERE / "LARGE_ARTIFACT_MANIFEST.json"),
    "native_parity_manifest": str(HERE / "P03B_NATIVE_PARITY_MANIFEST.json"),
    "qualification_history": {
        "p01": "Reference and solver qualification failed; retained without scientific admission.",
        "p02": "Pilot numerically admitted after prospective reference/solver repair; documented CPU-affinity messages exclude timing inference.",
        "p03": "Dev16 provisional because one initial-fit raw-cost rounding tie selected a nonstationary solution; retained.",
        "p03b": "All original development cases qualified after uniform stationary cost-tie selector repair and physical replay check.",
        "p04": "Fixed pilot cases qualified at the registered richer test counts; finite sensitivity only.",
    },
    "parent_coordination": "Parent owns canonical LAB-LOG closing append and main report. Worktree remains unmerged; parent asks user about merging at campaign close.",
}
write(HERE / "ARCHIVE_CLOSURE.json", closure)

parity_path = COST / "runs/p02/output/implementation_parity.json"
parity = load(parity_path)
failed = [entry for entry in parity if not entry["passed"]]
p1 = load(HERE / "runs/p03b/output/summary.json")
case_evidence = []
for entry in failed:
    case = next(c for c in p1["cases"] if c["index"] == entry["case"])
    fit = case["head_ic_fit"]
    case_evidence.append({
        "case": case["index"],
        "p1_selected_start": fit["best_index"],
        "p1_all_initial_starts": [{k: start[k] for k in ("reason", "gradient", "cost", "iterations")} for start in fit["starts"]],
    })
write(HERE / "reviews/p3-p02b-initialization-review.json", {
    "schema": "independent-code-review-v1",
    "reviewer": "head_mechanism_run",
    "decision": "Cleared for the bounded unchanged-grid P02B replay, conditional on cluster fixture and unchanged all-target parity gates.",
    "reviewed_sources": evidence(COST, ["ns_query.py", "run_paired.py", "check_ns_fixture.py", "AMENDMENT_P02B.json", "configs/refined_dev16.json"]),
    "failed_evidence": {"path": str(parity_path), "sha256": sha(parity_path), "checks_completed_before_guard": len(parity), "failures": failed},
    "p1_reference_evidence": case_evidence,
    "findings": [
        "Failure was native head initialization stationarity, despite tiny physical output difference; output agreement does not waive the convergence gate.",
        "Replacing vmap by lax.map retains the same independent starts, adaptive LM objective, damping, acceptance guards, budget, stationary cost-tie selector and tolerances.",
        "All per-start costs, gradients, termination reasons, iteration counts and acceptance counts are returned by the complete query and persisted within the timing contract.",
        "Every target head/span case is qualified before POD regeneration and comparator compilation; all comparator candidates are still regenerated and timed in the same new allocation.",
        "Synthetic fixture checks selector boundaries, solver budget rejection, query shape/finiteness/reproducibility and classical wrapper parity. Actual case acceptance remains the strict production guard.",
        "Batched arithmetic is a plausible cause, not established until the replay. Sequential compiled arithmetic still differs from P1 host arithmetic.",
    ],
    "if_replay_fails": "Stop at the unchanged parity gate. Retain returned initial codes/residual/Jacobian and final proposals for diagnosis; no threshold relaxation, case exclusion, or frozen-input replacement.",
    "timing_claim_status": "No P02 timing result admitted; all eligible contenders must be retimed together.",
})

write(HERE / "reviews/p2-width-protocol-review.json", {
    "schema": "independent-protocol-review-v1",
    "reviewer": "head_mechanism_run",
    "decision": "Protocol scientifically acceptable. Concrete W0 projection implementation separately cleared; W1 implementation still requires review.",
    "reviewed_sources": evidence(WEAK, ["PROTOCOL-WIDTH.md", "configs/width_projection.json"]),
    "findings": [
        "Frozen nested prefixes and the original six development cases are retained; training, new cohorts and changes to the discrete operator are excluded.",
        "The all-time primary error includes initialization. Wider native projections intentionally combine representation and initialization improvements and are labeled accordingly.",
        "Projection-screen admission and equation-solve improvement have prospective worst-case criteria and all-case nonregression gates. Original test-count evidence and pointwise counterexamples remain intact.",
        "Common test count isolates width, richer tests within each width probe visibility, and the common disjoint observer lies beyond every minimized row set.",
        "Same-grid and same-step scope is explicit; no continuum, timing or uniform stability conclusion is implied.",
    ],
    "implementation_requirements": [
        "Record Gram eigenvalues/condition and full rank at all declared cutoffs; never silently truncate a prefix.",
        "Audit projection orthogonality and nested error curves using stable QR/SVD or a demonstrated numerical bound from the admitted Gram spectrum.",
        "Explicitly evaluate baseline reproduction, every reference/control gate, every candidate admission criterion and every original case.",
        "Respect the remaining lane budget and parent-controlled campaign slots; preserve any failure and stop without changing the gates.",
    ],
})

write(HERE / "reviews/p2-width-W0-implementation-review.json", {
    "schema": "independent-code-review-v1",
    "reviewer": "head_mechanism_run",
    "decision": "Cleared for the projection-only screen, conditional on the registered cluster fixture and post-pull baseline/environment checks. This does not release W1 equation solves.",
    "reviewed_sources": evidence(WEAK, ["screen_prefix.py", "cohort_guard.py", "cluster/stage.py", "PROTOCOL-WIDTH.md", "configs/width_projection.json", "PROVENANCE.json"]),
    "frozen_assets": evidence(WEAK, ["inputs/frozen.pkl", "inputs/rotation_R512.npz"]),
    "findings": [
        "The ordered mass QR and triangular coefficient solves use every original column, without numerical rank truncation or basis reordering.",
        "Each prefix retains Gram eigenvalues and independently computed triangular-factor singular values, with all declared rank cutoffs enforced. R128 baseline rank and projection controls enter every candidate admission.",
        "Per-prefix QR orthogonality, normal-residual orthogonality, coefficient-field versus direct QR parity, finite error curves and nested-projection checks are explicit.",
        "All original development cases are regenerated under the legacy source/seed/count guard, with original/stable operator and Jacobian-action checks and tight/tighter references retained.",
        "Both prospective worst-initial and worst-all-time projection criteria enter candidate admission. Original primary metrics and prior test-count evidence are preserved.",
        "Admission stays provisional until the accepted R128 error curves are reproduced after collection and logs/checksums are reviewed. Per-prefix controls allow a valid smaller candidate to proceed if a larger candidate fails, without excluding cases.",
        "The staged batch has a ten-minute ceiling, GPU/f64/highest preflight and the original operator fixture before the screen.",
    ],
    "not_yet_admitted": ["W0 numerical results", "W1 equation-solve implementation", "Any broader accuracy, continuum, stability or speed claim"],
})
print(json.dumps({"written": ["ARCHIVE_CLOSURE.json", "reviews/p3-p02b-initialization-review.json", "reviews/p2-width-protocol-review.json", "reviews/p2-width-W0-implementation-review.json"]}))
