"""Generate the canonical closing append from native and coordinator audit JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main(record, coordinator):
    root = Path("/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude")
    result = json.loads((record/"archive/outputs/results.json").read_text())
    audit = json.loads((record/"analysis/audit.json").read_text())
    independent = json.loads(coordinator.read_text())
    archive = json.loads((record/"ARCHIVE.json").read_text())
    cleanup = json.loads((record/"REMOTE-CLEANUP.json").read_text())
    assert audit["passed"] and cleanup["absent_verified"]
    assert independent["source_json_sha256"] == audit["results_sha256"]
    assert independent["verified_timed_invocations"] == audit["timed_invocations"]
    assert independent["declared_checkpoint_case_and_repetition_counts_match"] and independent["frozen_spatial_parameters_match"]
    assert independent["fom_initial_passthrough_verified"]
    digest = hashlib.sha256(coordinator.read_bytes()).hexdigest()
    (record/"analysis/COORDINATOR-REVIEW.json").write_text(json.dumps(dict(source=str(coordinator), sha256=digest, result=independent), indent=2)+"\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    groups = [g for g in audit["groups"] if g["cohort"] == "union" and g["model"] != "fom"]
    section = ["", "", "## 2026-09-07", "", "### Heat frozen-head transfer04 closed: accurate transfer, efficient FOM still faster", "",
        f"Continued in the user-approved separate heat worktree/branch and existing namespace, with no new branch or merge. The bounded transfer study used scientific source `{result['source_manifest']['source_commit']}` and job `{result['metadata']['job_id']}` on {result['metadata']['node']} / {result['metadata']['gpu']}; driver elapsed time was {result['elapsed_seconds']:.6f}s. Native source, complete checked archive, audits and generated findings are committed through `{commit}`. The exact raw results SHA256 is `{audit['results_sha256']}`.", "",
        f"Both expanded-coverage heads and their training-code libraries stayed frozen. The declared restricted polynomial-boundary Gaussian family, diffusivity and output times were unchanged. Original development seed {result['settings']['cohorts'][0]['seed']} contributed {result['settings']['cohorts'][0]['count']} cases and fresh development seed {result['settings']['cohorts'][1]['seed']} contributed {result['settings']['cohorts'][1]['count']}; neither cohort entered training or initializer lookup. Final confirmation remains unopened. Requested meshes were {result['settings']['requested_intervals']}, shared observations {result['settings']['observation_intervals']}, fixed Crank–Nicolson step {result['settings']['dt']} and validated gradient tolerance {result['settings']['gradient_tolerance']}. Mesh-dependent full-input QR projections and weak operators were rebuilt, and all complete/modular query parity checks passed.", "",
        "Every FOM returns the exact supplied requested-grid initial field; host coarse restriction, direct-time sine-transform propagation, physically aligned interpolation and complete contiguous host output construction are charged. ROM instead returns its actual fitted initial field and charges full input transfer/projection, multistart fitting, evolution and all outputs. Coarse solver choices and the same-grid solver are deduplicated only when their actual solver/output configuration is identical. Paired GPU burn-in and synchronization precede each block; every cost and error uses the same actual invocation.", "",
        f"The native independent NumPy/SciPy audit checked {audit['timed_invocations']} invocations and {audit['full_fields_checked']} field files, maximum metric disagreement {audit['max_metric_disagreement']:.12g}. Every FOM evolved/interpolated field matches independent SciPy/NumPy calculations to relative discrepancy {audit['independent_fom_kernel_relative_disagreement']:.12g}; complete spectral-reference trajectories match to {audit['independent_spectral_reference_relative_disagreement']:.12g}. The observed nested-reference discrepancy is {audit['reference_empirical_delta']:.12g}; it is empirical evidence only and the strict bound is null. Selected nonstationary initial fits/steps: {audit['nonstationary_initial']}/{audit['nonstationary_steps']}. Root independently checked the preserved production fields, cohorts, frozen weights/libraries, repetitions and t0 policy; maximum metric difference {independent['maximum_metric_disagreement']:.12g}, source audit JSON SHA256 `{digest}`.", "",
        f"Across both heads and all union-cohort meshes, the worst full current-relative physical error was {max(g['full_error'] for g in groups)*100:.9f}% and worst shared-grid error {max(g['common_error'] for g in groups)*100:.9f}%. Both frozen heads satisfy the empirically adjusted development target throughout this mesh ladder. The efficient FOM remains faster. The following union-cohort selections use the full-grid 5% target, medians of per-case timing medians, and paired ratios as medians of per-case FOM/ROM ratios of those medians. All repetitions and reference allowances enter eligibility.", "",
        "| Output intervals | Selected ROM | Selected FOM | ROM ms | FOM ms | Paired FOM/ROM |", "|---:|---|---|---:|---:|---:|"]
    for s in audit["selections"]:
        if s["cohort"] == "union" and s["norm"] == "full" and s["target"] == .05:
            section.append(f"| {s['intervals']} | {s['rom']} | {s['fom']} | {s['rom_seconds']*1000:.6f} | {s['fom_seconds']*1000:.6f} | {s['paired_fom_over_rom']:.9f} |")
    section += ["", "Full and shared physical norms are reported separately, with current normalization, initial normalization, absolute error and decay/energy/advancement records at every output. Same-grid semidiscrete discrepancies remain separate from discrete-versus-spectral spatial errors. ROM-versus-discrete discrepancy contains representation, weak-solve and time error; this transfer study does not independently separate those contributions. No earlier scientific result is retracted. The planned GPU memory figure is a live-array estimate, not a measured allocator peak; detailed Slurm host-memory accounting is preserved.", "",
        f"All {audit['archive_files_checked']} member checksums passed, and the {archive['bytes']} archive bytes are tracked in {len(archive['parts'])} checked chunks with a tested restoration helper. Large extracted NPZ fields are omitted only as duplicate git blobs. Exact remote `{cleanup['remote']}` was deleted and absence verified. Generated native report, audit/summary JSON and PNG/PDF accuracy/cost/component figures live at `worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/`.", "",
        "This establishes frozen transfer for the declared restricted single-bump development family, not coverage of archived multi-bump heat or an efficient-FOM speed advantage. Broader scientific cohorts, tighter accuracy and further runtime work remain open for a later separately bounded continuation. No additional GPU study was launched this round, and all worktrees remain separate.", ""]
    text = "\n".join(section)
    (record/"analysis/CANONICAL-APPEND.md").write_text(text)
    canonical = root/"LAB-LOG.md"
    marker = "### Heat frozen-head transfer04 closed: accurate transfer, efficient FOM still faster"
    assert marker not in canonical.read_text(), "Already appended"
    with canonical.open("a") as handle: handle.write(text)
    print("Canonical transfer04 closing section appended")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("record", type=Path); p.add_argument("coordinator", type=Path)
    args = p.parse_args(); main(args.record, args.coordinator)
