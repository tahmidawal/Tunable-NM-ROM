**C1 RESULTS NOT ACCEPTED — reporting fixes required.** The audited numerical results passed. I found no reason to rerun the production ladder.

Read-only audit completed: no repository, main, LAB-LOG, or cluster changes. Scratch usage: 6.7 MB.

| Item | Verdict | Evidence |
|---|---|---|
| 1. Job integrity | **CORRECT** | All four specified jobs record H200, `jax_backend=gpu`, x64 and highest precision, complete and accepted. All **282** archived F/C files pass their SHA256 manifests. All **1,110** screening repetitions and **370** warm-ups record bitwise identity; repetition coverage is complete. |
| 2. Calibration and convergence | **CORRECT** | Independently recomputed **all 97** calibration choices from C fields. No t0 failures, no recoveries; **37 t1 / 60 t2** choices. Full evidence scan agrees on **277,628 FD production stages**, including residual thresholds, Newton counts and stage/unit coverage. |
| 3. E0 scoring and eligibility | **CORRECT** | Rescored **11 configurations × 38 cases = 418 fields** through refstore, spanning both FD families, selected PS configurations and pathological PS outputs. Scores agree exactly, including non-finite classifications. Sampled field hashes match production records. All 38 ST cases are eligible; eligibility and uncertainty records agree throughout the ladder. |
| 4. Admissibility | **CORRECT** | Recomputed all 185 decisions: **170 admissible**, 15 rejected PS configurations. Finite-but-inaccurate outputs remain admissible where the registered rules permit them; joint matching rejects them. See the median-only qualification below. |
| 5. Pareto fronts | **CORRECT** | Independently reconstructed fronts from screening repetitions and archived scores: **UB 26, CC 26, PS 22**, with identical membership. The enormous-error PS N16 point can legitimately remain nondominated because of its cost and median error. |
| 6. Matching and controls | **CORRECT** | Recomputed joint, median-only and worst-only candidate ordering and selections for **all 48 H/X targets**. Same-mesh selections, common-population intersections, uncertainty agreement, edge/boundary labels and K-syn/K-false checks agree. All joint selections have neither edge nor boundary flags. The PS anchor recomputes within floating-point roundoff. |
| 7. Extension decision | **NEEDS-RESTATEMENT** | “None triggered” is correct **for these P1 preview targets**: no joint-selected comparator reaches the base minimum resolution—UB 64, CC 32, PS 8. It does not finalize the A3.2 decision for the forthcoming FRONT targets. |
| 8. Report numbers and wording | **NEEDS-RESTATEMENT** | Scratch-only regeneration reproduces `report.md` byte-for-byte, and checked C1 tables agree with the JSON. Conditional E0 and upper-bound labels are present. Required PS disclosures are missing, and several statements overstate the current scope. |

Calibration detail requested: for **UB L2048, Δt=0.000625**, maximum distances from t0 are:

| Level | Distance |
|---|---:|
| t1 | 5.1422060076×10⁻⁷ |
| t2 | 1.0539014488×10⁻⁵ |
| t3 | 1.0448587745×10⁻⁵ |

Thus **t1** is correctly selected under the 10⁻⁶ threshold.

The rejected PS configurations are **N24 at 0.05**, plus **N32, 48, 64, 96, 128, 192, 256 at both 0.05 and 0.025**. N24 has finite solver evidence but a non-finite score caused by norm-squaring overflow; the other 14 also fail solver validity. The archived PS warning is that scoring overflow, not an OOM or GPU fallback.

**Important qualification to “must never match”:** `ps|N16|0.05` has worst error **6.1414743887×10²²**, on `val32|19`. It matches **zero targets jointly**, as required. However, it passes the median criterion for 30 target variants and is the **selected median-only comparator for six**: R128 CNR at all three meshes, H and X populations. That follows DESIGN’s separate median-only rule. An unconditional statement that it “never matches” would be false; banning it from median-only comparisons would require changing that rule.

Ranked fixes:

1. **Disclose the selected PS diagnostic and anchor.** All **24 H selections / 48 H+X selections** choose `ps|N48|0.0125`. Its registered padded-grid stability label **fails on all 38 cases**. This is diagnostic and does not invalidate admissibility, but DESIGN explicitly requires disclosure. Its valid adjacent-step comparator is `ps|N48|0.01`; the normalized maximum field distance is **0.0009142809616439**. Report both steps, both-valid status and that the anchor certifies nothing.

2. **Scope finality and extension claims to the preview.** Replace “C1 final” with wording distinguishing the completed base ladder from provisional P1 matching. Label the extension decision “not triggered for P1 preview targets; recompute for FRONT or the authorized fallback.”

3. **Replace “most accurate” with “lowest median error.”** The distinction changes the selected configuration:
   - CC’s lowest-median member is L4096, but L512 has smaller worst error: **0.0594135% versus 0.0624053%**.
   - PS’s lowest-median member is N128, but N256 has smaller worst error: **0.000944215% versus 0.00250953%**.
   
   These comparisons are descriptive conditional E0 values, not uncertainty-resolved superiority claims.

4. **Tighten comparison wording.** Say “cheapest by lattice screening cost among the calibrated admissible tested ladder.” Remove the assertion that this lane already supersedes earlier speed-up comparisons: C2 production has not run. Explicitly distinguish median-only matches from joint accuracy matches when exposing the blown-up N16 comparator.

These fixes belong in the report generator and regenerated report; they do not require new GPU jobs.

**NOTED — authentication limit:** archived checksums establish internal consistency. Warm-output identity is supported by persisted checks and inspected code, not an independent replay of historical GPU executions. This is not a blocking forgery-only finding.

Audit evidence: [numerical checks](/tmp/jcp-fom2d-audit-scratch/audit.out), [anchors and archive checks](/tmp/jcp-fom2d-audit-scratch/extra.out).