**NOT READY.** CPU-only checks reproduced selection and classification failures. No repository files, git state, jobs, or GPU were changed. Temporary fixtures are under `/tmp/speed3d-review-*` and `/tmp/speed3d-config-review-*`.

Line references below are within `experiments/jcp-speed3d/`.

**[analyze.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze.py)**

| Item | Verdict | Finding and fix |
|---|---|---|
| Basic completeness/provenance | **CORRECT**, limited | Lines 57–63 bind scores to run path and job ID. Lines 70–71, 95–96 and 111–113 check completed validation jobs, exact row population, and timing population. |
| Complete selection population | **NEEDS-FIX** | Lines 273–291 accept whatever jobs were supplied; missing comparators are silently omitted and duplicate configuration keys overwrite earlier evidence. Check the supplied population against the planned g1–g3 inventory and reject ambiguous duplicates before selection. Otherwise a partial screen can select a false winner. |
| H1.2/H1.3 tri-state classification | **WRONG** | Lines 128–130 fold missing rows and invalid ADT evidence into `eligible=False`; lines 233–235 then return **ineligible before checking evidence**. An unresolved ADT comparison or incomplete population can therefore become a scientific failure. Track evidence completeness separately; require usable ADT evidence before interpreting E-c. Report unavailable when the failure is evidentiary. Reproduced with a missing-evidence fixture. |
| H1.4 rescue | **WRONG** | Lines 253–254 inherit that classification problem: an unresolved FS-K2 comparator can produce “rescue.” Require a demonstrated FS-K2 eligibility failure with sufficient evidence. |
| H1.4 cheapest-QN tie rule | **WRONG** | Line 252 uses plain minimum cost. A1.7 requires the 3% cost-tie rule, then median and worst accuracy. Reuse H2’s tie-aware picker. |
| H2 accuracy endpoints and cost ties | **CORRECT**, for aligned cases | Lines 49–54 implement endpoint propagation; lines 203–209 implement the 3% tie rule. |
| H2 comparator admissibility | **WRONG** | Lines 193–217 never require the comparator to be eligible or have valid evidence. A fixture with an ineligible comparator returned `status="selected"` and `S_screen=2.0`; missing comparator timing can instead cause division by `None`. Require comparator eligibility and comparison evidence before computing a speed-up. |
| Ranked cases | **WRONG** | Lines 104–108 exclude unrankable cases, but line 130 makes every configuration unselectable unless all 64 are rankable. A1.4 allows ranking on reference-eligible cases while solver eligibility still covers all 64. Preserve ranked **case IDs**, enforce a common mask and reference provenance, and distinguish missing scores from legitimately unrankable references. |
| Missing B* | **NEEDS-FIX** | Lines 214–217 permit an overall selection with `B_star=None`, without an explicit unavailable decomposition status. Retain any valid H2 result, but mark both decomposition availability and its reason explicitly; never infer a missing factor. |
| K9 gating | **NEEDS-FIX** | Lines 283–292 record K9 but do not prevent verdicts when it fails or is absent. Require the distance-classifier control before accepting claims depending on E-c. |

A winner equal to its comparator is legitimate: the screening ratio is exactly one and H2 fails the speed bar. It must not trigger relabelling or missing-panel errors downstream.

**[fom_match.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/fom_match.py)**

| Item | Verdict | Finding and fix |
|---|---|---|
| Accepted data format and FOM keys | **CORRECT** | Lines 47–51 use FD entries’ fully qualified `key` and PS dictionary keys. Every accepted smoke candidate key parsed successfully through `fomrun.parse`. |
| Basic admissibility | **CORRECT**, limited | Lines 33–42 reject uncalibrated FD entries, incomplete case populations, unconverged/nonfinite solves and unusable warm timings. This agrees with the current producer’s records. |
| Same ranked cases and per-case η | **WRONG** | Lines 43–44 include every FOM case regardless of its reference `eligible` flag. Lines 97–99 discard ROM ranking metadata; line 66 ignores the FOM η array. No case-ID alignment or reference agreement is established. Carry explicit IDs and reference pins, verify matching η, require solver convergence on all 64, and compare accuracy only on the common reference-eligible population. Merely checking cohort length at line 91 is insufficient. |
| K10 convergence control | **WRONG** | Line 60 supplies `admissible=False` directly. The control tests filtering of an already-classified entry, not whether an unconverged stage is rejected by admissibility logic. Inject raw candidate evidence through the same validator as real entries. Ensure its zero cost is not the sole rejection reason. |
| K10 inaccurate control | **WRONG** near the reference floor | Lines 61–71 assume multiplying error by ten must defeat interval overlap. For `e=1e-5`, `η=1e-4`, the correct tie rule accepts that synthetic entry and reports `K10 FAILED`. I reproduced this. Explicitly handle an unresolved negative control; a prospectively documented control whose errors exceed the uncertainty endpoints can test rejection without falsely accusing the matcher. |
| No match versus unavailable | **WRONG** | Lines 72–74 say “ROM more accurate than every admissible configuration” even when there are **zero admissible configurations**. Also, failure to satisfy both summary criteria need not establish accuracy dominance. Return distinct statuses for unavailable FOM evidence and no joint median/worst match. |
| Upper-bound and boundary labels | **CORRECT** | Lines 75–79 apply A1.9’s unconditional finite-ladder upper-bound label and flag a match at the cheapest admissible family member. Screening costs stay within f1. |

**[select_s3.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/select_s3.py)**

| Item | Verdict | Finding and fix |
|---|---|---|
| Comparator/winner label collisions | **WRONG — executable crash** | Lines 43–56 deduplicate keys before all comparator names are reserved. If C_W768 wins against C_old, its key receives label `W_C_old`; registering C_W768 then returns that alias without creating `labels["C_W768"]`. Line 70 raises `KeyError: 'C_W768'`. Reproduced. Register all comparators first, then deduplicate winners and B*. Use canonical labels consistently. |
| Ordinary p1 runtime interface | **CORRECT**, after fixing labels | Lines 74–100 match `mode_prof`: configurations, profile/mf/batch labels, 8×3 confirmations, gl12 and converged-rule settings are supplied. Fabricated distinct winner/B* configurations produced the expected rule lists. |
| H1.4 confirmation pairs | **CORRECT** | Lines 58–62 and 86 include screened pass/fail pairs; rescue is correctly excluded from a speed comparison against ineligible FS-K2. The upstream tie/classification fixes remain necessary. |
| H2 decomposition panels | **WRONG** | Lines 82–86 generate C–W* and C–B*, but omit **B*–W***. A2.7(b) requires each factor’s own direct paired panel; dividing those two separately measured ratios is not a substitute. Add the distinct B*–W* pairs. Reproduced with distinct W* and B*. |
| Winner equals comparator | **NEEDS-FIX** | Line 84 silently omits self-pairs; all-self selection produced an empty confirmation list. Record an explicit identity result with ratio one and H2 fail, or retain a self-panel under the literal A1.5 confirmation requirement. Downstream analysis must not treat the omitted measurement as unavailable. |
| A1.13 converged-rule runs | **CORRECT** | Lines 78–100 include every selected W* and B*, including deployed-rule finalists, and map them to the registered converged rules. |
| A2.7 reduced-rule ADT checks | **CORRECT** | Line 100 includes finalists using non-deployed rules; `mode_prof` consumes these for tight/tighter checks on cases 0–7. |
| Unqualified comparators exported for matching | **WRONG** | Lines 67–71 export comparators unconditionally, and lines 111–113 include C_old/C_W1024 in m1 regardless of their analysis status. Diagnostic runs are reasonable, but only eligible configurations may receive H4.3 matches. Carry and enforce eligibility/evidence status. |
| H4.3 target after changing path/mesh | **NEEDS-FIX** | Lines 67–71 export grid mesh-path errors; lines 115–125 reuse those matches for mesh-free ROMs at every mesh. H3.2 qualification permits some error change, and mesh-specific operators can change accuracy again. A selected FOM may consequently cease to tie-or-beat the actual timed ROM. Revalidate matching against each actual mesh-free ROM’s scores before accepting H4.3; if the cheapest valid match changes, obtain its paired timing. |
| m1 runtime interface | **CORRECT**, subject to above | Settings include selected rules and W256 for G5/K7; mesh and timing populations fit `mode_mesh`; emitted accepted-smoke FOM keys are parseable. |
| A1.11 freeze order | **CORRECT**, incomplete workflow | This script writes screening artifacts, keeps validation seed 923801, and does not freeze or open replication. However, the referenced `freeze_s3.py` is absent. The final confirmation/qualification analysis and immutable freeze remain prerequisites before replication. |

**[resources.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/resources.py)**

| Item | Verdict | Finding and fix |
|---|---|---|
| Certificate binding | **CORRECT** | Lines 140–153 require accepted smoke and bind the certificate to config bytes and allocation metadata. `cluster/stage.py` enforces allocation coverage. |
| p1/m1 certificates | **NEEDS-FIX — launch blocker** | Lines 143–148 reject prof/mesh modes. The generated p1/m1 cannot satisfy the staging contract through this implementation. Add smoke-derived inventories and projections, including converged rules, confirmations, batching, ADT checks and the L384 feasibility pilot. |
| Grid proxy described as an upper bound | **WRONG** | Lines 54–62 ignore sweep count and QN refresh variant when choosing missing-step proxies. The accepted smoke maps `fs|CN|0.0125|K2` to **K1** at 0.01. Larger step count does not establish a conservative cost bound for different solver work. Preserve solver variant or use an explicitly conservative measured envelope. |
| ADT/tighter and FOM cost projection | **NEEDS-FIX** | Line 83 prices tighter ADT as ordinary tight ADT. Lines 110–125 price all calibration levels using production solve cost, extrapolate missing meshes/families, and omit explicit FOM compilation costs in favour of a fixed setup allowance. These are estimates, not demonstrated upper bounds. Use available calibration timings and compile measurements, identify extrapolation uncertainty, and apply conservative allocation checks. |

These findings do **not establish a need to rerun g1–g3 or f1**. Most fixes can use their existing raw outputs. Fix the p1 panel inventory before submission: otherwise the missing B*–W* measurement requires another paired run. Changed mesh-specific FOM matches may likewise require additional paired timings. Resource underestimation alone does not invalidate a completed, complete job.

**NOT READY — blocking items:** evidence-aware verdicts and comparator gating; aligned ranked-case FOM matching and meaningful K10 controls; canonical selection labels and complete direct confirmation panels; validation of matches against the actual timed ROM; and p1/m1 resource certificates.