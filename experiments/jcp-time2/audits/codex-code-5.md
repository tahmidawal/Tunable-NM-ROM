**Overall verdict: NOT CLEAN for submitting `a1kfast` from the current checkout, even after the smoke audit passes.** Staging still rejects untracked lane files, and several report safeguards remain incomplete. I found no evident L=1024 shape or memory blocker.

Reviewed HEAD `84b5afbe0`, the five commits, and the requested diff. Validation was read-only: source inspection, synthetic counterexamples, grid/resource arithmetic, certificate hashes, and syntax checks. No files modified, jobs submitted, or GPU computations run.

Paths below are relative to `experiments/jcp-time2/`.

1. **Item 12 — STILL-WRONG: audit gating is improved but unauthenticated.**  
   `make_report.py:176–261` now suppresses H1/H2 when prerequisites or comparator verification fail. However, it trusts `aud['all_pass']` without comparing the audit’s attempt/job/commit with the result. A synthetic audit with deliberately wrong job and commit still authorized selection.

   Separately, a failed audit still permits unqualified temporal-order claims: a counterexample returned **“order 2”** while H1 was correctly unavailable. An unverified comparator produces “no verified candidate,” obscuring the missing prerequisite.

   **Fix:** authenticate audit identity against the result, preferably also bind the result artifact hash; gate order claims as well as selection/H1/H2. Report an unverified comparator as **unavailable**.

2. **Item 13 — STILL-WRONG: smoke claims are restricted, but presentation remains misleading.**  
   `claims_eligible` now prevents ordinary `smk` selection and labels its orders diagnostic. However, `make_report.py:369–375` still introduces every report as **38 cases, dev6 ∪ val32, L=1024**. Plot labels, timing qualifications and glossary retain hard-coded cohort/mesh/repetition descriptions.

   Eligibility itself checks only `len(cases)==38` and an `a1k` name prefix, rather than the prescribed case identities/configuration.

   **Fix:** derive descriptions from each authenticated configuration; require exactly dev6 cases 0–5 and val32 cases 0–31 for scientific claims. Label the entire smoke report as machinery diagnostics.

3. **Item 14 — RESOLVED for the identified presentation defects.**  
   Anchor distances are now consistently called **anchor discrepancies**, percentages are converted correctly, eligible/resolved counts appear, and plots distinguish points unresolved in most cases. The qualification limits time-error interpretation to demonstrated GAL-BDF2 order and resolved distances.

   **Fix:** none for this item’s core defects. The audit-gating defect in item 12 still affects the order claim used by that qualification.

4. **Item 15 — STILL-WRONG: correct distance, missing paired verification.**  
   `t2run.py:256–259` computes the requested main-versus-HQ field distance. `audit_t2.py:130–131,232–233` reconstructs it and requires it for production attempts. These changes are correct.

   But `make_report.py:203–204` aggregates `vs_main` without checking verification of **both** trajectories. A synthetic unverified HQ trajectory was displayed as quadrature sensitivity `0.0123`.

   **Fix:** aggregate only verified main/HQ pairs; display valid-pair counts and mark incomplete comparisons unavailable or diagnostic. Keep HQ ST error separate, as now implemented.

5. **Item 16 — RESOLVED.**  
   `make_report.py:154–173` implements the literal three-MAD outlier rule without the artificial floor. The generated timing table now displays sample counts, ratio median/IQR/outliers, and drift median/IQR/outliers.

   **Fix:** none.

6. **Item 17 — STILL-WRONG, with most requested diagnostics now present.**  
   Failure tables, production-versus-tight distances, full/shared-node ratios, coarse triples, and unverified plot markers have been added.

   Remaining problems:
   - `full_over_257_median` at lines 200–202 includes unverified candidates and ineligible anchors, without a validity count.
   - Production-versus-tight aggregation does not expose whether its tight counterpart verified.
   - The historical motivation paragraph still lacks an adjacent provisional qualification; the old-method table has been fixed.

   **Fix:** provide verified-pair/eligible-anchor counts and explicit diagnostic labels for these aggregates; qualify historical reference-based numbers inline.

7. **A7 construction — RESOLVED by static inspection; numerical conditioning remains untested here.**  
   The new setting is injected before `qstudy.Mesh` uses `Q.SETTINGS`. The complete construction is dimensionally consistent:

   | Component | Wide construction |
   |---|---|
   | Nested bank | Edges `[0,128,384,512]`; block widths 128, 256, 128 |
   | `BK.prefix(...,512)` | Finds edge index 3 and retains all blocks |
   | Trust radius | `Hall @ Lrot[:512].T`, giving 512-dimensional codes |
   | Cold fit | Gauss 48²; weighted bank 2304×512; reduced QR gives 512×512 triangular solve |
   | Operators | 2048 lowest discrete sine modes; `A` is 2048×512 |
   | GAL projection | Reduced QR of `A`; compatible 512-dimensional system |

   There is no missing edge, fixed-384 truncation, or underdetermined cold-fit shape. This establishes construction consistency, **not** full rank or acceptable conditioning of the actual cold matrix.

   **Fix:** no structural fix identified. Inspect recorded `A` conditioning and solver verification; a wide-specific execution check remains valuable because `smk` contains only fast/acc.

8. **A7 disclosure — honest in DESIGN, incomplete in generated reporting.**  
   DESIGN explicitly calls Gauss 128² **unselected**, attributes the motivating result to a two-case C1 smoke, distinguishes its test-space choice, and makes wide secondary. That is appropriately qualified.

   The report generator does not carry the secondary/unselected warning into the wide section or hypotheses. Its sensitivity column covers only production arms with HQ counterparts; order and anchor diagnostics have no corresponding HQ check.

   **Fix:** label wide as secondary and its rule as unselected in generated output. State where sensitivity is unavailable, and describe Gauss 192² disagreement as a sensitivity indicator—not a quadrature certificate. Narrow A7’s “beside every number” promise to the comparisons actually measured.

9. **G2a certificate — RESOLVED under the stated certificate contract.**  
   The committed certificate passes, and both source hashes match current `t2core.py` and `test_lmm.py`. Reading the certificate from the job’s commit correctly avoids rejecting an older smoke merely because today’s core changed.

   **Fix:** none to that audit change. For consistency, report machinery evidence should also identify the applicable job certificate instead of silently using today’s certificate.

10. **L=1024 memory gate — no evident capacity blocker on A100-80G / 128G host.**  
    Calculated array payloads, excluding compilation/workspace:

    | Allocation | GiB |
    |---|---:|
    | Fast mesh bank | 0.998 |
    | Accurate mesh bank | 2.994 |
    | Wide mesh bank | 3.992 |
    | 232 retained shared-node trajectories, per case | 0.685 |
    | One full six-field trajectory, including boundary | 0.047 |
    | Wide main point-rule blocks | 0.375 |
    | Wide HQ point-rule blocks | 0.844 |
    | Wide old-method blocks | 0.136 |

    Fields are released per case; they do not accumulate across all 38 cases. The anchor and generic-BE full fields add about 0.094 GiB. Operator projection uses **32-column chunks**, so its principal wide intermediate is about 0.500 GiB, rather than a full mesh-by-test matrix.

    HQ construction and bank rotation require additional temporary copies, but the inspected dimensions leave substantial capacity margin. No array-size indexing limit is apparent.

    **Fix:** none demonstrated. These are static estimates, not measured peak-memory guarantees.

11. **Wall-time gate — 4–8 hours is an estimate, not yet supported by execution evidence.**  
    Each setting runs:

    - **8,816 accuracy rollouts**, totaling **1,035,310 time steps**.
    - **738 A–B–A records**, comprising 2,214 timed queries.
    - **252 warm-up queries**, with about 247 seconds of prescribed burn time across warm-up and measured calls.

    Ignoring every other cost, a four-hour allocation permits only **1.63 seconds per accuracy rollout**; eight hours permits **3.27 seconds**. Tight solves can consume up to 600 LM iterations per step. Wide’s main point-Jacobian multiplication has approximately **3.16×** accurate’s arithmetic and **164×** fast’s; runtime need not scale by those factors.

    **Fix:** use smoke per-setting case times and iteration distributions to justify the requested allocation, with compilation/setup and timing overhead included. Do not extrapolate fast timings directly to wide. Coefficients are saved only after all cases finish, so timeout during phase 1 loses the independently auditable coefficient artifact.

12. **Submission mechanics and scientific coverage — staging blocker remains; intended ROM evidence is collected.**  
    Configurations consistently specify L=1024, the complete cohort, correct main/HQ/old rules, and six timing cases × three repetitions. The staged script uses the intended GPU partition, precision, GPU preflight, host memory and paralab paths. Submission retains the lane lock and cap-one check.

    However, the checkout contained untracked `audits/codex-code-5.log`; a later status also showed untracked `fom2.py`. `cluster/stage.py:43` rejects both. Passing the smoke audit does not remove this blocker.

    The outputs support the planned **ROM** order, failure, anchor, ST/TX selection, timing and quadrature-sensitivity analyses once reporting is repaired. They do **not** establish converged continuum accuracy, wide-rule convergence, historical-test replication, or the separate fair second-order FOM comparison.

    **Fix:** deliberately resolve the untracked files before staging, repair the reporting safeguards, and justify wall time from smoke evidence. No driver shape/memory defect was found that independently requires withholding `a1kfast`; the current checkout and claim pipeline nevertheless remain **NOT CLEAN**.