**Overall verdict: NEEDS-FIX.** Do not submit the real-size smoke yet.

Audited committed baseline `38a0f73ba9d2`, with the concurrent edits noted below. No files changed by me, no jobs submitted, no scratch files created, and no lab-log writes.

The CPU unit gates passed. All eight unchanged copies matched their sources and recorded SHA-256 values; every other `PROVENANCE.json` source/edited hash checked also matched. I compared all four requested edited copies against their originals.

Paths below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

1. **Sealed/OOD access and E0-store protection — NEEDS-FIX.**

   The normal cohort routes are correctly restricted: `refjob.py:318–330`, `540–544`; `p1common.py:81–89`; `rscore.py:79–87`. The inherited RNG gates generate public-seed fixtures, not secret cohorts.

   **The output boundary is not protected.** `refjob.py:590–592` accepts arbitrary `--out`; `366–369` removes unverifiable existing reference files there. Pointing it at E0’s store can therefore overwrite E0 references. `refslim.py:28–38` similarly permits arbitrary destinations. The staged workflow supplies safe paths, but the drivers do not enforce the promised prohibition.

2. **IC identity through reference, lattice, spectral initialization and ROM — CORRECT numerically; NEEDS-FIX identity verification.**

   The routes agree: `rcohorts.py:156–170`, `295–314`; `refcore.py:239–246`; `refjob.py:80–83`; `p1common.py:116–122`; `qacc.py:115–123`. The unchanged query preserves nodal-input sampling.

   CPU checks found exact single-bump agreement with `engines.initial`; two-bump spectral coefficients agreed with direct tensor quadrature to **3.20×10⁻¹⁵**.

   However, A1.6’s **per-scored-record identity assertions are absent**. `rscore.py:64–74` checks reference cohort metadata but not the descriptor, stored `t0` hash, or ROM/reference identity. `error():108–115` checks only shape. `qacc.py:118–123` checks one case per cohort, without binding it to reference metadata.

3. **E6 allowance and escalation — CORRECT core branches; NEEDS-FIX enforcement.**

   `refjob.py:159–193` implements the specified blocked-floor outcome, no refinement after tight, no refined tight triple, and skip flags. Small mocked executions confirmed all four principal branches. The previous design-audit concern about `unresolved-floor` is resolved by `verdict.judge` precedence.

   Remaining holes:

   - Allowance validation at `331–336` applies only to regime cohorts; dev6 can receive an arbitrary allowance.
   - `cases == sorted(cases)` permits duplicates; `kinds` is not restricted to `['ST']`. An extra S solve can consume the cohort allowance.
   - Resume deductions occur only as cases are encountered (`350–365`). There is no preliminary accounting of all verified resumed tight cases or binding to the original population/configuration.
   - Verified resumed cases skip probes entirely (`365`); previous probe results are not restored by `main():599–602`.

4. **E5 calibration and upwind probes — NEEDS-FIX.**

   Calibration chooses the last coarse production/retry attempt and requires both solves valid plus finite deviation (`refjob.py:203–218`). Upwind uses the intended base triple and CN-R schedule (`219–227`). The smoke predicate correctly demands **exactly `order-fail`**, with solver, shared-t0 and floor passing (`smoke_accept.py:52–57`).

   **But the probe arrays and residual histories are discarded.** Only summaries survive. Consequently, acceptance cannot independently recompute either probe’s verdict. Upwind rejection for the intended reason remains a real-size smoke question; it has not been demonstrated by these CPU checks.

5. **Stiffness statistic and memory — NEEDS-FIX.**

   The stencil is correct: inclusive interior box, zero padding, central differences and Euclidean gradient norm (`refjob.py:105–120`). A small linear-field fixture matched exactly. Computing k=0 additionally is harmless because `sigma_of()` uses k=1…5. The denominator is descriptor amplitude, consistent with A4.

   **Unavailable-value handling is wrong.** `rscore.py:95–105` ignores `steps_ok` and non-finite gradients. A fixture with a failed finest solve returned a usable `sigma=2.0`. JSON-cleaned non-finite gradients can also become `None` and crash `max()`.

   At L=12288, one f64 plane is approximately **1.21 GB**; six outputs occupy **7.25 GB**. E3 adds several plane-sized temporaries, while `refcore.py:230–233` also pads all six outputs. This does not itself establish an H200 OOM, but total solver/FFT/cached-executable peak memory remains unverified. It must be measured in the corrected smoke.

6. **Frozen ROM adapters, populations and f257 split — NEEDS-FIX.**

   The numerical adapters preserve qualify’s tolerances, query implementation and historical ICs. Rules/ranks match the design (`make_configs.py:13–25`); production Q2/all and tighter/sub selection match A1.6 (`qacc.py:157–159`). Explicit control case lists resolve correctly, including both C2 and B2 pilots. I found no explicit-list crash.

   Two deficiencies remain:

   - Checkpoint/rotation hashes are recorded, not checked against frozen expected identities (`qacc.py:101–104`). The mode JSON is copied but the mode list is regenerated without checking that pinned artifact (`p1common.py:154–158`).
   - The f257 split retains coefficients needed for errors and distances, but **weakens the full-field independent decode check** to samples and norms (`qacc.py:244–252`, `rscore.py:200–212`). Unsampled field corruption preserving a norm can pass; the saved full-field hash is unused.

   The committed smoke used `tighter_cases='sub'`, leaving C2 c9/c11 without tighter/check runs. A concurrent uncommitted change now uses `'all'` for smoke and regenerates matching JSON. That correction needs inclusion in the audited commit.

7. **Qualification, labels, trends, separation and paired statistics — WRONG.**

   Per-regime certificate pooling itself is correct (`rscore.py:175–194`). **Qualification is not semantically equivalent to `qscore.qualify`:**

   - Expected production, tighter and Q2 populations are never checked.
   - Shape and recorded form/step checks were dropped.
   - Missing tighter cases make the tighter predicate pass (`273–274`).

   A CPU counterexample returned **`numerically_qualified=True`, no disagreements**, despite missing expected production/Q2/tighter cases and incorrect recorded form/steps. Compare original `qscore.py:93–101`, `134–146`, `185–186`.

   Separation is also wrong (`rscore.py:341–367`): it accepts pooled N0/N0s coverage without eight N0 cases and accepts C1/C2 comparisons with fewer than eight comparator cases. Fixtures passed with **zero N0 cases** and with **one A− case**.

   The odd slope expression is **mathematically correct when y=ln(error)**: division by ln(10) gives the requested log10 slope. Its dead conditional is cosmetic. Constant-rank inputs are not handled: they produce undefined correlation with a misleading finite permutation p-value.

   Bootstrap helpers lack robust empty/all-invalid handling; an all-zero denominator fixture raised `IndexError` at `306`. Committed `rscore.py` has no paired Wilcoxon implementation or complete label orchestration. A newly appearing, untracked `analyze.py` supplies those—including the correct Wilcoxon tie/zero selection—but is outside the committed snapshot and does not repair the qualification defects.

8. **Smoke acceptance recomputes every predicate — WRONG.**

   It trusts saved flags for unit gates, upwind, calibration, IC equality and wrong-mesh rejection (`smoke_accept.py:34–35`, `52–64`, `80–82`, `118`).

   Further false-pass paths:

   - Q-repro never asserts all six expected dev6 cases are present (`87–99`).
   - Budget-zero acceptance uses saved certificates; the independently computed LSPG discrepancy is merely reported, not gated (`107–117`).
   - Missing arm files disappear from the glob-based verification; `all()` can pass incomplete populations (`122–134`).
   - Only `.out` logs are inspected, excluding warnings/errors in `.err` (`30–33`).
   - The registered separation fixtures are absent from `test_regimes.py`.

   Thus a broken or incomplete control can escape acceptance.

9. **Staging, submission and ledger — NEEDS-FIX.**

   Correct safeguards include committed-blob staging and checkout equality (`stage.py:59–74`), GPU/f64/highest preflight (`98–111`), unique directories, two-hour smoke cap, reservation-before-submit, and campaign flock/account cap (`submit.sh:12–40`).

   Required corrections:

   - A2.6 mandates H200, but staging explicitly permits A100 (`stage.py:24`, `37`).
   - The account-cap command runs `squeue | wc -l` without remote `pipefail`; a failed queue query can become zero and permit submission (`submit.sh:30`).
   - Stage/ledger accept non-positive or non-finite reservation inputs without explicit validation (`stage.py:39–50`; `ledger.py:68–74`).
   - Production certification validates one job’s supplied projection/config hashes, not the complete A3.3 workload/reduction certificate (`stage.py:43–48`).

10. **Other crashes, paths and configuration defects — NEEDS-FIX.**

   `refslim` output layout **does match** `rscore.STORE` (`refslim.py:30–38`, `rscore.py:66`); the smoke redirect is consistent (`smoke_accept.py:37`). Required filenames and FRONT reproduction entries exist.

   A definite configuration error remains: `make_configs.py:68–69` selects A+’s PS case by **largest ν**, although the PS rule requires largest s. The generated selection is **c13; largest-s is c7**. The largest-ν exception belongs to the calibration pilot.

   No real-size solver run was performed, so this audit does not certify peak GPU memory or the two-hour runtime.

**Required fixes**

1. Enforce allowed output roots before any write/removal; explicitly reject E0-store destinations, including resolved symlinks.
2. Restore qualification’s expected-population, shape, form/step and subset-completeness checks; add the demonstrated false-pass fixtures.
3. Implement per-record ROM/reference identity checks and enforce frozen checkpoint, rotation and mode identities.
4. Preserve raw probe evidence, independently recompute its predicates, and make probe completion/resumption reliable.
5. Implement A2.3 sigma validity/null handling and all amended separation coverage rules.
6. Make smoke acceptance reject missing cases/arms/evidence, gate independent certificate comparisons, inspect stderr, and run separation fixtures.
7. Restore full-field independent decode verification, potentially on-cluster with a small persisted verification result.
8. Close E6 configuration/resume loopholes; retain and test the currently correct escalation branches.
9. Harden submission/resource validation, enforce H200, and select A+ PS c7.
10. Commit the smoke-population correction and complete statistical orchestration; handle empty/degenerate statistics, then re-audit a **fixed snapshot** including the newly added scripts.