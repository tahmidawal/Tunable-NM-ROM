**Overall verdict: NOT READY.** The core contractions and CN equations look correct, but qualification, offline verification, timing provenance, and submission gates have defects that can produce incorrect production conclusions.

Audited HEAD `ad0e42b1d9255ecb16da7a9bae1d794e81fff514` plus the working files. I made no file changes, submitted no jobs, and ran only small CPU-only checks. During the audit, another process added untracked `smoke_accept.py`, `ledger.py`, and `configs/smk1.json`; I inspected them, but they are not part of that committed snapshot.

1. **`hrtab.py` — CORRECT mathematics; WRONG fitted provenance.**

   The sine contraction is
   \[
   P^\mathsf T=R_m^{-1}Q_m^\mathsf T\Psi^\mathsf T,
   \qquad \Psi=L\,\operatorname{diag}(w)\psi.
   \]
   The code retains the required \(L\), transpose order, negative diffusion sign, and discrete eigenvalues. Bank tables correctly use \(P^\mathsf T=\mathcal M^{-1}G^\mathsf TW\) and \(L_0=-\mathcal M^{-1}\mathcal K\). Mass and stiffness use weighted values and gradients; A5’s rules, fallback, and convergence threshold are implemented.

   Fitted rows correctly subset `Gq/Gs` and multiply **both** `P` and tested rows by \(\hat w/w\). Duplicated rows correctly halve weighted tables only; NaN fixtures reach both dynamics and tested vectors.

   **Required fix:** `fitted()` copies the parent metadata, including its `fingerprint`, without recomputing it. Thus different fitted operators claim the parent’s fingerprint. Its rule hash also hashes indices and weights rather than points and weights as registered. Duplicate/NaN metadata similarly retain stale identities. See [hrtab.py:148](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hrtab.py:148).

   **NEEDS-RESTATEMENT:** the module docstring still describes the superseded mass rules.

2. **`gstep.py` — CORRECT ordinary CN path; WRONG zero-scale certificate.**

   Correct:

   - CN residual and analytic Newton Jacobian.
   - Extrapolated predictor after the first step.
   - Production Gram norm: `Rm` for sine tests, `chol(Mass).T` for bank tests.
   - Fifty-step storage and six output indices at the registered timestep.
   - Budget/tolerance/nonfinite reason codes.
   - Production validity permits finite budget exits, leaving their residuals to Q1.
   - Certifier rebuilds the same residual and forms a fresh `jacfwd` Jacobian.
   - G-cert compares separately recorded solver residuals with reconstructed residuals; it is meaningful.

   **WRONG:** [gstep.py:136](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/gstep.py:136) unconditionally assigns infinity when the previous state has zero Gram norm. A2.4 requires zero residual at zero scale to certify.

   CPU reproduction: an exact zero trajectory gives solver `eta_rec=0.0`, certifier `eta=inf`.

   **Required fix:** implement A2.4’s three cases consistently in solver, certifier, and NumPy control recomputation.

3. **Q1 / `hjob.py` — WRONG registered tolerances; otherwise correct main aggregation.**

   [hjob.py:35](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hjob.py:35) uses tight/tighter tolerances \(10^{-12},10^{-13}\). A1.3 registers \(10^{-13},10^{-15}\). The comment cites “DESIGN E5R”, but DESIGN contains no E5R amendment.

   **Required fix:** restore A1.3 or complete the mandatory audited E5 reconciliation before production. P2/equivalence and fallback status also need an explicit implementation/evidence path.

   Tight/tighter anchors correctly require all steps to terminate by tolerance. Distances use the five evolved scoring-lattice outputs, and the nonconverged fraction pools steps correctly.

   **WRONG coverage validation:** neither the label functions nor scorer assert the required populations and shapes. `q1_label({}, [])` returns `pass`. Reduced debug populations can receive ordinary qualification/stop labels.

   **Required fix:** validate production mesh, rank, timestep, cohorts, anchor coverage, and array shapes; explicitly separate debug/smoke outcomes from production verdicts.

4. **Q2 / `hlabel.py` — CORRECT vectors and ladder wiring; WRONG precedence.**

   Classical successor/check indexing is correct. Fitted rules correctly compare against parent Gauss128, with Gauss160 as their check. The binding vector is the full sine-tested vector or bank-tested vector; the Gram-projected quantity remains separate. `rho()` handles zero and nonfinite inputs correctly.

   **WRONG:** [hlabel.py:77](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hlabel.py:77) returns `fail` for a rollout/rho exceedance **before checking check-run validity**. A2.3 says invalid successor/check evidence is unresolved.

   CPU reproduction: valid successor, exceeded rollout bar, and invalid check → `fail`, rather than `unverified`. An invalid own run is also directly classified as a quadrature failure despite the “demonstrated bar exceedance (valid runs)” requirement.

   **Required fix:** implement the registered invalid/unresolved precedence before bar-failure classification.

   **WRONG projected-parent wiring:** [hjob_main.py:98](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hjob_main.py:98) uses the parent’s **binding** label to downgrade both binding and projected fitted labels. The separately reported projected counterfactual must use its corresponding parent criteria.

5. **`fit.py` / fitting orchestration — CORRECT objective and NNLS usage; WRONG failure handling and incomplete provenance.**

   Correct:

   - Training generator matches the prior lane’s 576-row generator.
   - Registered snapshot indices for F3a/F3b/F2F3.
   - Family-specific unweighted fitting rows.
   - Snapshot-block normalization and threshold.
   - Independent residual formula equals the normalized NNLS objective.
   - Support targets, cap, deadline, first crossings, positive-weight checks.
   - Conservative path-wide drift rejection.

   `nnls.py` is byte-identical to the consolidated prior-lane copy. Its defaults supply the registered single-index update, drop limit, refactor interval, and rank tolerance.

   Defects:

   - **All blocks dropped crashes before the handler.** `build_system()` reshapes with `(0, -1)` before `fit_family()` can inspect `snapshots == 0`. Actual CPU reproduction raises `ZeroDivisionError`. Return “not built” before constructing that reshape.
   - Only target finiteness is checked; the registered contract requires rejecting any nonfinite system entry.
   - Leakage checking covers dev6/val32 descriptors, **not test64**, despite the assertion’s docstring and registration.
   - Parent training replacement requires every solver reason to be `tol`, stricter than the registered “invalid rollout” replacement criterion. **NEEDS-RESTATEMENT or correction.**
   - Used rows are listed but not separately hashed; snapshots are hashed but not saved. Per-fit stepper provenance is not explicitly bound as A2.5 requests.

6. **Controls — several fixtures are sound, but acceptance is incomplete.**

   | Control | Verdict | Finding |
   |---|---|---|
   | C1 | CORRECT fixture logic | Selects the first valid coarse pair and requires both bars. Must demonstrate the registered population; newly added smoke config uses dev6 only. |
   | C2 | WRONG acceptance | Correct family fallback and permutation. `fired` ignores `fixture_valid`; the new acceptance script combines them, but scoring does not consume that acceptance. |
   | C3 a/b/c | CORRECT main logic | Correct affected steps, certified baseline requirement, NumPy crossing check, and pooled rejection. Perturbed trajectories are not retained for independent offline reproduction. |
   | C4 | CORRECT | NaN reaches validity and Q1 rejection. |
   | C5 | CORRECT construction; incomplete evidence | Different states, returned accumulator, synchronization, explicit table arguments. Full microbenchmark repetition arrays are discarded. |
   | C6 | incomplete | A–A is timed; core scoring never applies its acceptance predicate. New script does. |
   | C7 | CORRECT fixture | Injected dev descriptor raises. Broader leakage coverage remains incomplete. |
   | C8 | CORRECT narrow check | Swapped request identity raises; this does not establish field provenance. |
   | P1 | CORRECT algebraic fixture | Full parent weights reproduce R. |
   | P3 | CORRECT control | A5 convergence and positive definiteness checked; production `bank_ok` itself uses only convergence. |
   | P4 | WRONG completeness | Can report `passed=True` when no fitted fixture exists; Jacobian checks use only the first dev case. |
   | P5 | WRONG A6 coverage | AD-Laplacian and energy formulas are correct, but use only `states[0]`, not all dev6 trajectories. Host assembly comparison is correct. Descriptive mesh stiffness and FD-RHS comparisons were removed rather than retained. |
   | P6 | CORRECT fixtures, conditional on Q2 fix | Budget-miss verification and NaN successor/check fixtures exist. P6c only exposes precedence when the normal bars happen to pass. |

   **Critical acceptance gap:** [hscore.py:242](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hscore.py:242) never gates qualification on control success. Neither staging nor submission requires smoke acceptance.

   The newly added `smoke_accept.py` does not close this: missing C1/C2/P3/P5 entries can be omitted from the acceptance set, several checks trust summary booleans, and it is not wired into production submission or scoring.

7. **Timing — CORRECT pairing/statistics; WRONG defaults and output linkage.**

   A–B–A ordering, synchronization, median per-case ratios, case-level bootstrap, 2000 resamples, and seed zero are correct. Fitting finishes before query timing.

   Defects at [hjob.py:294](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hjob.py:294):

   - Defaults are **3 repetitions / 2 burn-ins**, not registered **5 / 3**. The new smoke config also uses 3/2.
   - `timed_W_sha` comes from an **additional untimed invocation**.
   - Neither that hash nor `timed_valid` is checked against scored outputs.
   - Evolution-only results are discarded without a parity check.
   - Timing records do not bind their actual tables to verified fingerprints.
   - Registered descriptive ratios of separate medians are absent.

   **Required fix:** retain/check outputs from actual timed invocations against scored trajectories, enforce the registered timing settings, and persist microbenchmark repetitions.

   No large online table closure was found: fused query, evolution, and Jacobian benchmark receive tables as arguments. Setup/control bank evaluators capture model parameters, but not the large online tables or fitting system.

8. **`hscore.py` — WRONG; principal production blocker.**

   **Offline verification is incomplete.**

   - Fitted arms call only `offline_rho_fit()`, so their rollout distances are never independently recomputed.
   - Check-rule trajectories are not saved; no arm’s successor/check distance can be independently verified.
   - Projected rho is not independently checked.
   - Q2 labels are trusted from JSON rather than reconstructed from verified evidence.
   - Rho agreement uses \(10^{-8}\) relative plus \(10^{-12}\), looser than the registered agreement requirement.
   - Fitted rho comparison can accept finite-versus-infinite disagreement: its inequality becomes `inf <= inf`. Confirmed by CPU reproduction.

   See [hscore.py:136](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hscore.py:136).

   **E0 identity is only partially enforced.** ST scoring and loader request checks are correct, but the prior scorer’s store pin is optional; no mandatory lane pin is supplied. `score_arm()` constructs identity from array labels, omits arm/rule assertions, and discards metric/kind information from saved scoring records. The current local bank/rotation are not checked against job hashes before independent decoding.

   **Eligibility and invalid handling are wrong.** The common set is intersected across whatever arms/cases happen to exist, allowing missing coverage to silently shrink it. Any invalid case, including an ineligible one, sets worst error to infinity and `worst_case=None`. An invalid R then causes `no_worse()` to raise `KeyError(None)` before reference-unqualified precedence is reached. Confirmed by CPU reproduction.

   **Stop rules:**

   - Basic `no_worse`, `faster`, and `gain15` formulas are correct for complete finite evidence.
   - F2’s ordinary identified-minimum/incomplete branches mostly implement A3.1/A4.1 correctly.
   - **F3 violates A2.3:** `q_failed` treats a failed Q1 or Q2 as fully resolved even when other registered evidence is absent or Q2 is unverified. CPU reproduction with six Q1-failed candidates, invalid checks, and no cost/error evidence returns **“fired (tested failure)”**.
   - Projected F2 selection can lack Q1 evaluations because orchestration selects candidates using binding Q2 only.
   - F2+F3 comparison against F2(\(m^\*\)) is absent.

   **Required fix:** reconstruct acceptance from complete saved evidence, enforce population/identity pins, distinguish incomplete from tested failure literally, and make reference-unqualified handling robust before comparisons.

9. **`resources.py` — WRONG as a production certificate.**

   [resources.py:18](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/resources.py:18) does not establish the registered reservation:

   - Assumes “up to two F2 Q1 blocks”; several Q2-passing candidates can fail Q1 before the first qualification, requiring more.
   - Missing smoke phases default to zero without checking mandatory coverage.
   - No same-size/device/completed-smoke validation.
   - No implementation of ordered deferrals or mandatory-work coverage.
   - No enforced ledger input; spent defaults to zero.
   - No memory feasibility gate or required fallback sampling when device counters are unavailable.
   - `resources-*.json` is never required by staging/submission.

   **Required fix:** validate a complete real-size smoke, project all possible required candidate work, bind certificate/config/ledger hashes, and enforce it before submission.

10. **Staging/submission — CORRECT committed-tree staging; WRONG enforcement gaps.**

   `stage.py` correctly stages Git blobs, checks present checkout files against them, creates checksums, uses the GPU partition, absolute cluster Python, f64/highest precision, paralab paths, and backend preflight exit 42.

   `submit.sh` uses the specified campaign lock and checks `squeue` before/after.

   **Required fixes:**

   - Move the **lane-cap check inside the shared lock**. Two concurrent submissions can both pass the current pre-lock check.
   - Require accepted smoke evidence, valid resource certificate, ledger reservation, and production configuration before `sbatch`.
   - Validate the registered smoke reservation rather than allowing arbitrary `--hours`.
   - Bind staged config identity to the certificate and attempt.

   See [submit.sh:10](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/cluster/submit.sh:10).

The debug summaries support ordinary-path execution only. They do not validate real-size acceptance, and their P5 result is the superseded pre-A6 fixture. Production requires the fixes above, an audited tolerance/E5 reconciliation, and a real-size smoke demonstrating the corrected controls and acceptance path.