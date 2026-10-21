**NOT READY for the real-size smoke.** The stepper algebra is sound, but certificates can accept incorrect trajectories/outputs, several required controls are missing, and qualification and timing enforcement are incomplete.

I made no file changes, submitted no jobs, and wrote no scratch files. I inspected the existing CPU result and ran small CPU-only mutation checks.

1. **CORRECT — precontraction and three stepper equations.**

   [fscore.py:101](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fscore.py:101) implements the specified QR/eigenbasis maps and precontracted nonlinear force. CN uses the correct residual, extrapolated predictor, Newton Jacobian, and tolerance proportional to $\|z_n\|$, with A1’s revised tolerance levels. ETDRK4 matches the registered recurrence and contour coefficients. The one-pass linearised CN equation is correct. Coefficients and coordinate maps execute inside evolution.

   This verdict covers the formulas; their independent verification has holes below.

2. **WRONG — certificates can pass incorrect runs.**

   I reproduced two failures with a one-coordinate CPU fixture, original diffusion eigenvalue 2, zero advection, and horizon 0.25:

   | Mutation | Certificate mismatch | Returned final coefficient | Correct coefficient |
   |---|---:|---:|---:|
   | Transformed diffusion eigenvalue changed to 3 | 0 | 0.47236655 | 0.60653066 |
   | Output map `RinvV` replaced by zero | 0 | 0 | 0.60653066 |

   Both also returned `out_vs_states=0`.

   The ETD checker’s coefficients are independently *evaluated*, but their eigenvalues come from the same transformed `D` as the stepper ([fsrun.py:207](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:207)). The checker therefore accepts a shared diffusion error. The stored eigen-residual is diagnostic, without an acceptance threshold.

   Certificates check `Z`, while `out_vs_states` compares `z_out` with `Z`; neither verifies returned `W` against an independently reconstructed coordinate map. Initial-state consistency is also unchecked. Twins share these maps, so twin agreement does not close the hole.

   **Fix:** enforce original-table diffusion/eigenbasis consistency; independently verify initial coordinates, every returned coefficient against the scheduled state, and decoded fields against those coefficients. Make these checks qualification gates, with mutation controls.

3. **WRONG — Q1/Q2 evidence is incomplete.**

   The bank loops correctly compute successor rollouts for every configured case and evaluate both forces at the main-rule output states. Configuring `all38` provides the required rollout coverage.

   However:

   - [fsrun.py:74](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:74) adds an unregistered LSPG anchor exception: budget/tiny-step exits may pass at `100 * gtol`. It also uses recorded solver stationarity rather than freshly certifying the anchors.
   - CN `budget_unconverged` is recorded but does not make `valid=False`, contrary to §5.
   - CN’s `cert.ok` applies the 1% allowance per run; no configuration-level pooled qualification is implemented.
   - Tight/tighter states and per-step certificate evidence are not fully retained for independent checking.
   - LSPG step-halving evidence required on dev6 is absent.

   **Fix:** use the registered anchor contract with fresh certificates, enforce budget-exit invalidity, implement pooled qualification, and persist the evidence needed to recompute it.

4. **WRONG overall — monitors are present; diagnostic coverage is incomplete. A4 itself is CORRECT.**

   [fsselect.py:102](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsselect.py:102) implements A4’s exhaustive classifier correctly. All six specified boundary fixtures returned `unresolved` in my check.

   Accepted-step growth and ETD stage finiteness are monitored. But [fsrun.py:353](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:353) truncates classification to eight tripped runs, explicitly skips LSPG refinements, and provides no corresponding POD classification. The registered diffusion-coordinate alternation diagnostic is absent.

   **Fix:** classify every tripped run using its own certified refinement scheme, including LSPG/POD, and implement the alternation diagnostic. Any deferral requires an explicit specification change.

5. **WRONG — E0 field identity is not guarded; cohort restriction is CORRECT.**

   The lane’s case generator/selector admits only dev6 and val32; sealed/OOD access is additionally blocked by `refstore`.

   However, [fsrun.py:87](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:87) passes an unlabelled field to `RS.score`, then checks that the returned identifiers equal the identifiers supplied. A swapped field still passes. This bypasses `p1score.score`’s case/time identity checks. C12 does not exercise the required E0 case-swap detector.

   **Fix:** carry field provenance through scoring; verify case, schedule, reference kind, pinned store/metric, and reference digest. Exercise a real case-swap failure.

6. **WRONG overall — POD equations/providers are correct; verification and labels are incomplete.**

   The central Galerkin ODE, ECSW stencil force, full-node provider, and un-precontracted nonlinear twins match the design. Successor construction skips identical rule hashes correctly.

   In [podarm.py:90](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/podarm.py:90):

   - Configured `phi_sha256` and `native_sha256` pins are not enforced.
   - POD transformed tables lack the registered manifest guard.
   - Q2-POD distances/ratios are recorded, but successor validity/stability/certification, trivial-indicator handling, and HR qualification labels are not aggregated.
   - Duplicate rules remain separate reported/timed members.
   - Native 5L tight runs are not certified; tighter anchors and tight/tighter distances are absent.
   - Native per-step states/certificate arrays are discarded.
   - POD reconstruction shares the diffusion/map verification holes in item 2.

   **Fix:** enforce artifact identities, complete native anchor certification and persistence, and implement the registered Q2-POD and duplicate-rule reporting rules.

7. **WRONG — timing machinery exists, but A3.6 is not enforced.**

   [fstiming.py:123](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fstiming.py:123) implements warm-up, burn-in, E/F/D triplets, and two drift retries. Ratio directions are consistent: E is incumbent/subject; F and D are B/A. Direct matched-competitor and CN–R1 triplets are scheduled.

   Blocking gaps:

   - Warm-up only checks coefficient closeness to scored output at $10^{-12}$; subsequent hashes compare against warm-up. This silently weakens the registered scored-output identity requirement.
   - Fused hashes omit fields and states.
   - Identity failures in discarded retry attempts can disappear because the final attempt determines validity.
   - No job-wide identity-failure gate makes every speed condition fire.
   - Coverage is whatever `dev` and `reps` the config supplies. My three-record, one-case fixture returned `status='ok'`; six cases are not enforced.
   - In-job selection chooses matched competitors, but never the registered best stepper.

   **Fix:** enforce scored identity and full-block failure propagation, require six distinct dev6 cases and three distinct valid repetitions for headline statistics, label smoke subsets incomplete, and implement/freeze best-stepper selection.

8. **WRONG — the control summary is not the registered acceptance gate.**

   [fscontrols.py:203](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fscontrols.py:203) can report `all_ok=True` without:

   - C9 timing self-comparison;
   - C10, C17, C19–C21;
   - C11 evidence imported from the ECSW job;
   - C23/C24, required by A3 despite the requested C1–C22 grouping.

   The referenced `checks.py` does not exist in the audited tree. C2 does not verify anchor validity. The summary ignores C3’s positive order threshold and C5’s positive slope threshold. C12’s table-swap branch manually raises after comparing fingerprints rather than invoking the production guard; its E0 identity test is missing.

   The existing CPU run reports `all_ok=True`, but that is only this incomplete summary.

   **Fix:** implement one exhaustive control inventory and gate. Require each registered witness, positive threshold, escalation result, and external C11 artifact; missing evidence must block acceptance.

9. **NEEDS-RESTATEMENT — array passing is sound, but resource readiness is unproven.**

   I found no obvious captured full bank/POD arrays in the reviewed main JIT paths; large tables are passed as arguments.

   Calculated storage alone is approximately:

   | Allocation | 1024² | 4096² |
   |---|---:|---:|
   | Rank-384 bank | 2.99 GiB | 47.98 GiB |
   | 38 full initial fields | 0.30 GiB | 4.75 GiB |
   | One six-output full decode | 0.047 GiB | 0.750 GiB |

   These exclude construction temporaries, POD bases and rotated copies, tables, executable caches, and concurrent output buffers. Repeated `Setting.check` calls scan the full bank for fingerprints, adding substantial untimed work. Certificates, long refinements, and full-node POD evaluations also need explicit pricing.

   The existing small CPU run took 1638.3 seconds; it cannot certify an H200 runtime projection.

   **Fix:** retain the mandatory real-size memory pilot and measure setup, compilation, certification, guards, POD-none, timing, and refinement costs separately. Treat current runtime estimates as provisional.

10. **WRONG — outputs cannot yet support the complete pre-registered conclusion.**

   [fsrun.py:235](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:235) allocates saved `w0` arrays filled with NaNs and never populates them.

   More broadly, the driver writes `complete=True` and `COMPLETE` without an acceptance gate. There is no implemented full Q1/Q2 aggregation, frozen best-stepper result, separate all-case/eligible worst-error predicates, complete F5 outcome, or population-specific match reporting. Several discarded evidence arrays would require rerunning work to support an independent checker.

   **Fix:** distinguish execution completion from numerical/control acceptance; populate initial states, preserve required evidence, and implement the checker and registered decision outputs before the smoke.

The immediate blockers are certificate independence, scored-output identity, anchor validation, and the missing acceptance/control layer. Fix those and re-audit before spending the real-size smoke allocation.