**Overall verdict: NOT CLEAN.** The core LMM formulas are sound, but GAL verification has a demonstrated false-acceptance bug, and the validation pipeline has material gaps.

No files were modified. I inspected the saved all-pass manufactured-test report and ran two small, read-only CPU counterexamples. I did not run the production GPU grid. Paths below are relative to `experiments/jcp-time2/`.

1. **WRONG — GAL can verify a non-root stationary point.**  
   `t2core.py:180` accepts either stationarity or residual convergence for **both** forms. With GAL’s `gtol=0`, an exactly zero gradient passes even when the root residual exceeds tolerance. My counterexample returned `nfail=0`, five stationary exits, and `worst_tolratio=1e10`.  
   **Fix:** make verification form-specific: GAL requires finite `rn <= tol`; LSPG permits the stationarity alternative. Retain rejection of reasons 0/3. Add this counterexample to G2a. **Invalidates affected GAL trajectories, anchors, order estimates and selections.**

2. **WRONG — the manufactured residual checker shares the production specification.**  
   `test_lmm.py:101–103` imports coefficients and startup lengths from `T2.SCHEMES`. Thus a wrong production table also changes the supposed independent check. Changing CN-R to **one** BE startup step in memory still passed: finest order `1.98777`, step-check residual `2.98e-15`.  
   **Fix:** independently hard-code the specified coefficients and startup schedule in the checker; assert production-table equality separately. **Invalidates the claimed independent startup/coefficient validation, although the current production table is correct.**

3. **WRONG — the generated grid omits required runs.**  
   `make_configs.py:8`, all three configs’ `tighter_factors`, and `t2run.py:55–64` omit tighter replays at `2Δt₀`. Consequently the reported coarse dyadic triple cannot obtain its required solve-sensitivity validity check. TH06 tight runs also omit `5Δt₀` and `10Δt₀`, despite section 3 specifying the tight grid. Each current configuration generates 218 unique runs per setting/case.  
   **Fix:** include factor `2` in tighter replays; use the full tight ladder for TH06, or explicitly amend that requirement. **Invalidates completeness claims and any coarsest-triple claim made without its missing replay.**

4. **WRONG — vendor timings bypass G4 output verification.**  
   `t2run.py:328–329` deliberately omits `B_matches_accuracy` for the vendor subject. `audit_t2.py:205` treats that missing check as `True`. The vendor compiled-cache size is also absent from the before/after records.  
   **Fix:** retain the vendor accuracy-phase hash for every timing case, compare every timed vendor output against it, and include `vq` in cache checks. Persist expected and observed hashes. **Invalidates acceptance of the vendor overhead timing comparison under G4.**

5. **WRONG — the independent audit can certify incomplete evidence.**  
   `audit_t2.py:91–128`, `136`, `167`, and `202–217` have several fail-open paths:
   - `--max-cases 1` can still produce `all_pass=true`.
   - Run/case inventories come from the artifact rather than an independently required inventory.
   - Missing pair metrics are silently skipped.
   - Missing timing skips G4; empty invocation lists satisfy `all(...)`.
   - Output times are assumed rather than explicitly pinned.
   - Nonfinite flags are informational; precision and G2a evidence are not acceptance checks.

   **Fix:** require exact inventories, unique rows, expected shapes/times, mandatory metrics and timing sample counts; distinguish partial diagnostics from acceptance; include every required gate. **Can invalidate acceptance of truncated or incompletely checked results.**

6. **WRONG — a failed audit exits successfully.**  
   `audit_t2.py:218–227` prints `AUDIT FAILED` but returns normal process status.  
   **Fix:** exit nonzero when `all_pass` is false. **Invalidates automated acceptance if downstream commands rely on exit status.**

7. **WRONG — lane-cap enforcement is not atomic.**  
   `cluster/submit.sh:10–17` checks the queue before transferring and submitting. Two simultaneous invocations can both observe zero lane jobs and submit different attempts. Only RUNNING/PENDING states count, excluding other live allocation states.  
   **Fix:** hold a shared, paralab-backed submission lock across the queue check and `sbatch`; reject any live `t2_` job. Atomically create the attempt directory. **Operational defect; overlapping jobs would violate the required lane cap.**

8. **NEEDS-RESTATEMENT — some manufactured mutation assertions are weaker than their labels.**  
   `test_lmm.py:114–122`, `176–184` aggregate residuals over five steps and oracle errors over five output times. A2.10 specifically names the first output step and final-time oracle error for m4. The current m4 does pass the intended final-time comparison in my probe, but the code does not assert it.  
   Also, `t2core.py:120–140` changes D for m5 while retaining a Jacobian simplification valid only for the correct D, so that mutant has an inconsistent Jacobian.  
   **Fix:** assert the named step/time checks explicitly; use the unsimplified derivative for a mutation intended to isolate incorrect weighting.

9. **NEEDS-RESTATEMENT — vendor BE bit-compatibility is not established.**  
   `t2core.py:118–140` is algebraically equivalent to vendor BE, but contains traced coefficient multiplications and extra history terms absent from `vendor/quad2d/qcore.py:291–310`. Identical floating-point evaluation is not guaranteed. The copied LM logic otherwise matches the vendor’s `q=0` specialization.  
   **Fix:** describe compatibility through the specified G1a tolerances, not bit identity. If literal arithmetic identity is required, provide an explicit vendor-expression BE branch and validate it on the target GPU.

10. **NEEDS-RESTATEMENT — G2a evidence is not bound to the staged code.**  
    `cluster/stage.py:20–25`, `45–57`, and `104–106` neither stage nor validate the manufactured-test certificate. The supplied passing report exists, but carries no source hashes establishing which stepper/test revision passed.  
    **Fix:** record and stage a passing certificate containing hashes of `t2core.py`, `test_lmm.py` and the applicable design; reject mismatches before submission.

11. **NEEDS-RESTATEMENT — raw distances are not yet valid anchor/order results.**  
    `t2run.py:234–259` computes anchor discrepancies and `s_h` even for unverified trajectories. That is acceptable as raw evidence, but A2.8/A2.11 eligibility, unresolved labels and A4 claims must be applied downstream. The named `analyze_t2.py`/`make_report.py` are absent. Production-versus-tight distances are also absent for old/HQ arms because those arms have no tight counterparts.  
    **Fix:** implement the downstream verification masks and uncertainty rules before interpretation; clarify or complete the “every arm” production-versus-tight requirement.

12. **CORRECT — unmutated residuals and Jacobians.**  
    `t2core.py:114–142` implements the specified LMM residual. The LSPG Jacobian simplification is valid with the correct D; GAL uses the correct projected residual and derivative.  
    **Fix:** none beyond the verification correction in item 1.

13. **CORRECT — startup, history, conditional spatial evaluation and output indexing.**  
    `t2core.py:162–168`, `198–207` correctly selects BE startup coefficients, shifts the two-step history, evaluates \(F(c_n)\) under `lax.cond`, and stores the newly solved state at the required output index. Schedule arguments have fixed traced dtypes.  
    **Fix:** none for the configured schedules.

14. **CORRECT — A2.14 alternation diagnostic.**  
    `t2core.py:157–158`, `183–208` uses the specified stiff-mode mask, amplitude threshold, consecutive-event counter, numerator restricted to runs of at least three events, and amplitude-qualified denominator.  
    **Fix:** downstream labeling must use `alt_index > 0.5` and the word **alternating**.

15. **CORRECT — metric formulas and permutation bookkeeping.**  
    `t2run.py:204–229`, `241–275` uses the correct initial-field normalization, evolved-time maximum, TX formula and pair distances. The `order` permutation is handled correctly: coefficients retain original run indices, row updates follow execution order, and G1a iteration lookup reverses that permutation correctly.  
    **Fix:** none to these calculations; apply item 11’s eligibility rules.

16. **CORRECT — reference checks and ordinary staging safeguards.**  
    `t2run.py:125–145` checks reference acceptance, mesh/tolerance contracts, cohort hashes, shapes, finiteness and array hashes. `cluster/stage.py:45–74`, `89–108` stages pinned source bytes and references, verifies checksums, uses the required absolute venv, f64/highest precision, GPU partition, paralab paths and GPU-preflight exit 42.  
    **Fix:** retain these safeguards; add items 7 and 10.

17. **CORRECT — bounded rollout memory for the configured jobs.**  
    `t2core.py:153–207` retains six coefficient outputs and accumulated statistics, not every time step. `t2run.py:207–239` retains shared-node fields across runs, approximately **659 MiB per case** for the current grid, plus the bank and other live arrays. Memory does not grow with the number of time steps.  
    **Fix:** none required for the current grids; reassess peak memory before larger-mesh replication.

18. **CORRECT — A4 closes design-audit item 12.**  
    `DESIGN.md:462–465` explicitly defines the zero/insufficient adjacent-coverage outcome, qualified primary-only claims, and rejection when sufficiently populated adjacent checks fail. It resolves the ambiguity in `audits/codex-design-r3.md`.  
    **Fix:** implement those branches exactly in the eventual analyzer, after checking primary eligibility.