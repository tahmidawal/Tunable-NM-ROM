**fom1k: NOT CLEAN.**  
**b3d65: NOT CLEAN.**

Reviewed HEAD `2f13c1c4e`. No files modified, lab-log entry appended, or jobs submitted. Checks included syntax parsing, small CPU algebra checks, reference inspection, and synthetic execution of audit logic. I did not rerun the reported GB10 experiment or benchmark either production workload.

**A. FOM — dispositions of every numbered finding in `codex-fom-2.md`**

1. **RESOLVED / CORRECT — diagnostic contract.**  
   A10.1 explicitly supersedes the per-Newton requirement, and `fom2.py`’s description now matches its per-step arrays. The registered schedules fit the buffer.

2. **STILL-WRONG / WRONG — calibration independence.**  
   Exact case identities, nonnegative discrepancies, saved calibration fields, fallback status, and tolerance linkage are substantial fixes. However:

   - Tight/candidate verification still comes from Boolean flags; calibration diagnostics are discarded.
   - The audit selects tolerances using reported `dd_job`, allowing a discrepancy of up to `1e-6` from the reconstructed value. That allowance can straddle the actual acceptance threshold.

   See [audit_fom.py:91](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:91).  
   **Fix:** save calibration diagnostics, reconstruct verification, and apply the threshold to reconstructed discrepancies. Save F64 evidence or define an explicit uncertainty interval for F32 evidence near the threshold.

3. **RESOLVED / CORRECT — accuracy–timing linkage and case indexing.**  
   ROM timings now hash coefficients, which the audit independently decodes for accuracy. This avoids requiring NumPy/JAX field decodes to match bitwise. FOM timing fields are hashed and checked against their F32 casts. Explicit timing-case indices fix cohort-order dependence. See [audit_fom.py:141](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:141).

4. **STILL-WRONG / WRONG — cross-job agreement is incomplete and unpinned.**  
   Missing matches are silently skipped; **one matching row is sufficient** to make the cross-job gate eligible to pass. Files come from an unrestricted `a1k*` glob without requiring accepted audits, matching mesh/rule, or fixed artifact identities. Currently only `a1kfast` has a pulled `archive/output/result.json`; absent acc evidence would not itself fail this check. See [audit_fom.py:168](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:168).

   **Fix:** require all **3,040** arm/case matches for this config, authenticate the designated source artifacts and their audits, and verify mesh/rule/config compatibility. Missing evidence must mean incomplete.

5. **STILL-WRONG / WRONG — order verification remains asserted, not reconstructed.**  
   Exact record/factor inventories and finite positive self-differences are fixed. But the runner saves only order fields and aggregate failure counts; the audit cannot reconstruct those counts from diagnostics. See [fomrun.py:110](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fomrun.py:110).

   **Fix:** retain order-trajectory diagnostics and verify their active lengths, tolerances, residuals, and failure counts.

6. **STILL-WRONG / WRONG — evaluation verification can still be bypassed.**  
   FOM verification now uses `step_rel`, but does not require the expected number of steps or diagnostic entries. Empty residual arrays pass `np.all`; changing `steps` to zero and clearing failures can therefore manufacture verification. Newton/linear-residual inventories are not validated. ROM verification remains only `verified == (nfail == 0)`, without consistency checks against other statistics. See [audit_fom.py:130](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:130).

   **Fix:** derive expected steps from the registered schedule, require complete active diagnostic arrays, validate aggregates, and explicitly check field/coefficient shapes and finiteness—including row zero. Check ROM statistics against the form-specific acceptance rules; persist per-step evidence where aggregates cannot establish them.

7. **RESOLVED / CORRECT — timing values and rejection-test mechanism.**  
   Durations must now be finite and positive; ratios/drift are reconstructed; hashes bind to saved evidence. The twelve mutations execute the same validator used for acceptance. Their coverage is still insufficient for findings 2, 4–6, but the previous mechanism defect is fixed.

8. **RESOLVED, with NEEDS-RESTATEMENT — progress preservation and resource estimate.**  
   Evaluation arrays now checkpoint every eight cases, and timing JSON every fifty triples. That addresses the specific previous recoverability complaint. Calibration arrays still save only at phase end; there is no resume implementation or transactional multi-file checkpoint.

   Memory remains structurally plausible. Saved calibration fields add approximately **0.567 GiB** to the previous inventory. Six hours remains a planning reservation, not a demonstrated production runtime.

9. **A11: CORRECT exemption; NEEDS-RESTATEMENT causal claim.**  
   Exempting undamped CN from a *real-data asymptotic-order gate* is honest **provided the amendment remains disclosed as post-observation**, CN’s failed rates remain visible, and CN receives no established-second-order claim. The code preserves finite-difference and zero-failure requirements while exempting only the rate band.

   CN-R passing on the same case and CN passing on another case support the stiff-transient explanation. They do **not prove** “not a coding error,” and large self-convergence rates alone do not establish ringing.

   **Fix wording:** “Consistent with a pre-asymptotic stiff-mode transient; second-order convergence is not established on this case at these steps.” Preserve an independent algebraic/manufactured CN correctness gate. A11 itself is not my submission blocker.

**B. 3D**

10. **CORRECT — advection, Jacobian, projection, and trust.**  
    [t3run.py:61](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/t3run.py:61) implements the vendored point-form advection and exact Jacobian. A small CPU comparison gave advection agreement within `3.6e-15` and identical Jacobians.

    `TB.project` correctly solves the Gram-system L2 projection; decoding uses the matching bank-row blocks. Trust is the registered `0.05 × coefficient_rms_spread`, approximately `0.005883` for both ranks. Current model/rule hashes match the config.

11. **CORRECT — lattice, reference layout, and output schedule.**  
    At `n=65`, `lattice65_index` is exactly `arange(63³)`, and independently generated lattice coordinates equal the mesh’s interior coordinates.

    The actual reference has:

    - `n=513`, `dt=0.0025`, seed `923801`, count `64`.
    - Keys `c<j>`, F64 arrays shaped `(6, 250047)`.
    - A file hash matching its accepted `.done` record.

    Both steppers store the initial state at row zero and subsequent outputs at `0.05` intervals. The driver checks reference row zero against the regenerated initial field. No indexing or schedule crash was found.

    **Hardening:** explicitly assert reference mesh, step, shape, and output-time contract; several are currently recorded rather than enforced.

12. **CORRECT — shell completion and basic memory scale.**  
    The actual completed counts are **M=2052** for `R′=512` and **M=1027** for `R′=256`. `tensor=False` avoids constructing the large quadratic tensor.

    | Storage/component | Approximate GiB |
    |---|---:|
    | Full 512-column interior bank | 0.954 |
    | gl24 B, D, P blocks | 0.317 |
    | lat4096 B, D, P blocks | 0.047 |
    | All 132 restricted trajectories retained for one case | 1.475 |

    These are components, not peak-memory measurements. Construction temporaries, compiled workspaces, and duplicate buffers add overhead, but I see no structural 80-GB/H200 memory blocker.

13. **CORRECT runner; WRONG audit — deployed comparator evidence.**  
    The runner calls the actual fixed-sweep BE implementation at `dt0=0.01`, including three adaptive startup steps, and uses it as timing A. Both query paths include initial projection and field decoding.

    However, comparator coefficients/fields are not saved. The audit checks only the comparator row inventory—not its errors, anchor distance, generic-BE distance, finiteness, or solver health. See [audit_t3.py:98](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_t3.py:98).

    **Fix:** save comparator output coefficients and diagnostics; independently reconstruct every reported comparator metric. State explicitly how a fixed-sweep output becomes eligible—the deployed algorithm does not guarantee the adaptive stopping criterion.

14. **WRONG — 3D timing hashes remain disconnected from saved accuracy evidence.**  
    Both “expected” hashes are trusted JSON strings, and drift is unchecked. I executed the **complete validator** with exact configured inventories, NaN comparator errors, NaN drift, and invented matching hashes: every returned check passed. See [audit_t3.py:109](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_t3.py:109).

    **Fix:** use coefficient hashes bound to saved generic/comparator trajectories, as the FOM audit now does; reconstruct drift and validate the baseline identity. Add mutations for these failures.

15. **CORRECT rescoring path; WRONG completeness claim.**  
    Evaluating bank features at lattice coordinates provides a useful separate path from the runner’s mesh-table decode. It is not an independent integrator check.

    The audit ignores `e_ref_per_time` and initial-row correctness, and verifies solver acceptance only through `nfail`. It also does not authenticate the decoding bank against the config/result, validate rule identity, or enforce the manufactured-test certificate. See [audit_t3.py:65](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_t3.py:65).

    **Fix:** validate all six times, independently check initial projection, check solver-statistic consistency, authenticate model/rules/certificate, and add time-shift/reference/initial-state mutations.

16. **NEEDS-RESTATEMENT — anchor distances are available; anchor eligibility and order claims are not yet enforced.**  
    The runner correctly computes evolved-time reference errors, anchor distances, self-differences, production–tight distances, and tight–tighter sensitivities. The required anchor trajectories exist.

    Neither this driver nor audit implements the complete A2.8/A3/A4 claim decisions. Raw distances can exist even when anchor/tighter trajectories fail verification.

    **Fix:** require a reporting gate that reconstructs anchor uncertainty and eligibility, masks unresolved solve sensitivities, and applies the 16-case thresholds: at least **13** valid cases for 80%, with **8** as the half-cohort adjacent-check threshold. Until then, label anchor quantities as distances, not validated continuous-time errors. Keep reference errors explicitly provisional.

17. **NEEDS-RESTATEMENT — wall time and interruption recovery.**  
    The config entails **4,224 adaptive accuracy rollouts and 391,040 time steps**, plus comparator runs and timing. There are **768 A–B–A triples**, with at least **256.8 seconds** of configured burn-in including warm-ups.

    One gl24 Jacobian contraction costs approximately **29 GFLOP**; lat4096 costs **2.15 GFLOP**. Runtime depends heavily on realized LM iterations—the budget is 600 per step. A sub-minute toy check cannot establish the six-hour production reservation.

    Coefficients save only after all sixteen cases of an arm; timing records save only after that arm’s entire timing phase. **Fix:** checkpoint coefficients per case and timing records in batches; obtain a representative real-size cluster timing before treating six hours as validated.

18. **CORRECT principal staging; minor unnecessary dependency.**  
    The 3D staged file set covers the inspected imports. GPU partition, precision, GPU preflight, committed-byte checks, and current manufactured certificate are present. CPU imports and syntax checks passed.

    [stage.py:65](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/cluster/stage.py:65) nevertheless reads the 2D reference manifest before branching into 3D. It exists here, so this is not a present crash, but the 3D stage unnecessarily fails if that unrelated artifact is absent. Move that read inside the 2D branch.

**Final gates:** hold **fom1k** for calibration/verification evidence and complete authenticated cross-job matching. Hold **b3d65** primarily for comparator evidence, disconnected timing hashes, and incomplete acceptance checks. Neither hold is based on a demonstrated core-algebra or GPU-memory defect.