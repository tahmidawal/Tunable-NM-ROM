**Overall verdict: NOT CLEAN.** The core fixes substantially improve the implementation, but reporting still permits unsupported claims, the audit still accepts incomplete timing evidence, and submission staging has a concurrency defect.

I made no file changes and submitted no jobs. Validation included source inspection, read-only synthetic counterexamples, certificate-hash checks, grid enumeration, and in-memory rendering plus `bash -n` of the smoke batch script. Paths below are relative to `experiments/jcp-time2/`.

1. **Previous item 1 — RESOLVED: GAL verification.**  
   `t2core.py:182–186` now requires residual convergence for GAL while retaining stationarity-or-residual acceptance for LSPG. Budget and damping-limit exits remain rejected.  
   **Fix:** none required for the identified false-acceptance bug.

2. **Previous item 2 — RESOLVED: independent manufactured specification.**  
   `test_lmm.py:101–108` now uses an independent coefficient/startup table and separately checks production-table equality.  
   **Fix:** none.

3. **Previous item 3 — RESOLVED: missing grid runs.**  
   Tighter replays now include \(2\Delta t_0\); TH06 tight runs cover the full ladder. Enumeration produces **232 unique runs per setting/case**: 180 main, 40 HQ, 12 old.  
   **Fix:** none to the grid. The report still omits the coarse self-convergence triple; see item 17.

4. **Previous item 4 — RESOLVED: vendor timing verification.**  
   `t2run.py:269–270,330–335` stores the vendor accuracy hash, checks timed vendor outputs, and includes the vendor compiled-cache size.  
   **Fix:** none to the driver; the independent audit’s completeness checks still need strengthening.

5. **Previous item 5 — STILL-WRONG: incomplete evidence can still pass the audit.**  
   The new inventory, mandatory-metric, finiteness and partial-audit checks are useful, but `audit_t2.py:246–267` still has gaps:
   - G1a requires a nonempty list, not exactly one unique record per expected case.
   - Timing counts use the artifact’s own `tm['candidates']`, not an independently derived subject inventory.
   - Repeating one valid timing record to the required total satisfies the count check; unique `(subject, case, repetition)` coverage is unchecked.
   - Cache dictionaries need only compare equal; mandatory query keys, including vendor, are not required.
   - Stored hash-equality booleans are trusted without checking the persisted expected/observed hashes.
   - Output-time coordinates are still implicit rather than explicitly pinned.

   **Fix:** derive expected evidence from the pinned staged configuration; require exact unique G1a/timing inventories, mandatory cache keys, consistent hashes, and explicit output times.

6. **Previous item 6 — RESOLVED: audit exit status.**  
   `audit_t2.py:285` now exits nonzero on failure.  
   **Fix:** none.

7. **Previous item 7 — STILL-WRONG: submission is only partly atomic.**  
   The namespace lock correctly protects queue checking and submission, and all live states are considered. However, `cluster/submit.sh:13–14` deletes and uploads to the shared `.staging/$A` **before acquiring the lock**. Two invocations for the same attempt can delete or modify one another’s upload, including around its move into the final directory.  
   **Fix:** use a unique staging directory per invocation, then validate and move it under the lock; alternatively lock the entire transfer.

8. **Previous item 8 — RESOLVED: named mutation checks.**  
   m4 now explicitly checks the first step and final-time oracle error. The m5 mutant now uses the unsimplified derivative consistent with its altered weighting.  
   **Fix:** none for the cited defects.

9. **Previous item 9 — NEEDS-RESTATEMENT: arithmetic compatibility wording.**  
   The core docstring now correctly says “algebraically” equivalent and refers to G1a tolerances. But `make_report.py:422–423` still prints **“same arithmetic.”**  
   **Fix:** say “same-job G1a agreement within the specified tolerances.” Bit-identical arithmetic remains unestablished.

10. **Previous item 10 — RESOLVED under A6’s certificate contract.**  
    The committed certificate passes and its `t2core.py` and `test_lmm.py` hashes match current files. Staging checks those hashes against HEAD; the audit checks the core hash against provenance.  
    **Fix:** none for the stated A6 contract. This is narrower than the previous recommendation to include the design hash.

11. **Previous item 11 — STILL-WRONG: downstream interpretation remains incomplete.**  
    The new analyzer correctly masks unverified tight/tighter trajectories for self-convergence and checks all four anchor trajectories. A6.3 legitimately limits production-versus-tight reporting to main-rule arms. But claim eligibility, anchor interpretation and diagnostic presentation still have defects detailed below.  
    **Fix:** complete those report safeguards before interpreting results.

12. **WRONG — selection ignores required gates; H1/H2 ignore comparator verification.**  
    `make_report.py:164–234` receives no audit verdict. Eligibility uses verification count, timestep and timing availability only; failed G0/G1a/G2a/G4 do not suppress selection. The audit is merely printed later.

    A read-only counterexample produced:
    - comparator verified: **false**;
    - selection: **no verified candidate**;
    - H1: **passed**;
    - H2: **passed**.

    **Fix:** gate all selection and hypothesis outcomes on the required audit evidence and a verified comparator. Failed prerequisites should yield **unavailable**, not passed/failed scientific outcomes. Bind the audit to the supplied result’s job/commit.

    With those prerequisites assumed, the ST/TX selection inequalities, comparator self-ratio, baseline-retained outcome, and specified tie-break ordering are implemented correctly.

13. **WRONG — smoke results can become full-cohort claims.**  
    `order_claim()` correctly implements the primary/adjacent rule for 38 cases, including withholding a claim when sufficiently populated adjacent checks fail. But line 101 substitutes a half-of-small-cohort threshold for smoke runs. A two-case probe returned unqualified **“order 2.”** Selection likewise treats the observed cases as the entire required cohort.

    The report and plots then hard-code **38 cases, dev6 ∪ val32, L=1024, six timing cases and three repetitions**, even when invoked on `smk`.

    **Fix:** distinguish smoke diagnostics from scientific claim eligibility. Require the prescribed cohort for claims and selection; derive mesh, case counts and timing counts from the configuration. Label smoke outputs explicitly as machinery diagnostics.

14. **WRONG — anchor distance is presented as time error without the required qualification.**  
    `anchor_unc()` implements \(d_{8,16}+s_8+s_{16}\) correctly, and `time_error()` checks candidate and anchor verification. However:
    - The report never conditions continuous-time wording on real-data asymptotics.
    - Plot titles say “ROM time error”; the glossary calls it an estimate of error “due to the time step alone.”
    - Unresolved distances remain plotted identically to resolved distances.
    - The main table declares errors in percent but prints anchor distances as raw fractions.

    **Fix:** use **anchor discrepancy** consistently, apply A2.8’s fallback wording, show uncertainty/eligibility counts, distinguish unresolved points, and use explicit consistent units.

15. **WRONG — “quadrature sensitivity” is the wrong quantity.**  
    `make_report.py:190–191,408–420,480` labels the HQ arm’s worst ST error as quadrature sensitivity. That is not the requested distance between main-rule and HQ trajectories; equal ST errors can conceal different fields.

    **Fix:** compute the paired main-versus-HQ field distance from saved coefficients, with verification of both trajectories. Report HQ ST error separately.

16. **INCOMPLETE — timing reporting does not satisfy A2.15.**  
    Paired-ratio medians and quartiles are calculated correctly. But:
    - Ratio outlier counts are saved only in analysis JSON, not displayed.
    - Drift medians/IQRs are likewise not displayed.
    - Drift outlier counts are not calculated.
    - The `max(MAD,1e-12)` floor changes the literal prescribed outlier rule.

    **Fix:** publish per-subject sample counts, ratio median/IQR/outliers, and drift median/IQR/outliers using the specified definition.

17. **INCOMPLETE — required diagnostics and verification labels are missing.**  
    The report omits:
    - per-case/per-arm first-failed-step tables;
    - displayed production-versus-tight discrepancies;
    - full-mesh/shared-node anchor-discrepancy ratios;
    - the coarse triple starting at \(2\Delta t_0\).

    Accuracy/cost plots also connect unverified arms without marking their status. The glossary does not fully explain anchor uncertainty, adjacent-check outcomes, or all displayed quantities.

    **Fix:** expose these diagnostics and label failed/unresolved data directly in tables and plots. Keep provisional labels beside historical motivation and old-method accuracy numbers too.

18. **STAGING BLOCKER — the current checkout will fail staging, leaving a poisoned attempt directory.**  
    At inspection, `audits/codex-code-2.log` and `cluster/collect.sh` were untracked. `cluster/stage.py:42–43` rejects non-run changes anywhere in the lane. Worse, line 39 creates `runs/smk` **before** that check, so failure leaves a directory that makes the next staging attempt fail immediately.

    **Fix:** perform all read-only preflight checks before creating the attempt directory, or stage transactionally into a temporary directory. Deliberately resolve the checkout-status policy for audit logs and the collection script before staging.

19. **SMOKE EXECUTION PATH — suitable machinery coverage, but not yet a clean submission gate.**  
    The in-memory generated default batch script passes `bash -n` and requests:
    - job **`t2_smk`**, GPU partition, A100-80G, one GPU;
    - eight CPUs, 128 GB RAM, six hours;
    - absolute cluster venv, f64/highest precision;
    - paralab output/cache/temp paths, manifest verification and GPU-preflight exit 42.

    The driver correctly exercises both settings, dev6 cases 0 and 2, main/HQ/old paths, and **82 A–B–A records per setting**. The GAL-BDF2 tight anchor runs first, so `anchor_full` is populated for subsequent runs. Old uses the mesh/upwind implementation; HQ uses off-mesh point quadrature. Shared-node field retention is approximately **701 MiB per case**, excluding banks, rule blocks and compilation workspace; I found no evident L=256 memory blocker.

    **Fix/gate:** repair staging and audit completeness before submission acceptance. Treat completion/`ALL-DONE` as execution completion only; require the corrected independent smoke audit before either 1024 job. The smoke can test real code paths and integrity gates, but cannot establish full-cohort order, selection, or production timing conclusions.