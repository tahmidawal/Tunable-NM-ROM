**NOT CLEAN for submitting `fom1k`.** The runner fixes several substantive problems, but the audit still permits inconsistent calibration, verification, and timing evidence.

Reviewed HEAD `2d1129d8e`. No files modified, lab-log entry appended, jobs submitted, or GPU experiments run. Validation used syntax checks and in-memory execution of selected audit functions.

| Previous item | Verdict |
|---|---|
| 2 — diagnostic records | **STILL-WRONG** against A1.10; per-step portion fixed |
| 3 — failed calibration reference/fallback | **RESOLVED** in runner |
| 5 — accuracy–timing linkage | **STILL-WRONG** in audit; runner substantially fixed |
| 6 — comparison scope | **RESOLVED** by A9 and expanded config |
| 7 — order enforcement | **STILL-WRONG** as a complete gate; rate enforcement fixed |
| 8 — independent acceptance | **STILL-WRONG** |
| 10 — resources/progress preservation | **STILL-WRONG** on recoverability; memory reservation plausible |

1. **Diagnostic arrays now survive, but per-Newton linear residuals still do not.**  
   [fom2.py:63](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fom2.py:63) reduces linear residuals to a maximum within each step. The persisted `step_rel`, `step_newton`, and `step_lres` satisfy A9’s narrower description, but not A1.10’s per-Newton record requirement. The module description still promises that detail.

   **Fix:** persist per-Newton residuals with counts, or explicitly supersede that requirement and correct the description. Current schedules fit the diagnostic buffer; its clipping behavior is not a `fom1k` blocker.

2. **Calibration fallback and JSON round-trip are correct; independent calibration validation remains inadequate.**  
   [fomrun.py:148](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fomrun.py:148) correctly excludes configurations with failed tight references. Fallback consistently uses `(1e-10, 1e-12)`. I tested candidate, fallback, and unresolved values through the actual `clean` function and JSON: tuples become lists and compare correctly.

   However, [audit_fom.py:84](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:84) checks only **six entries**, not the six distinct required cases. Executing its calibration block accepted:
   
   - Six copies of `dev60`.
   - A negative discrepancy.
   
   It also trusts discrepancies and tight-reference errors without saved calibration fields, and never checks evaluation-row tolerances or timing `chosen` against the calibrated pair.

   **Fix:** require exact case identities, finite nonnegative discrepancies/errors, consistent status and tolerances, and saved evidence sufficient to recompute calibration.

3. **Array keys and ROM dimensions are correct for this config; timing reconstruction is incomplete.**  
   Writer/reader key formatting agrees, including `0.5` and rule identity. With `dev6` first, timing indices agree. Synthetic tests of the actual `Decoder.fields` implementation passed for both \(R'=128\) and \(384\), producing `(6,257,257)` fields with zero boundaries.

   But [audit_fom.py:156](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:156) takes the ROM expected hash directly from JSON. It never binds that hash to the saved coefficients. I executed the timing checker with matching arbitrary ROM hash strings: it accepted them.

   FOM hashes are reconstructed from F64 fields, but those fields are not checked against the F32 fields used for accuracy rescoring. Thus the two evidence streams can disagree.

   **Fix:** save restricted F64 ROM timing fields, hash those, and compare them numerically against independently decoded coefficients. Require FOM F32 fields to equal the F64 fields cast to F32. Do not require NumPy and JAX decodes to have identical bitwise hashes.

   **Latent indexing issue:** putting `val32` before `dev6` breaks the global-versus-local timing index convention and prevents saving timing fields. Current `fom1k` is unaffected; use explicit timing-case indices.

4. **The comparison scope is now defensible, but the promised cross-job check is absent.**  
   A9 clearly limits the claim to listed configurations, defines matching by cohort-worst ST error, and distinguishes 38 accuracy cases from six timing cases. The config contains **80 unique ROM arms**, including factors `0.5` and `10`; omission of wide is explicit.

   However, no code in this audit checks the A9.3 agreement with pinned `a1k*` accuracy artifacts.

   **Fix:** implement that numerical comparison before claiming A9 compliance, or explicitly defer it to a required report gate.

5. **Order normalization and rate indexing are correct; verification inventory is still bypassable.**  
   [audit_fom.py:71](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:71) uses the interior initial-field norm, while the runner uses the padded norm. Boundaries are zero: the norms matched for both order cases. Moreover, that common normalization cancels from the order ratio.

   `log2(d[0.5]/d[0.25])` correctly uses the finest available three trajectories. The registered bands are now enforced.

   But `nfail={}` passes `any(...)`, and the audit does not require all five failure-count keys or reconstruct verification from diagnostics. Coarser differences are calculated but not required to be finite and positive.

   **Fix:** enforce exact order-record and factor inventories, validate every trajectory’s diagnostics, and require finite positive differences.

6. **Verification and nonfinite-value checks remain submission blockers.**  
   [audit_fom.py:105](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:105) and line 142 only put unverified rows into `info`. Neither validates `verified` against saved statistics. Changing a flag or corrupting statistics need not change any acceptance check.

   The FOM rescore test uses `abs(error_difference) > tolerance`. My probe confirmed that **NaN makes this comparison false**, silently avoiding a mismatch.

   **Fix:** require finite arrays and metrics; reconstruct FOM acceptance from active per-step residuals and the calibrated tolerance; check ROM verification consistency. Failed arms may remain diagnostic records, but must be explicitly excluded from eligible comparisons.

7. **Timing values are completely unchecked; rejection tests do not exercise full acceptance.**  
   The extracted timing checker accepted negative/zero durations, NaN `tB`, and arbitrary ratio/drift values when hashes matched. No later check validates those quantities.

   The three rejection checks merely evaluate separate expressions; they never mutate evidence and run the complete validator. Consequently, they cannot expose the calibration, verification, or timing omissions above. The 1% perturbation also cannot reliably fire for sufficiently small errors because the acceptance test has an absolute floor.

   **Fix:** validate finite positive durations, recompute ratios/drift, and run mutated artifacts through the same validation function used for acceptance. Choose perturbations explicitly larger than its tolerance. Add mutations covering duplicate cases, missing diagnostics, wrong tolerances, NaNs, and disconnected hashes.

8. **Memory: acceptable on structural evidence. Wall time: plausible, not established.**  
   Uncompressed saved-array sizes, calculated from the config:

   | Arrays | GiB |
   |---|---:|
   | FOM F32: 16 configurations × 38 cases | 0.898 |
   | FOM F64: 16 configurations × six timing cases | 0.283 |
   | ROM coefficients: 80 arms × 38 cases | 0.035 |
   | Order fields: 40 trajectories | 0.118 |
   | **Total** | **1.334** |

   The persistent 384-column mesh bank adds approximately **2.994 GiB**, before operators, construction temporaries, and compiled workspaces. These arrays do not suggest an 80-GB GPU/128-GB host-memory blocker.

   Assuming all FOM configurations resolve, the expanded workload is:

   - **4,816** FOM calls at \(L=1024\), including calibration, evaluation, reproduction, warm-up, and timing.
   - **4,960** ROM calls.
   - **40** order-check calls at \(L=256\).
   - **1,710** A–B–A triples.
   - At least **570.6 seconds** of configured timing burn-in.

   Six hours remains a reasonable provisional reservation, but the previous 1–3-hour estimate is not validated for this expanded driver. Tight FOM solves, difficult ROM trajectories, compilation, and preparation remain unmeasured here.

   **Fix:** checkpoint array artifacts during evaluation and timing records in smaller batches. Currently arrays survive only after the entire evaluation phase, and timing saves only after each **570-triple** repetition. Periodic JSON alone does not make interrupted evaluation auditable or resumable.

**Gate: hold submission until the calibration-evidence, verification, hash-linkage, order-inventory, and timing-validation holes are fixed. Memory is not the reason to hold it.**