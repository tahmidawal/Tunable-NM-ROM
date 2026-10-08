**Do not submit `ev1` yet: P5 deterministically crashes, and the report can issue false H2 passes.** Read-only review completed; `bankeval.py` was not run. Checks used isolated functions and CPU-only vendor imports.

Paths below are relative to `experiments/jcp-smooth-bank/`.

1. **CORRECT — imports and staging.** `bankeval.py:34–45` resolves the repository’s `experiments/separable-decoder/sep_common.py` and `experiments/mr-burgers2d/engines.py`, plus the intended vendored modules. `qcore.py:35–38` inserts nonexistent nested experiment paths, but the explicit outer paths compensate. `cluster/stage.py:29–56,71–74` stages the dependencies/data and runs timing on device 0 after successful workers. All seven vendor Python files are byte-identical to the original lane. **Fix:** no functional correction needed; assert and record resolved module paths to prevent future shadowing.

2. **CORRECT — trust radius and cold codes.** `bankeval.py:70–85,260–261` substitutes LS training coefficients exactly where specified; deployed evaluation retains `Q.Model.trust_linear`. Coordinates are transformed by `Lrot[:Rp].T`. Candidate codes do not affect the linear initializer: `qcore.py:314–320` uses only the weighted Gauss QR fit. **Fix:** none required.

3. **CORRECT — reference rollout metrics.** `bankeval.py:301–313` matches `qstudy.py:321–339`: restriction to 257² nodes, normalization by restricted initial-field norm, and maximum excluding time zero. The pinned R1 targets are:

   | Setting | Gauss64 worst ρ, lat64 population | gref worst ST | lat64 worst ST |
   |---|---:|---:|---:|
   | acc | 0.01855400027 | 0.02746551840 | 0.03383906214 |
   | fast | 0.02329974040 | 0.04636993877 | 0.05032507568 |

   **Fix:** use these source-JSON values in an executable R1 gate; that gate is currently absent.

4. **WRONG — P2 is incomplete, although its coordinates/indexing are correct.** `bankeval.py:340–364` fits the 255² interior in the same rotated coordinates as rollout `internal`; common-state coefficients are valid rho inputs. D4 indexing passes a cubic-polynomial check exactly. However, D2 error is reported only on the full interior; A2.2/A4 require D2 **also on D4’s common support**. D4 per-state errors are discarded (`366–367`). **Fix:** compute/save common-support D2 and per-state D4 errors, and feed the prescribed common-support metrics into H1. The secondary training-mesh projection comparison is also missing.

5. **WRONG — P3 arithmetic is sound, but invalid denominators can pass.** `bankeval.py:384–435` correctly streams unrenormalized point sums, computes per-state rho, applies monotone confirmation/censoring, and fits the amended tail. `vendor/hops.py:35–41` stably sorts discrete eigenvalues; the first 320 are a consistent prefix. But `bankeval.py:396` replaces zero target norms with `1e-300`: a zero target and zero approximation produce rho zero and potentially a passing smallest rung, contrary to eligibility requirements. **Fix:** persist target norms, reject zero/nonfinite denominators explicitly, and save the specified descriptive interpolation separately.

6. **WRONG — fatal spectra key mismatch.** `bankeval.py:187` creates **`n_0.0001`**, whereas `468` requests **`n_1e-04`**. Isolated execution reproduces `KeyError('n_1e-04')`; every worker stops during its first `acc` setting. Points, DCT scaling, \(f=u(u_x+u_y)\), envelopes, resolution comparison, and amended classifier otherwise check out; C2/C3 pass. Additionally, `473` omits unresolved classifications, and `475` discards individual envelopes and resolution diagnostics. **Fix:** standardize epsilon keys, count unresolved classifications, and persist `spec_states`; label bandwidth aggregates as conditional on resolved states.

7. **WRONG — control implementation is not fully faithful.** Exit mapping/counting is correct (`bankeval.py:306–308`; `qstudy.py:42`), and tight gref runs on dev6 (`282,335–336`). However:
   - `finite` checks only restricted output fields (`305`), unlike qstudy’s full fields.
   - C4 uses restricted rollout distance (`317–319`), whereas qstudy’s primary distance uses full fields/full initial norm.
   - Target convergence uses `rho(Tk,Tc)` (`401`), while qstudy uses `rho(Tc,Tk)` (`404`).
   - C5a selects the minimum **error** across steps (`251`), rather than explicitly checking the smallest step.

   **Fix:** check full fields/internal states, retain both full and restricted distances with explicit names, match qstudy’s target-check denominator, and gate C5a on the `1e-05` result.

8. **NEEDS-RESTATEMENT — memory looks plausible; six hours is unverified.** `bankeval.py:328` clears rule blocks per arm. At `acc`, Gauss640/768 contain **409,600/589,824 points**, with persistent point-rule arrays of approximately **7.03/10.13 GiB**, plus a **2.99 GiB** mesh bank. Construction temporarily duplicates arrays (`qcore.py:225`); spectra retain unrotated derivative chunks (`144–145`). No obvious unavoidable 80-GB overflow emerges, but these are not peak-memory measurements. `jobs/ev1.json`’s six hours includes two sequential banks per worker and subsequent timing. **Fix:** after the crash fix, benchmark representative acc rollout/P2/P3/P5 phases and peak memory; extrapolate including compilation, tight solves, and timing before accepting the limit.

9. **WRONG — timing protocol works, but its evidence is incomplete.** `banktime.py:80–97` correctly warms every subject/case, runs forward/reverse order, burns before calls, and compares against evaluation hashes. Staging confines it to one GPU. But it saves only hash-match booleans, omits exits, and marks timing complete even after mismatches (`105`). **Fix:** save actual/expected hashes and exit counts per invocation, validate complete evaluation inputs, and invalidate timing when any comparison fails. The report must suppress invalid timings.

10. **WRONG — decision/report gates can produce false passes.** `make_report.py` has several substantive gaps:
    - **Noise:** `135–140` ignores `base_s1`; missing noise defaults to zero.
    - **H2:** `162–171` checks common-state direction and frozen/base equality only in the improving setting. An isolated fixture passes despite worsening common-state `fast` results and disagreeing noise controls.
    - **Eligibility:** `99–118` allows missing rotation diagnostics, does not enforce complete populations/artifacts, and cannot inspect discarded denominators. Frozen-lane rotation diagnostics are available under training `rotation_frozen`, but unused.
    - **Acceptance:** no R1 comparison exists; C5b failure and incomplete inputs do not block verdicts.
    - **Selection:** censoring reduction and A4 rank-score arithmetic are correct, as are useful-winner error caps; the mean-\(E_S\) tie-break and actual ranked selection are missing.
    - **Reporting:** numerical table cells generally derive from JSON, but unresolved classifications falsely print zero, and required errors with/without C6-failed cases are absent.

    **Fix:** validate inputs and acceptance gates first; implement noise from both comparators, require H2 conditions across both settings, consume common-support gradients, and generate explicit ranked eligibility-aware decisions. Add regression fixtures for the demonstrated false pass and missing-data cases.

No repository files—including the lab log—were modified.