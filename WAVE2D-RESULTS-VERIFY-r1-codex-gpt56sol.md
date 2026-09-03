The Reading is not correct as written. The central empirical result—reflective W3 fails while an energy-conserving arm also fails—is supported, but several gate verdicts, ranges, and the formal decision-table interpretation are overstated.

1. **WRONG.** The tangent-residual ranges are accurate, but “all three heads pass G0” is false: reflective `sup`, N=128 is the sole failure, so the count is 11/12. Its raw metric is favorable, 0.74116 versus POD-K’s 0.82529, but G0b fails because the random-tangent control is 1.1787, below its 1.2 threshold. See the [phase-2 table](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/tables/wav2d-phase2-gates.md:3).

   The quoted raw ranges are supported:

   - Reflective: tangent 0.62695–0.74147; POD-K 0.68520–0.82529.
   - Absorbing: tangent 0.23188–0.30094; POD-K 0.67847–0.78936.

2. **WRONG as a bundle.** “Reflective W3 fails for every head, arm, and N” is correct, including incomplete rollouts counting as failures. The quantitative explanation is not:

   - At each arm’s W3-selected RS, arm A is 2.236–5.007× floor, but `Edyn_ratio_T_median` is 6.452–19.086—not 4.5–15×. The 4.5 lower endpoint comes from coarser RS8, while 19.086 is omitted.
   - The quoted `sup` sequence 4.5→6.5→8.1 has no certified third median: RS40 is 13/16 complete and its JSON aggregate is null.
   - For fully completed arm-C cells, `err_T_median/floor_T` is 2.233–5.716×, not uniformly 4–6×. At N=128 `sup`, arm C is also better than arm A: 2.233× versus 2.590×.
   - At the selected RS, `Er_ratio_4T_median` spans 0.999582–1.001160. Across all fully complete arm-C rows it spans 0.996067–1.002430—not 1.0000–1.0007.
   - “No secular trend on every completed head” is directly contradicted by N=128 `sup`: W4 reports 116% maximum deviation and slope 0.00971/T. N=64 `auto` also narrowly fails W4.

   The prose appears to have used the generated table’s **T-energy** column while calling it conservation “over 4T.” See [phase-3 rollouts](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/tables/wav2d-phase3-gates.md:21).

3. **CORRECT for the cell’s own numbers.** Reflective N=64 `auto` has held-out oracle median 0.18615, which rounds to 0.19 alongside the historical 0.189. Arm-A `err_T_median` is 0.88750 at RS8, hence 0.888. It should say RS8, because the W3-selected finest complete RS is RS20, where the result is 0.93210. The external 0.189 and 0.878 comparators were not independently checked.

4. **CORRECT, with a horizon qualifier.** At absorbing N=64:

   - `sup` W3 passes for A at RS8 and C at RS40.
   - Arm C has 1.2938× floor, 0.2470× same-dt POD-K CN, and 16/16 completion at RS8, 20, and 40.
   - The selected **T-error** ratios for `auto`/`auto+vc`, both arms, span 2.388–3.293× floor.

   Add “at T”: their 4T floor ratios extend to 4.845×. See [N=64 absorbing gates](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/tables/wav2d-phase3-gates.md:146).

5. **WRONG as the formal decision-table assignment.** The design row requires, after D2 and W6 pass, “G0 pass + reflective W3 fail + W4 pass, **all heads**.” The observed pattern holds for `auto+vc` individually at both N, but not for all heads:

   - N=64 `auto` has W6-C PASS but W4 FAIL at 1.03408—3.41% over the threshold, not a passing “3% margin.” The design consequently calls this an implementation bug to resolve.
   - `sup` has incomplete/W4-failing behavior, and N=128 `sup` also fails G0.
   - Therefore the formal all-head row has not obtained. See the [predeclared decision table](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/WAVE2D-DESIGN.md:295).

   The exclusions are only partly stated correctly:

   - W6-C passes for reflective `auto` and `auto+vc` at both N, but its values are 1.65e-5/0.0425 for `auto` and 0.00349/0.00346 for `auto+vc`; “RS-independent to 1e-5” is not general.
   - D2 passes, but “0.07–0.1 along every rollout” confuses phase-2 training minima with rollout conditioning. Completed reflective `auto`/`auto+vc` rollouts have per-trajectory minima roughly 0.059–0.141, with larger values elsewhere.
   - G0c is 1.447 and 1.490 for `auto+vc`; for `auto` it is 1.808 and 2.374. Thus 1.4–1.8 is not valid across both heads and N.
   - The design also requires the curvature alternative to be addressed; the Reading does not exclude it.

6. **NEEDS-RESTATEMENT.** The data support the narrower conclusion that energy loss is **not necessary** for the accuracy failure: `auto+vc` passes W4 and fails W3 at both N. They do not support universal “arm C keeps it to 1e-4” language. For `auto+vc`, maximum W4 energy deviations are 5.45e-3 and 6.07e-3; its 4T median endpoint deviations are about 4.18e-4 and 3.47e-4. Other completed heads include a W4 failure as large as 116%.

7. **CORRECT.** W3 explicitly requires a reflective pass for at least one head/arm pair or the ladder is not run, and phase 4 is declared “only if W3 passes.” See [W3 and phase 4](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/WAVE2D-DESIGN.md:257).

8. **NEEDS-RESTATEMENT.** Numbers or summaries not properly carried by the current generated artifacts include:

   - Historical comparators 0.189 and 0.878, plus the prior-run 18% stride discrepancy, are hand-maintained history rather than current JSON results.
   - The certified JSON aggregate does not contain the claimed RS40 `sup` energy median 8.1 because that rollout is incomplete.
   - W2 “1±1e-12” understates its max-over-16 deviations, which are 1.65e-11–4.99e-11.
   - The ≥3× bank-ceiling claim is false; the smallest reflective oracle/ceiling ratio is about 2.62×.
   - “Every control fired” is false as stored: N=128 reflective `sup` G0b’s control fails. Absorbing W6 controls are also stored against 0.2 and marked FAIL, despite Retraction 8 declaring a new 0.1 control threshold.
   - “Nothing measurable,” “flat in N,” and the causal tangent-space explanation lack repeats, uncertainty, or a completed curvature exclusion.
   - The N=128 `sup` interpretation reverses the metric direction: 0.741 is **better**, not worse, than 0.825; G0b fails because of its control.

| item | verdict | one-line fix |
|---|---|---|
| 1 | WRONG | Say 11/12 G0 cells pass; N=128 reflective `sup` fails its G0b control, while retaining the correct raw ranges. |
| 2 | WRONG | Keep the universal reflective W3 failure, but replace the energy/error ranges and report 4T energy and W4 failures explicitly. |
| 3 | CORRECT | Label 0.18615 as N=64 reflective and 0.88750 as RS8; do not imply 0.888 is the W3-selected RS20 value. |
| 4 | CORRECT | Add “at T”; the stated N=64 `sup` pass and ratios are supported. |
| 5 | WRONG | Say `auto+vc` individually matches the pattern; the formal all-head design row is unmet and curvature remains unexcluded. |
| 6 | NEEDS-RESTATEMENT | Say energy loss is not necessary for failure; replace universal 1e-4 conservation with the actual W4 bounds. |
| 7 | CORRECT | No fix: phase 4 is correctly withheld for lack of a reflective W3 pass. |
| 8 | NEEDS-RESTATEMENT | Label historical/derived numbers, remove unsupported aggregates, and downgrade causal statements to hypotheses. |

**Single most important overstatement:** the Reading converts a formally **inconclusive** design outcome into a causal claim that tangent-space error is the reason for failure, even though the all-head decision row is unmet, curvature was not excluded, and its strongest supporting comparison—0.741 versus 0.825—is interpreted in the wrong direction.