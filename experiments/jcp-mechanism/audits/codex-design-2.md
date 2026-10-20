1. **A1-1 — WRONG.** Not a well-defined partition: median separation permits individual \(s_j=0\), making \(f_j\) undefined; reason-3 failures can satisfy both **N** and **X**, or **X0** and **X**. The 1%, 90%, and 75% margins are conventions without an uncertainty justification. The causal restraint is improved, but “reaches the continuum rollout” overstates approximate recovery. **Fix:** order decisions explicitly: invalid/failed → X; insufficient separation → X0; otherwise R/N/X. Specify handling of each insufficiently separated case, tie conventions, and uncertainty-backed margins. Say “recovers the reference rollout within the declared margins.”

2. **A1-2 — CORRECT mathematically; memory feasibility still conditional.** `test_block` gives
   \[
   P=L^{3/2}L^{-3}2^{3/2}\prod\sin=(2/L)^{3/2}\prod\sin=\Phi.
   \]
   In `dst_axis`, the odd-extension FFT supplies twice the sine sum; division by \(\sqrt{2L}\) yields \(\sqrt{2/L}\) per axis. `phiT` selects the specified one-based modes. **Fix:** explicitly require matching coordinates, flattening and mode order; transpose its row-batched output to obtain \(J_u\). Keep a largest-shape memory gate: 2D construction concatenates retained chunks, so final-array size alone does not establish peak fit.

3. **A1-3 — WRONG as written, although the leading coefficients are correct.** The bracket is at least \(0.25\) in 3D and \(0.5\) in 2D, hence the state is strictly positive inside. Backward has the stated negative sign; central has the stated positive sign. However, the remainder must also carry \(L^{d/2}\):
   \[
   L^{-d/2}(N_h-N)=-\tfrac h2\!\int\psi u\Delta u+O(h^2),
   \qquad
   L^{-d/2}(N_h-N)=\tfrac{h^2}{6}\!\int\psi u\sum_j u_{x_jx_jx_j}+O(h^4).
   \]
   **Fix:** use these normalized expansions; establish nonzero coefficient norms and finite-window expectations independently. Both controls can fail: C-pos through implementation or unresolved asymptotics; C-neg(a) through normalization/fitting bugs. Its zero-slope gate is valid for **relative** gaps, with nonzero \(N,e\); it does not validate stencil assembly.

4. **A1-4 — WRONG as a resolution guarantee.** Agreement between two Gauss rules is not an error bound; both can share error or agree to rounding. There is no roundoff floor. Filtering states separately at each mesh can also change the population underlying the median. **Fix:** call this an empirical screening rule, add independent target validation and a numerical floor, and freeze a common state population and explicit mesh-selection rule. The normalized fourth-order nodes prediction is sound.

5. **A1-5 — NEEDS-RESTATEMENT.** Separating historical reproducibility from within-job pairing is correct. But scalar ST errors cannot reconstruct per-case field distances; G1–G3 still lack the previously requested finite-difference steps and near-zero absolute tolerances. **Fix:** use saved fields/coefficients for field comparisons, report scalar errors separately, and specify those gate details.

6. **A1-6 — NEEDS-RESTATEMENT.** `adaptive_first=25` does cover every 3D step, but retains the original stopping tolerance. Reporting nonstationarity and sensitivity without an acceptance threshold leaves solver error able to drive R/N. The 2D “population arm” still need not be `nodes`. **Fix:** prescribe sensitivity/stationarity margins that force X when exceeded, qualify untested configurations, and explicitly check targets on 2D nodes-reached states.

7. **A1-7 — CORRECT.** Backward versus sign-upwind, unrenormalized weights, inspected cohorts, and preconfigured subsets are appropriately distinguished. **Fix:** clarify that an actual timeout produces an incomplete result—not retrospective permission to select the fallback—and list every missing case.

No files modified or jobs run.

**Remaining WRONG items: A1-1, A1-3, A1-4.**