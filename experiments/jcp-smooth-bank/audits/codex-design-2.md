**A1 is a substantial improvement, but it does not yet close the audit.** The remaining blockers are ambiguous decision rules, a reproducibly broken C2 classifier, and solver failures that can still pass C6. No files were modified, including the lab log.

Verdicts below assess whether A1 resolves each previous finding.

1. **Provenance — CORRECT.** Verified r3a’s log/JSON: job 2835788, A100-PCIE-40GB, 10,684.648 s, `snap_norm=false`, and the stated initial losses and fingerprints. Vendored hashes match, including `git show 5ae420414:experiments/separable-decoder/sep_solvers.py`; that trainer updates \(B\) and zeroes only the `out_scale` gradient. The frozen checkpoint gives \(B\)-norm min/median/max **0.079755/4.630113/11.007558**. Rotation code and extraction JSON support the corrected grid and expanded population. **Fix:** none.

2. **Lane evaluation procedure — CORRECT.** A1 distinguishes deployment from lane rotation, gives the coefficient transformation and radius, and correctly removes candidate codes from the linear initialization claim. [`qcore.py:314`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/vendor/quad2d/qcore.py:314) confirms the QR initialization. **Fix:** none to this finding; numerical rank handling remains under item 9.

3. **Sobolev justification — NEEDS-RESTATEMENT.** Calling FD a filtered target and benchmarking cost fixes the original overclaims. However, second-versus-fourth-order disagreement is a **stencil-sensitivity indicator, not measured derivative bias or a certified uncertainty bound**: both stencils can miss the same unresolved component. It also excludes near-wall points included in the gradient metric. **Fix:** use a common support, recompute treatment rankings with both stencils, and call close results stencil-sensitive; require finer-reference derivatives before claiming resolved physical-gradient improvement.

4. **Metrics — NEEDS-RESTATEMENT.** Discrete naming, common-reference projections, tail-rate conversion, rollout terminology and timing retention are improved. Remaining problems:
   - Fibonacci censoring must say **\(>46368\)**, not \(>65536\).
   - \(n_{10^{-12}}\) can be undefined; zero coefficients make logarithms undefined.
   - The spectrum classifier fails C2 below.
   - Common-reference projections still produce different physical fields; agreement supports the mechanism but does not isolate it.
   
   **Fix:** define unresolved-spectrum outcomes and censoring per ladder; retain projection-error diagnostics alongside both populations.

5. **Decision equations — NEEDS-RESTATEMENT.** They are executable for positive, finite H1 metrics only after specifying the FD comparison. They are **not fully defined for all planned outcomes**:
   - Censoring supplies an ordering, not a reduction factor. `base >65536`, treatment `65536` does **not establish 1.5× improvement**.
   - “Same direction” needs an explicit per-setting inequality, including ties.
   - Censored rung reductions have no specified rank; “lower \(E_S\)” leaves two settings without an aggregation rule.
   - Zero denominators, invalid ladders, nonfinite errors and failed-case exclusions have no decision semantics.
   - The generic resolved-effect rule and H2’s descriptive pass need an explicit statement of whether the noise gate applies to H2.
   
   **Fix:** specify these branches, use conservative ratio bounds for censored comparisons, and make failed/invalid required measurements ineligible for promotion. Compare FD sensitivity with error changes using consistent normalization. The noise formula is an operational sensitivity threshold, not a variance estimate; A1’s observed-run limitation appropriately narrows the claim.

6. **Controls — WRONG as a complete repair.** Several are useful, but the following remain:

   | Control | Re-audit |
   |---|---|
   | R0/R1/R2a | Can fail on actual mismatches; historical targets are supported. R0’s two moments are a fingerprint, not unique dataset identity. Scope R0 explicitly to the original-resolution dataset. |
   | R2b | The current [test](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/tests/test_r2b.py:1) **never runs the \(\lambda=0.1\) path**. It constructs one index stream and prints it; it cannot detect treatment-dependent sampling. Instrument and compare both actual paths, plus same-input losses/gradients. |
   | C1 | Correctly downgraded to a stress arm. Add a manufactured inaccurate integral if quadrature-failure detection itself must be demonstrated. |
   | C2 | **Fails on correct smooth data.** A read-only NumPy/SciPy evaluation of the stated Gaussian convention and DCT/window gives \(n_{10^{-8}}=27,87\), correctly ordered, but **both classify “inconclusive” at both 256 and 512 points**. Center symmetry puts odd-degree shells near roundoff; logging every shell destroys the fit. Define the Gaussian precisely and fit a registered block envelope or supported subsequence above a noise floor. |
   | C3 | Accepting “inconclusive” lets an always-inconclusive classifier pass. For the stated kink, my finite transforms never reach \(10^{-12}\), so the registered fit endpoint is undefined. With an explicit unresolved-tail fallback, it classified inconclusive. Require a resolved algebraic manufactured control as well. |
   | C4 | Restored checks can fail, but must cover **both** own-rollout and common-state populations. Passing on one does not validate targets on the other. |
   | C5 | Fixed-step FD can disagree with a correct analytic derivative for a sufficiently rough bank. Use step refinement and an aggregate norm. Part (b)’s displayed formula is only \(D_xu\); specify and test \(D_yu\) too. |
   | C6 | **Still misses finite stalled solves.** [`hfast.py:96`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/vendor/quad2d/vendor/hfast.py:96) permits tiny-step and damping-limit exits without stationarity. Fifty such exits could pass the finite/budget rule. Require accepted exit/stationarity criteria and tighter-solver sensitivity checks. |

   Thus **not every control currently detects its intended failure**, and C2 can reject correct behavior. Replace the inherited “every control must fire” wording with separate positive-control, negative-control and acceptance-gate contracts.

7. **Cohorts/references — NEEDS-RESTATEMENT.** Disjointness against both training populations and discrete-reference provisional labels address most concerns. **Fix:** explicitly propagate reference provisional status to H1/H2/useful-winner conclusions. Seed-repeat qualification alone does not address reference uncertainty. These remain development/validation selections.

8. **GPU accounting — NEEDS-RESTATEMENT.** A1’s arithmetic totals **82 allocated GPU-hours** correctly. But [`jobs/tr1.json`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/jobs/tr1.json:1) requests **four GPUs for eight hours**, permitting 32 GPU-hours versus A1’s 20. Evaluation also needs an explicit schedule for five banks on at most four GPUs. **Fix:** reconcile resource limits and scheduling. I did not independently verify live cluster topology.

9. **Remaining safeguards/coarse control — NEEDS-RESTATEMENT.** **129 nodes is geometrically sound:** its nodes nest in 257, eliminating interpolation. **The comparison is still mismatched:** the coarse FOM is scored on 129 nodes while the ROM’s registered error uses 257. **Fix:** score ROM, base and both FOM comparators on the common 129-node restriction for the coarse-control verdict; report additional 257-node comparisons separately. Beating the coarse FOM supports the limited coarse-training claim; failing to beat it does not prove finer training data are necessary.

   Conditioning is recorded but no numerical-rank cutoff, singular-bank failure policy or tested-operator conditioning check is specified. A1.8 also needs explicit state identifiers, checkpoint/reference hashes and saved derivative/spectrum diagnostics—or exact reconstruction specifications—to justify “every reported number” being independently recomputable.

**Before interpreting round 1**, fix C2/R2b/C6, define failed and censored decision outcomes, validate both ladder populations, and register rank handling. Otherwise `base`, `sob01`, `sig2`, and `sig1` can produce useful descriptive measurements, but an apparent winner may still reflect solver failure, unresolved derivatives or undefined selection logic.