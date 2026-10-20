**Verdict: the design needs amendment before execution.** The main blockers are incorrect rotation provenance, an initialization-scale sweep described as bandwidth control, confounded noise estimates, and broken C3/C5 controls. No files were modified.

References below use `D` for [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/DESIGN.md), `Q` for `experiments/jcp-smooth-bank/vendor/quad2d/`, and `SD` for `experiments/separable-decoder/`.

1. **Provenance — WRONG in specific claims; most training details are CORRECT.**

   - **CORRECT:** K=16, R=512, n_ff=128, widths, optimizer, global value loss, orthogonality weight, schedule, EMA, 300,000 steps, point subsampling, and state pick match `SD/runs/push_r3a/out/sep_burgers_r3_N256_K16_R512.json:2,5547`. The run used all 64,516 interior points and 16,384 states, including 3,456 early states. The generator uses **256 nodes, h=1/255**, backward Euler and sign-upwind advection.
   - **CORRECT:** hfit preserves the bank. `SD/sep_hfit_run.py:386` copies parameters and replaces only head parameters. Its expanded data include **4,032 additional trajectories from seed 1000**, beyond the original 576 (`dn256b/out/sep_coeff_N256_K16_R512.json:10`). This matters downstream.
   - **WRONG:** D:59 calls the deployed rotation mesh the training mesh. Branch `exp/2026-09-25-burgers2d-test`, `experiments/burgers-bank-knob/make_rotation.py:65`, calls `on_grid(256)` with **256 intervals**, producing a **255²-interior** grid; see `Q/vendor/arms.py:51`. Training instead used 254² interior points.
   - **NEEDS-RESTATEMENT:** σ=4 is the **initialization default**, not a fixed final frequency scale. `SD/sep_common.py:100,118` puts B into the differentiable feature map; `SD/sep_solvers.py:458` updates all parameters and suppresses only `out_scale` gradients.
   - **WRONG:** the “same recipe” timing sibling in D:61 used `snap_norm=true`; the source bank used false. The source job recorded **10,684.65 s on A100-40GB**, versus the sibling’s **9,737.80 s on A100-80GB**.

   **Fix:** separate training and rotation grids, document the expanded hfit population, call σ an initialization scale, and cite the actual source-bank timing.

2. **“Only the training recipe differs” — NEEDS-RESTATEMENT.**

   D:70–78 changes **coefficient source, population size/composition, rotation grid, truncated subspace, and trust radius** relative to deployment. The head was trained to approximate LS coefficients, but its outputs are not identical to them. Equal row-normalization formulas do not make the resulting rotations equivalent.

   The cold-start claim is specifically **WRONG for the linear settings**: `Q/qcore.py:314` uses only the Gauss-point QR factors; it discards candidate-code fields in the cold tuple. Replacing candidates does not change this initialization. Replacing coefficients **does** change the trust radius (`Q/qcore.py:152`).

   **Fix:** describe a common *lane evaluation procedure*, distinct from deployment. Compare frozen/deployed against frozen/lane to measure that procedure change. State the exact rotation grid, radius formula, coefficient transformation and closed-form Gauss-48 initialization. Preserve the original point-sampling RNG stream when adding gradient sampling.

3. **Sobolev loss — CORRECT definition; WRONG justification as written.**

   D:88–96 defines a reasonable normalized gradient-matching loss. Uniform state/point sampling gives an unbiased estimate with the fixed denominator.

   D:98–105 overclaims:

   - Initial bump width does not bound derivatives after nonlinear steepening.
   - The estimate \(h^2u'''/6\) requires smoothness and derivative bounds; \((h/w)^2/6\) is not a demonstrated relative-error bound for these discrete trajectories.
   - Algebraic sine decay does not imply a spectral derivative necessarily produces harmful wall ringing.
   - Central differences suppress near-Nyquist components, including a checkerboard null mode; they are not an unconditional roughness detector.
   - Subsampling states reduces the coefficient-contraction work, but not the shared bank-gradient computation by the same factor. “About 1/8 of its cost” is unsupported.

   **Fix:** justify FD as a chosen filtered derivative target. Measure its bias on representative fields using resolution/stencil comparisons; qualify all gradient improvements by that uncertainty. Benchmark Sobolev training cost.

4. **Metrics and mechanism identification — NEEDS-RESTATEMENT.**

   | Metric | Verdict, reason and fix |
   |---|---|
   | Projection floor, D:147 | **CORRECT as a sampled discrete projection error.** It is not yet a continuum \(L^2\) floor: unresolved bank oscillations can hide between 257² nodes. Name the discrete metric and add an independent finer/off-grid check. Label S-based projection comparisons provisional too. |
   | Gradient error, D:151 | **NEEDS-RESTATEMENT.** Measures analytic decoder gradients against an FD surrogate after value-only projection. It conflates representation error and derivative-reference bias. Report the FD uncertainty and an additional resolved derivative check. |
   | Own-bank gref populations, D:153 | **CORRECT operationally; insufficient causally.** A less accurate bank can generate easier trajectories and reduce quadrature needs. Add matched case/time projections of common reference fields into each bank, alongside own-rollout results. |
   | Minimum points, D:160 | **NEEDS-RESTATEMENT.** No rule for unbracketed thresholds or isolated crossings in nonmonotone ladders. Define right-censoring and require confirmation at subsequent larger rungs. Interpolation remains descriptive. |
   | Tail rate, D:163 | **WRONG parameter definition.** The slope against \(2p\) is \(-\log\hat\varrho\), not \(\hat\varrho\). The chosen range can include onset and unresolved plateaus. Define \(\hat\varrho=\exp(-s)\), minimum fit length, fit diagnostics and “no resolved tail.” |
   | Chebyshev bandwidth, D:165 | **NEEDS-RESTATEMENT.** A single 256² transform cannot exclude aliasing or establish the infinite tail; “decaying range” is discretionary. Specify DCT normalization, fit window, aggregation, unresolved outcomes and a 512² convergence check. Column spectra also depend on rotation; distinguish them from reconstructed-field spectra. |
   | End-to-end error, D:173 | **CORRECT as a measured rollout error.** “Floor” is too strong without solver and target-rollout convergence checks. Call it the converged-quadrature rollout error, subject to those checks. |
   | Timing, D:178 | **CORRECT pairing principle; incomplete execution contract.** Persist every repetition, paired output errors, exits and iterations. Validate a burn duration sufficient for this device. Run the actual selected \(m^*\), which need not appear in the fixed end-to-end rule list. |
   | Training cost, D:182 | **CORRECT**, provided generation, compilation, fitting and rotation costs are distinguished. |

   **Additional WRONG claim:** D:33–45 turns a frequency heuristic into an immutable onset and an “at best” improvement bound. Common tests do not imply identical onset: bank-dependent amplitudes, products and cancellation matter. The cited audit explicitly rejects a universal onset formula (`06-codex-audit.md:42`). **Fix:** retain this as an expectation, not a restriction on possible outcomes.

5. **Pass/fail and noise rules — NEEDS-RESTATEMENT.**

   D:186–199 has several unresolved choices:

   - Base–frozen difference is not automatically training noise, especially if “frozen” means deployed rotation. Name the lane-rotation comparator.
   - A same-seed rerun and one additional base seed do not estimate treatment variability. An almost-zero baseline difference makes the noise gate almost vacuous.
   - “Added as the seed-variance yardstick” lacks an equation: sum, maximum, or separate requirement?
   - H2 can pass by smoothing away important solution structure. Its separate winner gate limits this, but **H2 pass alone is not evidence of useful improvement**.
   - Specify which threshold must satisfy “no increase in the other,” how competing reductions rank, ties, and censored \(m^*\).
   - D:124 and D:197 give different combination gates. Moreover, “best λ” among round-2 arms cannot determine a combination trained concurrently with them.
   - `base_s1` must change training randomness while preserving the fixed data/state pick; the original driver uses `SEED0` for that pick too (`SD/sep_burgers_r3.py:322`).

   **Fix:** write executable decision equations, separate descriptive H2 success from useful-winner status, and schedule combination training after the required selections. Repeat promoted treatment–base pairs with matched additional seeds, or explicitly limit conclusions to the observed runs.

6. **Controls — mixed; C3 and C5 are broken.**

   | Control | Verdict and fix |
   |---|---|
   | R1, D:210 | **CORRECT numerical targets.** The prior report’s development/validation summary gives 0.019 and 0.023 at 1024²; its population is lat64, matching R1. Keep this separate from the new gref ladders. Pin the exact source artifacts and the ST-error targets rather than leaving the latter implicit. |
   | R2, D:211 | **NEEDS-RESTATEMENT.** Equality of formulas does not establish bit-identical execution. The original log exists at `SD/runs/push_r3a/logs/2835788.out`. Compare loss and gradients on identical inputs, plus initial parameters and sampled-point indices; then check logged steps. |
   | C1, D:212 | **NEEDS-RESTATEMENT.** Historical Gauss-8 failure does not guarantee failure for every retrained bank. Treat it as a stress arm; use a known inaccurate manufactured-integrand case to verify detection. |
   | C2, D:213 | **CORRECT basic sanity check**, but specify bump center, thresholds and expected ordering. It does not validate tail-rate classification. |
   | C3, D:214 | **WRONG.** An algebraic envelope \(j^{-2}\) has local exponential slope \(2/j\), exceeding \(\log(1.02)\) below roughly degree 101. A finite-window fit can therefore classify the kink as geometric. Fix with registered windows, resolution extension and algebraic-versus-geometric model comparison; allow “inconclusive.” |
   | C4, D:215 | **NEEDS-RESTATEMENT.** This is weaker than vendored G6: `Q/qstudy.py:396` also checks point-versus-flux agreement, and `Q/DESIGN.md:214` requires a 768² rollout comparison. Restore those checks. Agreement of two quadrature orders alone is not a certificate. |
   | C5, D:216 | **WRONG.** Span membership guarantees value reconstruction, not ≤1% FD gradient bias. A rough bank may legitimately fail “clean”; unspecified grid noise can lie near the central-difference nullspace, and a fixed amplitude cannot guarantee 10× amplification. Separate exact derivative implementation parity from FD bias; inject a specified resolved mode with a calculated FD response. |
   | C6, D:217 | **NEEDS-RESTATEMENT.** Finite stalled solves can pass with large residuals, and an aggregate 5% budget allowance can conceal a failed case. Specify per-case exit/residual criteria and a tighter-solver sensitivity check. |

7. **Cohorts and references — CORRECT policy, with qualification needed.**

   D:133–141 correctly restricts selection to dev6∪val32 and excludes test64. `Q/qstudy.py:155` checks reference completeness, acceptance, configuration, cohort hashes and field hashes.

   **Fix:** explicitly verify training/evaluation disjointness against both the original seed-0 draw and the seed-1000 expansion supporting the deployed rotation. Report all selected outcomes as **development/validation findings**, not independent confirmation. Apply provisional labels to S-based floors and verdicts as well as end-to-end numbers; a 5% improvement may be smaller than reference uncertainty.

8. **Single-job cap and GPU budget — CORRECT literal cap interpretation; WRONG allocation accounting.**

   The cited lane rules cap **jobs**, not GPUs (`lane-agent-rules.md:35`), so a multi-GPU allocation does not literally violate that cap. It still requires explicit GPU capacity and process isolation.

   D:237–240 gives active-work estimates as though they were allocation cost. Four GPUs held four hours consume **16 allocated GPU-hours**; five held three hours consume **15**, including idle devices. At the listed maximum allocations, the four jobs total approximately **66 allocated GPU-hours**, before overruns.

   **Fix:** distinguish active and allocated GPU-hours, verify the requested five-A100 single-node topology, and profile derivative training, LS extraction and large-rule evaluation before treating these estimates as credible.

9. **Missing safeguards making the outcome hard to interpret — WRONG omissions.**

   - **Coarse comparator mismatch:** D:125 trains on 128 **nodes**, but D:201 invokes an engine at L=128 **intervals**. Use the actual training generator at 128/256 nodes and specify interpolation to reference scoring points; these grids are not nested in 257².
   - **H2 mechanism measurement:** because B trains, record initial/final B distributions and measured field/integrand bandwidth. Smaller initial σ need not produce a smoother final bank.
   - **Solver conditioning:** record rotation/LS conditioning, singular-value policy, trust radii and test-space coupling. Otherwise apparent bank improvements may come from changed solver conditioning or clipping.
   - **Artifact sufficiency:** require saved coefficients, state identifiers, per-state quadrature errors, restricted fields, solver diagnostics and raw timings sufficient for independent recomputation.

   These fixes would make the study interpretable as a controlled comparison of training recipes and their resulting linear ROMs. As written, several outcomes could instead reflect rotation, sampling, derivative-surrogate or solver changes.