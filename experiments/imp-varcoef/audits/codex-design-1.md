Read-only audit completed. No files changed, jobs submitted, or arrays copied. **The design is not ready for production.** The reported ROM costs reproduce correctly, but reference validation, qualification, timing identity, and selection rules need fixes.

References below to “DESIGN” mean [experiments/imp-varcoef/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/DESIGN.md).

1. **CORRECT — The lane’s gate is quoted verbatim.**  
   DESIGN §9 preserves the binding lane-brief wording, including stopping without training, auditing a Phase 2 amendment, and staying within the remaining cap. Contrast 100 agrees with source-report §7.3.  
   **Fix:** None to the quotation.

2. **NEEDS-RESTATEMENT — The operational gate adds choices not specified by the quotation.**  
   “Pass at either 3% or 10%,” using a point estimate even when its interval crosses 2, is an explicit and reasonable exploratory interpretation—not a verbatim consequence. Moreover, “the where-we-win kill criterion is the same rule” omits that source’s Phase 2 requirements: equal median **and** worst error, the competitor union, and the bank-floor limit.  
   **Fix:** Identify the target aggregation and point-estimate rule as lane-specific operational choices. Carry the source’s Phase 2 criteria into the required amendment. Missing references or missing admissible competitors must yield **inconclusive**, never an infinite-cost pass.

3. **CORRECT — The saved ROM cost numbers and label joins reproduce.**  
   I independently recomputed the JSON selection without running the script’s writing entry point. Both recorded source hashes match. Each selected ROM has 18 valid triplets across six cases. Results are:
   
   | Target | Selected deployed ROM | Worst error | Median ratio to A | Median seconds |
   |---|---|---:|---:|---:|
   | 1% | None | Best available: 1.77468% | — | — |
   | 3% | R256, gauss128, CNR, Δt=0.0125 | 2.71379% | 0.3300868 | 0.0324192 |
   | 10% | R128, fib4181, BE, Δt=0.0125 | 6.73174% | 0.1505789 | 0.0147285 |
   
   The checkpoint prefix and CNR labels match the actual timing records. Restriction to deployed LSPG schemes agrees with jcp-fom2d’s amended primary population.  
   **Fix:** Preserve these numbers, subject to the population correction below.

4. **WRONG — `rom_box.py` searches an incomplete qualified population.**  
   It reads only `members`: six LSPG configurations. There are another **15 qualified LSPG configurations in `other_qualified`**, all with C2 timing. The inherited [targets.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/jcp-fom2d/targets.py:95) deliberately reads both. Front dominance uses tolerances and earlier timing; it does not prove exact cheapest-at-target status under C2.  
   **Fix:** Search both collections, validate both numerical-qualification labels, and check C2 completion, gates, vetoes and case/repetition coverage. **Recomputing all 21 leaves the present winners and the 1% conclusion unchanged.**

5. **WRONG — The proposed reference acceptance is not the inherited E0 recipe.**  
   [verdict.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/jcp-references/verdict.py:84) additionally checks shared initial data, accumulated solver-floor separation, and Richardson error-vector direction consistency. DESIGN replaces these with a two-case, coarse-level tolerance comparison. It also omits E0’s independent cross-check. Literally requiring order at “every \(t_k\)” includes \(t_0\), where shared initial data give undefined \(0/0\).  
   **Fix:** Port the actual acceptance predicates, apply order checks only at \(k=1,\ldots,5\), and register an independently converged variable-coefficient spectral cross-check on a fixed subset. A coarse-level tolerance rerun cannot certify the finest reference’s solver floor.

6. **NEEDS-RESTATEMENT — Coarser reference levels are permissible only after validation.**  
   The ratio-two Richardson formula is correct, and \(\epsilon_{\rm ref}\le10^{-3}\) is one tenth of the tightest target. But small cell Péclet number does not establish the asymptotic regime; simultaneous space/time refinement can also conceal error cancellation. The 75% fallback rule permits excluding up to eight difficult cases without refinement.  
   **Fix:** Freeze a common eligibility mask before competitor selection, report every exclusion and its parameters, define which cases receive fallback, and impose a minimum coverage requirement for any gate decision. If refinement cannot fit the cap, report an unresolved gate.

7. **WRONG — The stated numerical justification for raising \(\nu_{\min}\) is faulty.**  
   Even the original \(\nu_{\min}=0.01\), at contrast 100, gives \(\kappa_{\min}=0.001\) under the registered normalization and
   \[
   \mathrm{Pe}\le \frac{2}{1024(0.001)}=1.953125<2.
   \]
   Thus raising the bound to 0.02 is not necessary for the stated Péclet threshold. It also cannot isolate “loss of exact diffusion” from changed solution steepness. Grid-normalized extrema do not strictly bound the continuous coefficient.  
   **Fix:** Restore the original range or justify the narrowed family explicitly as a scope choice. Compute Péclet numbers using measured running amplitudes and conservative coefficient bounds; retain paired \(C=1\) controls.

8. **CORRECT — The PSI implicit matrix supports CG.**  
   With positive quadrature weights and \(\kappa>0\),
   \[
   A=\alpha I+4\theta\Delta t\,G^\top W\kappa G
   \]
   is SPD for the proposed implicit schemes. The factor four matches the inverse mass matrix of the unnormalized two-dimensional sine basis. The positive diagonal constant-coefficient inverse is a valid SPD preconditioner.  
   **Fix:** State this proof and define \(G\) as the stacked physical gradients. Implement the transpose exactly; random probes alone are not the justification.

9. **NEEDS-RESTATEMENT — Multigrid-PCG needs a construction-level symmetry contract.**  
   Rediscretized coarse operators do **not** inherently invalidate PCG. However, symmetric diffusion and a symmetric-looking V-cycle are insufficient without fixed linear cycles, adjoint restriction/prolongation, symmetric pre/post smoothing, and positive coarse solves.  
   **Fix:** Specify those relationships and test positive definiteness as well as symmetry. If the implemented preconditioner is variable or nonsymmetric, use a suitable flexible/nonsymmetric Krylov method.

10. **NEEDS-RESTATEMENT — The integrators are sensible candidates, not guaranteed strong competitors.**  
    SBDF2 with first-order startup and CNAB2 with CN–FE startup are reasonable second-order choices. Explicit advection still imposes stability constraints, and CNAB2 may retain stiff transients. ETDRK4 with an explicit variable-diffusion remainder is a legitimate split, but exact treatment of the constant part does not remove remainder stiffness; choosing \(\bar\kappa=\kappa_{\max}\) makes the remainder oppose part of that damping.  
    **Fix:** Register temporal-convergence and stability checks on fixed cases, plus a bounded step-size extension if the smallest tested step is still limiting. Retain the inherited warning that `pscore.nonlinear` is oversampled collocation, **not exact dealiasing**.

11. **NEEDS-RESTATEMENT — The lane-required competitor categories are present; the source’s stronger set is incomplete.**  
    Warm-started spectral CG, split exponential integration, and FD multigrid satisfy the lane brief’s three named categories. Source §7.3 additionally names full-operator Krylov/Leja exponentials, reused factorization, and a semi-implicit ETD remainder. DESIGN explicitly omits the first two and substitutes an explicit remainder for the third. COMMON-BRIEF permits budget-limited omissions with restricted claims.  
    **Fix:** Keep “against the strongest tested competitor” and add **“finite-ladder upper bound on speed-up” beside every ratio**. Before accepting a positive gate, assess whether a bounded implementation of a missing strong competitor fits the cap. Add a predetermined ladder-boundary extension rule; otherwise a boundary winner can make the FOM artificially slow.

12. **WRONG — Numerical admissibility does not satisfy COMMON-BRIEF §3.**  
    Finite outputs, a blow-up guard, and “no non-convergence at maxiter” omit the required analogous in-job certificate, tight-solve distance, and tighter-anchor agreement. Inherited `refcore` discards BiCGStab’s second return value; the text cannot assume reliable convergence flags emerge automatically. Historical ROM qualification is explicitly prior evidence only.  
    **Fix:** Persist independently recomputed true residuals and iteration/termination evidence for every implicit stage, verify schedule completion, and register the applicable tight/tighter comparisons. State the FOM analogue of the common qualification contract explicitly. Revalidate timed ROM subjects and freeze their checkpoint, rule and mode-list hashes.

13. **WRONG — Output shape matches C2, but timing identity is weakened.**  
    Six \(4097^2\) fields, FD sine interpolation, spectral synthesis, and prepared inputs match the inherited contract. However, C2 checks lattice/full-output fingerprints and ROM trajectory/iteration evidence. DESIGN checks only reproduction of a scalar error. Many incorrect fields can have the same scalar error.  
    **Fix:** Retain C2’s subject-level identity checks and vetoes, independently of the \(10^{-9}\) score check. Use separately synchronized executions for T-dbl so compilation cannot eliminate duplicated work.

14. **NEEDS-RESTATEMENT — Per-query setup exclusions extend the inherited timing contract.**  
    C2 excludes IC preparation, but its spectral query computes ETD coefficients inside the timed program, and its FD solver forms the query-dependent preconditioner denominator inside the solve. DESIGN’s blanket exclusion of “preconditioner setup” could exclude materially more work, especially multigrid hierarchies.  
    **Fix:** Enumerate every included/excluded operation. Preserve C2-equivalent kernel timings and also report total per-query setup plus solve/output cost. Phase 2 must charge variable-coefficient reduced-operator assembly consistently.

15. **WRONG — Stage 2 is used for further selection.**  
    DESIGN selects within each family in stage 1, then defines the overall fastest family using stage-2 costs. This breaks its stated selection/inference separation. Inherited C2 has explicit union groups selected before stage 2.  
    **Fix:** Select and freeze the union winner in stage 1; use fresh stage-2 measurements for its reported ratio. Report family-specific inference separately.

16. **NEEDS-RESTATEMENT — The bootstrap is not paired across physical ROM/FOM cases.**  
    Independent resampling of the two workloads is correctly disclosed, but calling it a “paired case-bootstrap” is misleading. The pairing is temporal A–B–A. Furthermore, eight FOM cases use A cases \(i\bmod6\), giving a different anchor-case weighting from the six-case ROM panel.  
    **Fix:** Define the estimand and anchor schedule precisely. Use balanced anchor weighting or a common fixed anchor, and bootstrap whole physical-case records while preserving shared timing-block dependence.

17. **WRONG — “Lossless pruning” is not lossless as written.**  
    A configuration failing 10% on an **ineligible** dev reference may still satisfy the final target over eligible pooled cases.  
    **Fix:** Prune only using the frozen eligible dev population. Numerical solver failures should separately disqualify configurations according to a declared rule, not disappear through eligibility filtering.

18. **NEEDS-RESTATEMENT — Cohort splitting is sound, but the final-test policy is incomplete.**  
    Seed-separated dev/val/test cohorts and dev+val selection comply with COMMON-BRIEF §2. Keeping test closed during the exploratory gate is sensible. However, the design says test is never run in Phase 1, whereas the common brief calls for a final test-once evaluation of new cohorts.  
    **Fix:** Specify a frozen final FOM evaluation if Phase 1 ends the lane, or explicitly limit the stopped result to developmental evidence. Also carry forward the required E0 A10, combined A10/A11, and retracted A15 labels; the inherited front contains them, but the extracted ROM JSON/design omit them.

19. **WRONG — Several “must-fail” controls have no guaranteed failure power.**  
    T-kappa’s \(>10\%\) threshold is a scientific effect-size hypothesis, not a guaranteed code check. T-loose can remain close to the tight solution because of warm starts or a favorable case. T-ref leaves perturbation direction, time support and normalization unspecified; a fixed perturbation need not force rejection. Constant-coefficient identities cannot catch every variable-coefficient gradient/flux error.  
    **Fix:** Separate scientific diagnostics from implementation controls. Register deterministic fixtures that force residual rejection and independently violate each reference predicate. Add a manufactured variable-coefficient solution containing the \(\nabla\kappa\cdot\nabla u\) contribution. Expand T-gate to cover exactly 2, missing ROM targets, empty FOM sets, failed references and contrast-10-only success.

20. **NEEDS-RESTATEMENT — The 6.2 GPU-hour estimate is an unvalidated placeholder.**  
    Arithmetic is correct, but the reference allocation implies roughly 169 seconds per three-level case for 64 case/contrast combinations, before calibration and fallback. Those references contain about 45,184 nonlinear stages in total. The ladder has 338 configurations per contrast—5,408 dev trajectories before validation, compilation and screening. Doubling reference resolution and halving time steps raises basic two-dimensional grid-step work approximately eightfold, before iteration growth.  
    **Fix:** Require smoke certificates covering reference levels, difficult coefficient draws, compilation, validation survivors, and memory with the bank resident. Explicitly reserve fallback cost. Mandatory reference/qualification work must not be silently deferred to preserve a positive gate.

**Overall verdict: WRONG — fix the blocking contracts and re-audit before production; the currently reported ROM cost values themselves are correct.**