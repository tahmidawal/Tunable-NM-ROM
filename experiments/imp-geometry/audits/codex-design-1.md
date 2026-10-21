**OVERALL: NOT-READY.** The design has a useful comparison structure, but its reference fallback, competitor definition, qualification rules, and compute estimate do not yet support the proposed verdict. This was a read-only audit; no files changed and no jobs submitted.

References below to sections mean [E2 DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-geometry/experiments/imp-geometry/DESIGN.md).

1. **WRONG — win/withdraw rule and operational interpretation (§9).**

   The quotation is nearly verbatim, but adds “and” before “label S.” Fix that literal-copy requirement.

   Choosing R-BANK as “the ROM,” worst error as primary, and requiring both steepness levels is defensible. Three targets per level and two geometries create multiple declared opportunities, but that is not inherently p-hacking if frozen beforehand and all outcomes are reported.

   The substantive problems are:

   - **R-PODG is excluded from “best competitor,” despite being evaluated.** A bank could therefore win while a tested POD–Galerkin implementation is faster.
   - Test-once confirms accuracy only. It does not confirm the speed ratio or explicitly renew numerical qualification.
   - “Same S” is operationalised as \(S_{\rm box}\ge S_{\rm geom}/2\). This is a conservative, newly chosen threshold, not something supplied by the brief.
   - Attribution is withdrawn “for that cell” but survives through “at least one winning cell per level”; there is only one wide-family cell per geometry/level. This appears to mean **target**.
   - Failed or censored box controls have no specified attribution outcome.

   **Fix:** include R-PODG in the competitor minimum; freeze all validation-selected finalists before opening any test results; specify confirmation of accuracy, qualification, and either speed or explicitly validation-only timing. Define attribution per target and its aggregation. Missing box evidence must mean attribution unestablished. Distinguish a family speed win from a geometry-attributed win.

2. **WRONG — geometry motivation; clearance arithmetic is CORRECT (§1).**

   The scale, hole, family widths, bump widths, and viscosity bands are compatible with the brief’s latitude. S1/S2 match the pilot’s A2/A3 viscosity bands.

   The stated minimum clearances check out:

   | Quantity | Calculated value |
   |---|---:|
   | Outer-wall clearance | \(0.30\) |
   | Maximum outer-wall Gaussian trace divided by \(a\) | \(1.523\times10^{-8}\) |
   | Hole clearance | \(\sqrt{0.3^2+0.3^2}-0.2=0.224264\) |
   | Maximum hole trace divided by \(a\) | \(4.2804\times10^{-5}\) |
   | Re-entrant-corner clearance | \(0.424264\) |

   These traces are small relative to 1–10% targets, although overwriting boundary values does **not** make the continuum initial condition exactly compatible.

   The interaction argument is wrong. A line along \((1,1)\) preserves \(x-y\); its distance from \((1,1)\) is \(|c_x-c_y|/\sqrt2\), reaching \(0.282843>0.2\). Some centre paths miss the hole entirely. Moreover, \(aT\) is a maximum-speed displacement scale, not a prediction of bump motion: amplitude decays, and Burgers transport deforms the bump. Centres near \((0.3,0.3)\) cannot reach the L-corner by \(T\), even using the stated maximum displacement.

   **Fix:** remove “on the advection path of every bump.” Pre-register a geometry-interaction diagnostic, such as reference-field differences from the paired box on the common domain, plus feature-neighbourhood energy. Require a meaningful effect before attributing a win to interaction with the feature. If it fails, revise the problem before production or restrict the claim. Check initialisation/boundary-layer sensitivity for the high-order arm.

3. **NEEDS-RESTATEMENT — cohorts are structurally sound (§2).**

   Sharing descriptors across geometries is correct. Train192/dev8/val32/test32 is consistent with selection followed by independent confirmation. The seed formula separates the named streams.

   However, different levels use independent descriptors, unlike the pilot’s paired viscosity sweep. This supports distributional comparisons, not a paired steepness effect. “Disjoint from every campaign seed” also exceeds what checking two named sources establishes.

   Freeze RNG draw order, bank seeds, descriptor hashes, and the complete finalist manifest. Explicitly prohibit using test reference eligibility, test timings, or test failures to choose replacement configurations. Clarify whether the eight-case Q2 validation subset is the intended scope of qualification.

4. **WRONG — references and fallback (§3).**

   Second-order FD/SBDF2 is a reasonable starting method, but the transplantation is not justified merely by its nominal order.

   - The L-corner’s \(\rho^{2/3}\) singularity can reduce uniform-grid convergence; a universal \([1.8,2.2]\) observed-order requirement can reject valid converging sequences.
   - Shortley–Weller can achieve second-order solution accuracy for smooth problems despite lower local boundary consistency. That does not guarantee clean Richardson ratios on this nonlinear problem and these three cut-boundary grids. Its near-boundary matrix is generally nonsymmetric. [Primary numerical-method study](https://www.sciencedirect.com/science/article/pii/S0021999112007292)
   - Refining space and time together can conceal cancellation. There is no independent temporal anchor.
   - If “every \(k\)” includes \(t=0\), identical initial samples give \(0/0\) in the observed-order calculation.
   - With 24–31 eligible cases, treatment of the remaining cases is undefined. Dropping them would change “worst over val32.”
   - The fallback calls itself two-way but supplies \(U\), while §4 applies a three-way rule. A last-grid difference is not automatically a reference-error estimate.
   - SEM disagreement is diagnostic in §3 but becomes a SEM failure in §9. It could instead expose an FD reference error.

   **Fix:** specify positive-time order checks, zero-difference handling, independent time refinement, and a corner-aware convergence/refinement procedure. Define casewise eligibility without silently shrinking the cohort. An unqualified-reference fallback may support numerical-reference diagnostics, **not a physical-accuracy WIN or E4 release**. Resolve independent-solver disagreement before assigning fault.

5. **WRONG — competitor strength and specification (§5.1).**

   Static condensation is not intrinsically mandatory for a strong GPU SEM implementation. However, “mass-dominated, therefore \(O(10)\) Jacobi-PCG iterations” is an unsupported assumption across the proposed orders, grading, meshes, and timesteps. The relevant balance includes the largest discrete diffusion eigenvalues. Published SEM Helmholtz comparisons show substantial differences between Jacobi and multilevel/Schwarz preconditioning. [Primary SEM solver study](https://www.mcs.anl.gov/~fischer/pubhtml/mg.pdf)

   Recording a weakness only above 50 iterations does not establish competitiveness below 50.

   Other gaps:

   - Grading ratio alone does not define graded meshes; layers, connectivity, and mappings are missing.
   - Over-integration needs an explicit projection and curved-geometry quadrature contract.
   - SBDF3 startup is unspecified.
   - FD multigrid lacks damping, coarse solve, hierarchy termination, cut-boundary transfers, and iteration caps.
   - C-SPEC has no numerical \(N\) ladder.
   - ECSW needs a defined fitting population, residual normalisation, support selection, and accounting for stencil neighbours.
   - Forcing every reduced competitor to use the bank’s chosen stepper can handicap POD/ECSW.

   **Fix:** benchmark a stronger SEM solve option—condensation, multilevel preconditioning, or an appropriately charged reusable factorisation—against Jacobi-PCG on representative cases. Freeze the missing specifications and independently tune competitors’ steppers. Include R-PODG in the minimum. No additional method is guaranteed to beat the bank, but the already-planned exact-tensor POD arm is an obvious omitted contender.

6. **WRONG — reduced-arm mathematics is incompletely pinned (§5.2–5.3).**

   Stage-A-only training is an acceptable disclosed variant. The envelopes have the intended zero sets, and the L-corner limitation is correctly acknowledged.

   The remaining problems matter:

   - With bank width 512, \(M=\min(4r,512)\) gives \(M/r=2\) at \(r=256\), and only \(4/3\) at \(r=384\). This is not a uniform transplant of \(M=4r\).
   - A longer prefix of the bank’s own span can miss residual components outside that span. It needs an independent residual diagnostic.
   - The QR/SVD recipe needs explicit weighted matrices, SVD orientation, rank cutoff, and zero-code handling.
   - If each quadrature candidate defines its own QR/rotation, Q2 can change the trial space while supposedly testing quadrature.
   - \(z_0=\Psi^\top Wu_0\) is valid without a mass solve only when the basis is orthonormal in that particular rule.
   - The hole’s projected FD Laplacian is not automatically symmetric in the stated grid inner product. An orthogonal diffusion diagonalisation cannot simply be assumed.
   - “Exact tensor” means exact contraction of the chosen discrete/quadrature operator, not exact continuum integration.

   **Fix:** freeze one accurately integrated basis transformation; form rule-specific mass/operators consistently; specify the generalised initial projection and ETD treatment of nonsymmetric operators. Define \(R_{\max}\) for each arm, acknowledge test-space saturation, and add independent residual tests. Preserve arm-specific stepper tuning.

7. **WRONG — metrics are reasonable, but search and timing contain contradictions (§4–6).**

   Using \(e\pm U\) with \(U=2\max\varepsilon_{\rm ref}\) is conservative **conditional on valid reference estimates**. It does not resolve the fallback problem.

   - FD resolution 96 cannot subsample every \(i/64\) scoring point. The claim that all ladder resolutions are multiples of 64 is false.
   - “Fine step” in the initial spatial screen is unspecified.
   - Screening spatial feasibility by worst error discards configurations potentially useful for the advertised median frontier.
   - The step-distance condition must be a **search-stopping heuristic**, not a requirement that every candidate be temporally converged to \(10^{-4}\). Otherwise 1–10% comparisons oversolve competitors.
   - The first \(n=5\) rung has no valid \(n/2\) comparison.
   - Three validation cases provide weak timing coverage for variable-iteration solvers.
   - Raw timing arrays, A–B–A aggregation, qualification overhead, and host/device input-output boundaries are not explicitly pinned.

   **Fix:** specify charged interpolation or compatible FD resolutions; score every valid visited timestep; maintain separate worst/median searches; define first-rung and ladder-cap handling. Persist all timings and compute accuracy from the timed solver outputs. State whether mandatory in-job certificates are charged, and use a representative timing cohort or restrict the performance claim to the three cases.

8. **WRONG — Q1/Q2 do not fully satisfy the common contract (§7).**

   ETDRK4 step-halving is a time-discretisation check, not the requested **in-job, every-step analogue** of the algebraic certificate. It also imposes an accuracy restriction absent from the implicit arms’ algebraic Q1 checks.

   The one-pass linearised CN alternative has no qualification contract.

   Iterative competitors may hit their cap in up to 1% of steps, despite those being called *invalid* steps. This conflates allowed nonstationarity with invalid solver exits. Their tight/tighter anchor is also absent.

   Q2 omits:

   - a \(q=14\) successor for the selectable \(q=12\);
   - zero/small nonlinear-vector denominators;
   - independent mass, stiffness, and initial-projection accuracy;
   - empirical-quadrature qualification for ECSW.

   Calling ECSW hyper-reduction “the arm itself” does not establish its numerical fidelity.

   **Fix:** define per-step ETD stage/update consistency and finiteness checks, and separate temporal refinement from algebraic qualification. Define linearised-CN residual checks. Require converged iterative solves and a tight/tighter anchor. Add the missing quadrature endpoints and operator checks; qualify ECSW against its full projected operator or explicitly obtain a narrower contract.

9. **WRONG — several must-fail controls need not fail (§8).**

   The prescribed outcomes are not guaranteed:

   - MF-REF’s fixed perturbation can accidentally produce an admissible ratio.
   - MF-SEM’s four-decade requirement depends on the manufactured solution, mesh, geometry map, and algebraic tolerance.
   - A staircase method need not exhibit order below 1.5 on the selected finite ladder.
   - An identity smoother can leave a coarse-representable manufactured error that coarse correction eliminates.
   - One LM iteration can genuinely converge.
   - \(q=2\) can pass on sufficiently subdivided elements.
   - A zeroed tensor slice can be inactive.
   - Random support can work when the support is large enough or the family is simple.
   - Random-span failure depends on its ambient space and construction.

   “A control that cannot fire … is a bug” is therefore false and invites changing controls after seeing outcomes.

   **Fix:** use predetermined, detectable faults in real computed states: measured nonzero tensor contributions, verified unresolved residual components, omitted boundary contributions, and known matcher perturbations. Keep coarse-method/random-rule performance as diagnostics unless failure is mathematically ensured. Add controls for scoring/interpolation, reference fallback, confirmation leakage, and hidden per-query setup. Ensure doubled-query timing cannot be compiler-eliminated.

10. **WRONG — the 62 GPU-hour estimate is unsupported (§11).**

   The S2 finest references alone entail approximately

   \[
   72\times2\times\bigl(3+(4-\pi0.2^2)+4\bigr)
   \times2048^2\times16000
   \approx1.05\times10^{14}
   \]

   grid-point time steps. Fitting those alone into eight GPU-hours requires about \(3.65\times10^9\) grid-point steps/s, before multigrid cycles, other levels, S1, and overhead.

   Twelve 300k-step bank fits at 0.6 GPU-hours each imply 7.2 ms per training step averaged over the entire fit. The pilot estimated approximately three A100-hours per bank. The reduction needs evidence, especially with the ambiguous “full-batch last 10k.”

   POD construction, tensor formation, NNLS, compilation, and large intermediate storage also lack explicit allowances.

   **Most likely budget failure:** reference generation with tight multigrid solves; bank fitting and the extensive SEM ladder are next.

   **Fix:** require actual full-size time/memory certificates before release and count setup, failures, compilation, and qualification. Define full-batch semantics and streaming storage. Defer whole secondary cells before removing strong competitor branches while retaining a strong-comparator claim. An incomplete required wide-family panel is unresolved, not a negative scientific result.

11. **WRONG — additional interpretation risks remain.**

   - A single \(1/64\) scoring lattice may underweight corner and hole-boundary errors. Add a finer-lattice and feature-window diagnostic.
   - Small reference solutions can make a zero predictor competitive under initial-norm normalisation. Report decay and a zero-field baseline at scored times.
   - Paired box controls help, but different geometry-specific training, meshes, and solver implementations prevent automatic causal attribution to “absence of a fast transform.”
   - A finite competitor ladder makes the observed speed-up an upper bound relative to a more thoroughly optimised competitor **with the selected bank fixed**. It is not a statistical upper confidence bound and should not be described as one.
   - Missing evidence, exhausted budget, reference failure, and demonstrated loss need distinct outcomes.

   **Fix:** add these diagnostics and scope every conclusion to the tested family, output contract, implementations, and qualification coverage.

**OVERALL: NOT-READY. Required fixes:**

1. Include every tested contender in “best competitor”; freeze target aggregation, attribution, and confirmation rules.
2. Replace the unsupported interaction claim with a pre-registered physical-effect diagnostic.
3. Define a corner-aware reference procedure and prohibit physical wins from unqualified-reference fallback.
4. Strengthen and fully specify SEM, FD multigrid, ECSW, and spectral ladders.
5. Resolve reduced-space, mass-matrix, test-space, and nonsymmetric-operator ambiguities.
6. Correct scoring-grid compatibility, staged search, and timing accounting.
7. Complete Q1/Q2 for every selectable stepper and hyper-reduced arm.
8. Replace contingent must-fail expectations with detectable fault controls.
9. Produce full-size cost certificates and a fairness-preserving defer plan.
10. Add finer-grid, feature-interaction, decay, and missing-evidence safeguards.