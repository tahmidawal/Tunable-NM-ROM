The design has several blocking errors. Most importantly, a negative result does **not** transfer from this POD proxy to the coordinate-network bank; the stated padding exactness contradicts the vendored solver; and the reference, scoring, and stability rules can produce misleading conclusions.

I reviewed [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md), the vendored solver, and relevant campaign context. No files were changed and no solver jobs were run.

1. **Hypotheses and reading rules — WRONG**

   - **The definitions change midway.** Section 2 defines \(N^*,r^*\) as the *smallest feasible dimensions*. Section 5 defines them as dimensions of the *cheapest feasible configurations*. Those need not coincide. A larger model taking fewer steps can be cheaper. Consequently, the proposed slopes do not have a consistent interpretation.
   - **H-A1’s falsifier has the wrong quantifier.** A claim that \(\alpha\ge0.5\) at *each* target is contradicted by a failure at one target, not only by failures at every target. H-A2 and H-A2′ need similarly explicit target-specific verdicts.
   - **H-A3’s rules are not symmetric.** Positive evidence needs one favorable target; negative evidence needs every target, uses a different speed threshold, and does not separately classify a reversal. Calling these rules symmetric is false.
   - **“Not reachable in this family” exceeds the experiment.** A nonpositive fitted slope over three or four small cohorts cannot exclude a later crossover. Conversely, extrapolated parity at \(s\le2000\) is a model-dependent forecast, not evidence that crossover is reachable.
   - Different random centers, widths, and amplitudes at each level confound steepness with cohort difficulty, especially when the outcome is a maximum over only 12 cases.
   - Dropping failed steep references creates informative censoring: the experiment can lose precisely the levels needed to test its premise.

   **Fix:** distinguish minimum feasible size from cost-optimal size. Issue verdicts per target, require at least three common eligible levels for trend fits, and use paired descriptors across viscosity levels. Report “observed crossover,” “extrapolated crossover under this fit,” “reverse trend,” and “unresolved.” Reference failure or ladder exhaustion must produce unresolved/censored outcomes, not evidence against crossover. Define outcomes when neither method—or only one method—reaches a target.

2. **Optimistic-proxy logic — WRONG**

   POD minimizes **aggregate squared training reconstruction error** among fixed-rank linear spaces. It does not minimize held-out worst-case error, the maximum over time, rollout error, or computational cost.

   The proposed proxy has optimism in some directions:

   - It avoids bank-fitting error on the training snapshots.
   - It removes nonlinear iterations and Jacobian construction.
   - It permits precomputed output bases and selects between nonlinear evaluation implementations.

   But it has pessimism or uncontrolled differences in others:

   - Finite-data POD can generalize worse than a learned spatial basis.
   - Average-error POD can allocate modes poorly for a worst-case objective.
   - Galerkin dynamics can be less accurate or stable than another reduced formulation in the same span.
   - Exact integration of reduced diffusion does not make the nonlinear evolution optimal.
   - Dense tensor evaluation is not a lower bound on bank evaluation cost; structured tensors, sparse rules, smaller effective coordinates, and alternative initializers can be cheaper.
   - The B proxy carries the spatial-discretization limitations of its \(L=512\) training/operator grid.

   The bank’s outputs lying in a linear span does not establish any of the missing inequalities. R5, even if completed, compares only particular existing banks and data; it cannot validate a universal negative-transfer claim.

   **Fix:** call this a **POD–Galerkin feasibility comparator**, not a performance bound. A loss means this registered implementation loses. A win licenses further investigation. Separate representation error, evolution error, and measured cost. Any rigorous exclusion must be restricted to a specified span, output contract, and demonstrable cost lower bound.

3. **Timing, query contract, and tuning fairness — NEEDS-RESTATEMENT**

   Same-device CPU timing, affinity, f64, warm compilation, initialization charges, decoding charges, and explicitly developmental matching are sound choices. They are insufficiently specified in several consequential places:

   - Synchronize completed JAX outputs before stopping timers. Persist every repetition, case timing, outlier count, and sentinel measurement.
   - A drift flag needs a consequence: retry under a fixed rule, then invalidate. Pair/interleave competitors and prevent concurrent all-core preparation jobs during timing.
   - Charge parameter-dependent ETDRK4 coefficient construction for both methods. Permit equivalent parameter-independent precomputation for both.
   - `ic_coefs_2d` uses numerical Gauss quadrature, not symbolic exact coefficients. Its implementation reconstructs quadrature/sine matrices. Charging this avoidable setup repeatedly while allowing extensive ROM precomputation can dominate the comparison unfairly.
   - The spectral step filter can remove accurate, fast large-step settings, particularly when diffusion stabilizes the dynamics. The ROM’s corresponding restriction is undefined because its state has no unique \(N\).
   - Minimum resolutions \(N=16\) and \(L=64\) can exclude cheaper competitors at loose targets. The reference-relative upper cap can exclude necessary competitors at tight targets. Add bounded edge extensions and censored outcomes.
   - **Quadrature accuracy cannot be borrowed from the tensor rollout.** Distance \(\le0.1\tau\) from a tensor solution with error \(\le\tau\) permits quadrature error up to \(1.1\tau\). Score and time the actual chosen quadrature implementation.
   - Shape-only contraction timing is a component estimate, not an end-to-end measured query. High-r accuracy must still come from an explicitly defined executable solver.
   - “Gauss quadrature” for the B FD ROM needs an off-grid reconstruction and derivative definition; otherwise it is not clearly evaluating the same discrete operator.

   **Fix:** freeze an end-to-end accounting contract, separate component estimates from measured speed ratios, and retime shortlisted winners with fresh repetitions. Symmetric in-sample accuracy matching is acceptable for this pilot, but label selection uncertainty and avoid generalization claims.

4. **Must-fail controls — WRONG as mandatory physical assertions**

   Several controls can legitimately fail to reject their designated “bad” method:

   - **MF1:** a coarse reference can already meet a loose target on smooth, strongly diffused data.
   - **MF2:** \(N=16\) is not guaranteed to miss 1% error merely because the nominal viscosity is small. Actual front formation, width, and sampled times matter.
   - **MF3:** compiler elimination of duplicated pure work, fixed overhead, or asynchronous dispatch can prevent a 2× ratio even without an incorrect timer.
   - **MF4:** zeroing an inactive tensor output slice may have negligible effect. No slice-selection or excitation rule guarantees the prescribed response.
   - **MF5:** poor representation at initialization does not guarantee failure at the first scored time after diffusion.
   - **D1:** stability between 48 and 96 samples does not establish data sufficiency; an improvement is evidence of sensitivity, not a quantified pessimism bound.

   Therefore, “a control that cannot fire … is a bug” is itself wrong. Sometimes the assumed physical effect is absent.

   **Fix:** distinguish deterministic software fault-injection checks from physical stress diagnostics. Test matcher rejection using deliberately perturbed real fields with known errors. Test tensor corruption on states that demonstrably excite the corrupted contribution. Make both outputs of doubled timing work observable and synchronized. Keep coarse-grid/random-span failures as diagnostics unless their separation has been established on frozen fixtures.

5. **Reference adequacy — WRONG for B; NEEDS-RESTATEMENT for A**

   **Part A:**

   - Two spatial resolutions and one time-halving comparison are convergence evidence, not error bounds. Shared quadrature bias or error cancellation can pass both.
   - The spatial threshold \(10^{-4}\) already consumes 10% of the tightest target; temporal and quadrature uncertainties add to it.
   - “At \(N_{\rm ref}\) size or the coarser one” leaves the comparison norm ambiguous and permits omission of fine-scale differences.
   - At A4’s maximum nominal steepness, \(4/s=0.005\), only about four \(N=768\) intervals span that nominal width. This does not prove under-resolution, but it makes adequate convergence evidence essential.
   - Training trajectories also need numerical qualification. Qualifying only test references does not ensure trustworthy POD snapshots.

   **Part B:**

   - Dividing by three assumes second-order convergence. Reporting an observed order without gating on it does not justify that assumption.
   - A \(3\pi/2\) re-entrant corner admits the leading Dirichlet singular behavior \(\rho^{2/3}\sin(2\theta/3)\). Smooth-domain second-order convergence cannot simply be assumed on uniform grids.
   - There is no separate temporal-reference gate.
   - A coarse scoring lattice can hide the very corner error the reference is supposed to resolve.

   **Fix:** require consistent spatial refinement on a sufficiently fine common scoring representation, separate time refinement, and an independent spatial/quadrature check on representative hard cases. For B, use a supported observed-order estimate with an asymptotic-consistency gate; otherwise mark uncertainty unqualified. Attach a reference-uncertainty margin to target eligibility and matching. Failed A3/A4 references leave the steep-front question open.

6. **Scoring lattice, metric, and “floors” — WRONG**

   The common metric and initialization-based normalization are legitimate deployment choices. They do not establish continuum accuracy.

   - On A’s scoring lattice,
     \[
     \sin(256\pi i/256)=0.
     \]
     Thus an entire represented mode is invisible; higher modes alias. A3/A4 references and some competitors exceed the lattice’s resolving capacity.
   - B’s \(h=1/64\) scoring lattice can miss corner-localized errors between its nodes.
   - **D2 is mathematically invalid.** Coefficient-space POD projection minimizes continuous spectral \(L^2\) error. Full-grid FD projection minimizes that grid norm. Neither necessarily minimizes error after restriction to the scoring lattice. A rollout may legitimately beat the sampled error of those projections.
   - Spectral truncation likewise need not be the best fit in an aliased sampled norm.
   - Five positive output times can miss early transient errors, and normalization by initial energy can make later, heavily decayed solutions comparatively easy to approximate.

   **Fix:** compute scoring-norm floors by weighted least squares using the actual sampled basis; use rank-revealing QR/SVD. Report continuous/full-grid projection errors separately. Add scoring-grid refinement and a corner-sensitive diagnostic. Keep the six-field output contract if desired, but restrict conclusions to those outputs unless additional temporal checks support broader claims.

7. **Missing strong competitors — NEEDS-RESTATEMENT**

   H-B is testable as “beats these two FD implementations.” It cannot establish “ROM wins without a fast transform” or superiority over the best FD solver.

   Important omitted competitors include:

   - IMEX diffusion solves with geometric multigrid or suitable preconditioned Krylov methods;
   - stabilized explicit diffusion methods with an advection-compatible treatment;
   - better-performing IMEX schemes and reusable sparse ordering/setup;
   - adaptive or corner-graded high-order FE, spectral elements, and multi-domain methods.

   Lack of a global sine transform does not imply expensive diffusion solves. The L-shape also permits structured subdomain methods.

   **Fix:** narrow the pilot claim explicitly and include at least one scalable diffusion treatment before interpreting B as evidence about geometry. High-order methods may remain deferred, but their absence limits a positive conclusion. A finite FOM ladder can make the reported ROM advantage too favorable.

8. **Thirty-minute CPU feasibility — WRONG as presently budgeted**

   The classical RK4 diffusion stability condition is approximately
   \[
   \Delta t\,8\nu L^2\lesssim2.7853,
   \qquad
   n\gtrsim\frac{0.25\,8\nu L^2}{2.7853}.
   \]

   This gives approximate diffusion-only requirements:

   | Grid | \(\nu=0.02\) | \(\nu=0.05\) |
   |---|---:|---:|
   | \(L=512\) | 3,765 steps | 9,412 steps |
   | \(L=1024\) | 15,059 steps | 37,647 steps |

   Advection can impose additional restrictions. Thus the registered 2,560-step ladder has no stable \(L=512\) classical-RK4 member across B’s viscosity band. If references and training use separate, longer ladders, those must be specified and budgeted.

   Other unbudgeted costs are substantial:

   - A4’s \(767^2\times2016\) f64 snapshot matrix alone occupies about **9.49 GB**, before POD workspace.
   - B’s training snapshot matrix is approximately **3.17 GB**.
   - A packed \(r=512\) quadratic tensor occupies approximately **538 MB**.
   - Fine-grid tensor assembly, POD, many compiled shapes, all-case accuracy sweeps, and repeated timings can dominate the proposed schedule.

   These are workload estimates, not measured runtime predictions; the current minute estimates have no supporting benchmark.

   **Fix:** register short stage benchmarks and explicit memory/work limits before production. Use streamed snapshots, a justified truncated-POD procedure, smaller staged ladders, and a more efficient B reference integrator. Specify checkpointable job units and what becomes unresolved at the time cap. Merely splitting a large workload into 30-minute jobs does not validate its total budget.

9. **Specific numerical claims — mixed verdicts**

   - **3/2-padding exactness — WRONG.** The vendored [pscore.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/vendor/pscore.py:4) explicitly describes *oversampled collocation*, not exact de-aliasing. Sine/cosine products on this Dirichlet domain do not inherit the standard periodic quadratic-product exactness argument. A tensor can reproduce the chosen discrete quadrature exactly; that does not make it the exact continuous Galerkin tensor. Rename it and test quadrature convergence independently.

   - **Tensor storage and evaluation — CORRECT with qualifications.** There are \(r^2(r+1)/2\) packed entries after symmetrizing the two input indices. Off-diagonal pair contributions need the correct factor. Evaluation is \(O(r^3)\); its actual operation count and packed-layout performance require measurement.

   - **“Crossover needs \(r\) roughly below \(N\)” — WRONG as a derived rule.** Equal tensor/FOM per-step work instead gives, up to constants,
     \[
     r_\times\sim(N^2\log N)^{1/3}.
     \]
     For quadrature, the condition depends on \(m(r)\). Total-query crossover also depends on step counts, initialization, and output cost. Use the measured \(r_\times(N)\), not \(r/N\) as a purported cost criterion.

   - **Reduced Laplacian diagonalization — CORRECT under explicit conditions.** The spectral operator is symmetric negative definite. The standard interior-node Dirichlet five-point L-shape operator is also symmetric negative definite: the missing rectangle does not destroy symmetry. Its orthogonal Galerkin restriction can therefore be diagonalized offline independently of scalar \(\nu\). This assumes correct boundary elimination and inner-product normalization; otherwise a mass matrix/generalized eigenproblem is needed. Rotate every operator, initializer, and decoder consistently. Diagonalize each selected POD prefix separately—truncating a rotated larger basis does not generally preserve the original POD prefixes.

   - **SBDF2 “one factorization per query” — WRONG with the stated start.** IMEX Euler requires \(I-\Delta t\nu\Delta_h\), whereas subsequent SBDF2 steps require \(\tfrac32I-\Delta t\nu\Delta_h\). One ordinary LU factorization does not solve both. Charge both or specify and validate a different startup.

   - **Front width and advective CFL — NEEDS-RESTATEMENT.** \(4\nu/\Delta u\) is a one-dimensional tanh scale. For a planar front with unit normal \(n\), the advective coefficient is \((1,1)\cdot n\), changing that scale. Actual jump, orientation, diffusion, and front-formation time matter. Likewise, \(0.6\) is an unvalidated heuristic here, not an ETDRK4 stability theorem. Neither should determine eligibility without checks.

   - **B boundary-amplitude bound — CORRECT.** The worst outer-wall Gaussian tail is \(e^{-8}a\approx3.35\times10^{-4}a\). However, this does not establish that the evolving bump meaningfully probes the corner; measure that.

   - **R-function envelope — NEEDS-RESTATEMENT.** It enforces the stated boundary zeros, but near the re-entrant corner it behaves like \(\rho\), whereas the leading singular solution can behave like \(\rho^{2/3}\). The quotient can therefore be singular. Polynomial-envelope fitting is a useful diagnostic, not proof of efficient neural-bank representability.

The minimum repair is to withdraw negative transfer, correct padding and floor claims, qualify references in a resolved norm, repair the B stability/startup accounting, and replace the crossover rules with consistent target-specific decisions.

OVERALL: NOT-READY