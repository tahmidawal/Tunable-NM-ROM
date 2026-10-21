A1 repairs several central errors, but **the design is not yet ready even as a small feasibility pilot**. The remaining blockers concern executable rules and valid interpretation, not publication-level evidence. I read the canonical lab log, both design versions, and `common.py`/`gen_a.py` for context. No files were changed and no jobs were run.

1. **Hypotheses and reading rules — NEEDS-RESTATEMENT**

   [A1.2–A1.3](experiments/spectral-pilot/DESIGN.md# a12-cohorts-paired-across-levels-compatible-initial-condition) fixes the size-versus-cost ambiguity, target-specific verdicts, negative extrapolation claim, and treatment of unsuccessful methods and references.

   **The paired construction is correct:** sharing \((c_x,c_y,w,a,u_\nu)\) and changing only the viscosity scale preserves each row’s initial condition and multiplies its nominal steepness by three. The context code uses exact factors of three; the table’s rounded bands are harmless.

   Two qualifications remain:

   - A4 uses 48 training trajectories while earlier levels use 96. Its ROM comparison therefore changes both viscosity and training budget. **Fix:** use the existing 48-trajectory arms for a common-budget trend through A4, and report the 96-trajectory arms separately.
   - Small wall values remove a substantial incompatibility issue, but do not establish that the experiment “isolates front steepness.” Changing viscosity also changes diffusion and decay; the initial-energy normalization can make strongly decayed solutions easy. **Fix:** describe this as a paired viscosity sweep and report a simple front-gradient and remaining-energy diagnostic.

   H-A2′ also overstates its interpretation: a measured **right-hand-side evaluation** crossover is not automatically a **time-step** crossover. ETD vector operations and other work remain. **Fix:** say “nonlinear evaluation cost,” define how \(r_\times\) is obtained from discrete measurements, and return “unresolved” if no crossing is bracketed. Use measured query costs for the speed verdict.

2. **Proxy logic — CORRECT**

   [A1.1](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:276) explicitly withdraws negative transfer and recasts the arm as a POD–Galerkin feasibility comparator. This fixes the central round-1 error.

   **Required treatment:** retain that wording in the report. Representation floors, rollout error, and measured cost are separate observations; a loss excludes only the tested implementation. No additional comparator is necessary to establish that limited conclusion.

3. **Timing, contract, and tuning fairness — NEEDS-RESTATEMENT**

   [A1.5](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:358) substantially improves the design: synchronization, interleaving, isolation, stored repetitions, drift consequences, fresh finalist timing, actual operators, and independently scored ROM-Q are appropriate.

   **The stopping rule is still unsafe.** A relative change below 2% in the *scalar error* does not establish time convergence. For example, errors \(1.015\tau\) and \(1.005\tau\) trigger stopping while both fail; a subsequent step refinement could pass. Cancellation between spatial and temporal error can also produce an accidental plateau.

   Importantly, increasing-\(n\) traversal does **not** directly skip earlier, cheaper step counts: those have already run. The danger is prematurely discarding a small spatial model whose later feasible setting could be cheaper than a larger model.

   **Fix:** for this pilot, either run the finite registered step ladder or use a trajectory-distance refinement check with explicit unresolved outcomes near target thresholds. Call minima “smallest observed feasible size” whenever the search ends early.

   Other missing contract details:

   - ROM-Q’s \(M_q\) ladder is unspecified. For B, “values and derivatives synthesised exactly” does not define evaluation of the discrete FD operator away from its nodes. **Fix:** register the ladder and use a defined grid-node sampling operator for B, or defer B ROM-Q.
   - The new query paragraph describes A’s Gauss coefficient initializer and 257² output lattice as applying to “both,” while B needs nodal initialization and its L-shaped output contract. **Fix:** state A and B accounting separately.
   - Store sentinel values and report timing dispersion/outlier counts. If fresh finalist timings reverse the ordering, report the selected configuration’s fresh time rather than claiming a newly established global minimum.

4. **Controls — WRONG**

   [A1.6](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:386) correctly demotes the coarse-grid, random-span, and data-sensitivity assertions to diagnostics. But two supposedly deterministic checks remain nondeterministic:

   - **SW3:** two different cases need not take twice the time of one case. Fixed overhead and differing case costs also prevent a guaranteed 1.8–2.2 ratio.
   - **SW4:** selecting the largest tensor contribution does not guarantee that removing it changes the rollout by \(10^{-6}\). Every contribution may be small, or evolution may suppress the difference.

   **Fix:** make SW3 a harness sanity diagnostic, comparing the combined call with the sum of separately measured calls. For SW4, verify the known instantaneous RHS change on a frozen state with a sufficiently large selected contribution; keep rollout sensitivity diagnostic.

   SW1 also needs a defined baseline. Adding a vector of magnitude \(2\tau\) to an approximate FOM solution does not guarantee error \(2\tau\) against the reference because errors can cancel. **Fix:** perturb the exact field array used as the matcher’s reference, with the same normalization and scored times.

5. **Reference adequacy — WRONG**

   [A1.7](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:406) improves A’s comparison norm and makes failed levels unresolved. Its refinement checks are reasonable **conditional pilot evidence**, provided they are not called error bounds.

   **B’s order gate is not computable as written.** Three resolutions provide two successive differences and therefore only **one** observed order:
   \[
   p=\log_2\!\left(
   \frac{\|u_{256}-u_{512}\|}
        {\|u_{512}-u_{1024}\|}
   \right).
   \]
   They cannot provide separate \(p_1,p_2\) for an asymptotic-consistency test. Orders measured on two scoring lattices are not two successive refinement orders.

   **Fix:** either add a fourth spatial resolution with an explicit common comparison representation, or retain three grids, use one observed order, and label the Richardson estimate conditional without claiming the two-order gate. The latter is sufficient for a limited pilot if borderline conclusions remain unresolved.

   Additional repairs:

   - B’s time check at \(L=512\) on three cases does not qualify the \(L=1024\) reference on all twelve. **Fix:** time-refine the actual reference on the scored cases, or restrict the qualification and conclusions to checked cases.
   - G9 tests an \(r=64\) reduced trajectory. It cannot qualify padding error in the full reference or every larger reduced span. **Fix:** compare reference FOM trajectories at both padding sizes on fixed representative hard cases; retain reduced G9 as a separate diagnostic.
   - The four A training checks are acceptable sampling for a pilot, but their failure needs a consequence. B training qualification remains unspecified. **Fix:** define these as gates or explicitly label affected bases unqualified.

   **Eligibility margins are incomplete.** Requiring \(\tau\ge20\delta\) or \(10\hat e_{\rm ref}\) limits nominal reference contamination, but a measured error just below \(\tau\) can still lie above the target. The maximum of separate gate differences is also not an aggregate uncertainty estimate.

   **Fix:** define a conditional uncertainty allowance incorporating the checked contributions. Accept only when measured error plus that allowance is below the target; classify the intervening band as unresolved. Alternatively, explicitly restrict every target claim to error against the numerical reference and show sensitivity to the allowance.

6. **Metric and floors — CORRECT**

   [A1.4](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:343) fixes the mathematical issue for A.

   Both FOM and ROM represent finite sine expansions. With the stated coefficient convention,
   \[
   \|u-v\|_{L^2((0,1)^2)}^2
   =\tfrac14\|b_u-b_v\|_2^2.
   \]
   The factor cancels in the relative error. Zero-padding the FOM and using \(\Psi z\) for the ROM therefore gives the exact continuous norm **between the represented numerical fields**. It does not certify their distance to the true PDE solution.

   **The floor remains valid under maximum-over-times scoring.** At every scored time, any state in the span has error at least its orthogonal-projection error. Taking maxima preserves that inequality. Taking the maximum over cases preserves it too. B’s sampled-basis least-squares projection similarly repairs its scoring-norm floor.

   **Required implementation clarification:** aggregate the pointwise projection errors with the same time/case maxima and denominator as the rollout metric. Do not substitute an RMS training floor. Allow numerical tolerance in D2, and compute floors using the actual selected span after rotation.

   B’s restricted-lattice metric remains a legitimate limited contract; its refinement and corner diagnostics do not establish continuum accuracy. That limitation is acceptable for this pilot.

7. **Competitors — CORRECT**

   [A1.8](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:424) adequately narrows H-B to the tested FD implementations and explicitly defers stronger alternatives.

   For this small pilot, adding multigrid or high-order methods is **not a readiness requirement** once that scope is enforced.

   **Required wording:** a positive result means “promising against these FD implementations.” It does not demonstrate that losing a global sine transform makes ROMs advantageous. Retain the stronger competitors as requirements for the subsequent broader comparison.

8. **Feasibility — NEEDS-RESTATEMENT**

   [A1.7–A1.9](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:420) adds useful staging, streaming, smaller A4 training data, and honest “not run” outcomes. The extended RK4 ladder addresses the previously obvious diffusion-stability shortfall for B training at \(L=512\).

   But the feasibility rule is not a workload certificate:

   - A2 generation under 15 minutes does not establish that A3 fits 30 minutes; resolution, step count, and validation work all change.
   - A3 under 20 minutes similarly does not qualify A4.
   - Raw snapshot sizes do not bound peak memory for Gram accumulation, basis reconstruction, tensor assembly, compilation, and temporary arrays.
   - There is no equivalent feasibility gate for B or the expensive POD/tensor/ladder stages.
   - The A4 statement “its snapshot set fits in 8 GB” is ambiguous about which level it describes.

   **Fix:** retain the preceding-level times as scheduling heuristics, but add a short stage-specific benchmark and projected peak-memory check before each expensive stage, plus a total pilot cap and checkpointable units. Define whether a timeout means resume or stop. Specify B training’s integrator and step-selection rule.

   Ending with only A1/A2 and “too few levels” is acceptable for a feasibility pilot. It is not a negative crossover finding.

9. **Specific numerical claims — NEEDS-RESTATEMENT**

   The round-1 subfindings resolve as follows:

   - **Padding exactness — CORRECT repair.** Discrete projected RHS equality is distinguished from continuous Galerkin exactness.
   - **Tensor storage — CORRECT.** The packed count and approximate storage are consistent; retain the off-diagonal weighting requirement.
   - **Crossover scaling — CORRECT repair**, subject to the evaluation-versus-step distinction in item 1.
   - **SBDF2 startup — CORRECT repair.** Charging both ordinary LU factorizations is necessary for the stated IMEX-Euler start.
   - **Reduced-Laplacian diagonalization — NEEDS-RESTATEMENT.** A1 still does not explicitly require separate diagonalization of each selected POD prefix and consistent rotation of every operator, initializer, and decoder. Add that requirement; truncating a rotated larger basis changes the intended prefix spaces.
   - **Front-width/CFL claims — CORRECT withdrawal as eligibility rules.** A heuristic initial reference step count is acceptable when actual refinement checks govern acceptance.
   - **Boundary compatibility — NEEDS-RESTATEMENT.** A’s wall bound is correct, but the Gaussian is only approximately compatible; setting boundary values to zero does not make its interior trace exactly zero. Call it “negligible wall mismatch at pilot tolerances.”
   - **R-function representability — NEEDS-RESTATEMENT.** A1 leaves the round-1 caveat unanswered. State that the envelope behaves like \(\rho\), while a leading corner solution may behave like \(\rho^{2/3}\); the quotient can be singular. D3 measures this particular polynomial-envelope space, not neural-bank representability.

OVERALL: NOT-READY