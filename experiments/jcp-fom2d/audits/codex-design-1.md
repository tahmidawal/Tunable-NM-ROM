**Overall verdict: NOT READY.** The design has the right broad structure, but its comparator selection, timing inference, crossover population, and several controls need amendment before implementation.

This was a read-only audit. I changed no files, ran no cluster commands or jobs, and copied no arrays.

Evidence notation below:

- **D** = [this lane’s DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/DESIGN.md)
- **B** = [campaign briefs directory](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs)
- **R**, **S**, **P**, **C** = respectively the `experiments/jcp-references`, `jcp-speed3d`, `jcp-p1-scaling`, and `jcp-combined` directories in the dated worktrees specified in your request.

1. **Question and ownership coverage — NEEDS-RESTATEMENT**

   **Correct:** D:74–124 includes all three families, varying FOM resolution below ROM resolution, time steps, tolerance calibration, joint median/worst matching, separate median-only/worst-only selections, and finite-ladder upper-bound wording. D:129–161 includes paired timing and H-comp at both required meshes. D:204–209 follows the brief’s defer order.

   **Missing or narrowed:**

   - **Same-mesh PS and same-mesh union:** B/`jcp-fom2d.md`:28–29 requests both crossovers for each family and their union. D:116–119 and 150–152 deliver same-mesh only for FD families, without a same-mesh union.
   - **“Cheapest” is not established:** D:113–115 confirms only three candidates selected by **lattice-output** timing, while the question concerns **full-field-output** timing. These orderings need not agree.
   - **K2 covers front members, not every linear setting:** D:160 narrows the governing quantifier. The governing paper plan:135 and PLAN-v3-amendments.md:85–87 say **every linear setting**.
   - **Fallback changes the promised deliverable:** D:211–213 substitutes P1 configurations for the qualified front after twelve hours. That can produce useful partial evidence, but cannot complete FRONT-2D C2/H-comp ownership.

   The positive gates and negative controls are present nominally; some do not yet test their claimed predicates reliably, as detailed below.

2. **Inherited facts — CORRECT, with scope clarifications**

   - **Combined H5:** D:12–15 accurately summarizes the 13.6× selected-scheme and 17.7× best-arm results, same-mesh sign-upwind comparator, tolerance calibration, and worst-only matching. C/`report.md`:127–133 confirms these facts. Preserve its additional qualification: E0 matching was recomputed from existing timings, not newly timed.
   - **Speed3d:** D:16–17 agrees with S/`report.md`:10, 224–231: the \(N=16\) PS comparator matched the selected ROMs, and the reported K2 outcome fired.
   - **P1:** D:18–19 is correct **as evolution timing**. P/`report.md`:183–194 explicitly distinguishes evolution from the complete query. Do not subsequently quote 22–88 ms as full-query latency.
   - **Cohorts:** dev6+val32 comprises 38 physical cases; R/`cohorts.py`:36, 82–85 confirms the construction.
   - **Metric/lattice/eligibility:** D:34–44 agrees with R/`refcore.py`:222–245 and R/`refstore.py`:132–173. The primary reference is the stored **finest** field, not the Richardson-extrapolated diagnostic: R/`refstore.py`:114–128.
   - **S-reference rule and conditional labels:** correctly inherited. R/`report.md`:51 identifies the BE \(0.005\) reference; COMMON-BRIEF.md:39–46 supplies the campaign restrictions.

   One population qualification matters downstream: P1’s 4096² R128/R256 accuracy results use **16 cases**, while several other configurations use 38; P/`report.md`:60, 92–99.

3. **Output contracts — NEEDS-RESTATEMENT; DST scaling is CORRECT**

   For the orthonormal transform in `engines.py`:54–63, a two-dimensional nodal DST coefficient equals \(L/2\) times the corresponding continuous sine coefficient. Consequently, interpolation from \(L\) to \(N\) requires multiplying by **\(N/L\)** before zero-padding and inverse transformation. D:61–62’s \(256/L\) is correct.

   Amend D:66 to state \(N/L\) explicitly. No analogous factor belongs on PS’s already-continuous sine coefficients; R/`pscore.py`:3, 31–35 explains that convention.

   The ROM contract is accurately inherited: its initializer samples the supplied field at quadrature points, and its decoder returns full fields; P/`p1core.py`:346–359. This is not a full-grid projection.

   **Fairness is conditional on the stated prepared-input workload.** FOM nodal/coefficient preparation is excluded, while ROM sampling and initialization remain inside the query. That is defensible as method-specific prepared-input latency, with preparation costs alongside, following S/`DESIGN.md`:478–483. It is not an end-to-end comparison starting from identical physical descriptors or identical sampled inputs.

   **PS exclusion is not justified by “PS has no mesh.”** R/`pscore.py`:3–7 defines both its coefficient resolution and padded collocation grid. A same-resolution convention can be declared—equal coefficient/state resolution or equal collocation resolution. Alternatively, explicitly amend the deliverable to “FD-only same-mesh diagnostic.” The current explanation does not fulfill the brief.

4. **Ladder and solver rules — NEEDS-RESTATEMENT**

   The ladder is substantial, but it is not yet demonstrated to bracket the cheapest matching FOM.

   - The accurate P1 ROMs have medians around 0.2%, while loose settings have worst errors around 5%; P/`report.md`:64–87. D:78–80 omits central FD below \(L=64\), upwind below \(L=128\), and upwind steps larger than 0.005. Those omissions may exclude cheap comparators for loose targets.
   - Add a bounded, predeclared extension rule when a match reaches a coarse-resolution or large-step boundary. Otherwise report the result strictly as the best tested restricted comparator.
   - **Tolerance calibration is a valid conservative numerical-qualification policy**, inherited from S/`DESIGN.md`:366–369, 468–471. It is not unrestricted optimization over tolerances: a looser solve could still meet ROM accuracy while failing the \(10^{-6}n_0\) agreement test. D:22–24 should say “cheapest among calibrated candidates.”
   - Rejecting an entire configuration after production nonconvergence is conservative and correctly inherited. It may hide an available tighter-tolerance comparator unless a fixed recovery rule is registered.

   **PS stability:** the coarse steps are legitimate candidates to test, but finiteness alone is weaker than E0’s stability qualification. R/`refjob.py`:448–452 additionally checks
   \[
   \Delta t\le\frac{0.66(2.8)}{2d\pi N\,u_{\max}}.
   \]
   For \(d=2,N=256,u_{\max}=2\), this is approximately \(2.87\times10^{-4}\), below the ladder minimum \(5\times10^{-4}\). This does **not prove instability**—the criterion is conservative—but the design should not imply that its finite-output rule inherits E0 stability validation. Include a sufficiently small-step anchor or explicitly distinguish the two standards.

   **No NST or schedule bug found in the proposed FD ladder.** R/`refcore.py`:38, 202–217 gives a 1300-stage buffer; the ladder’s longest BE schedule has 400 stages, and its longest CN-R schedule has 102. All listed steps align with output times. Four BE half-steps replace two CN steps correctly. The 2D upwind arithmetic matches `engines.spatial` exactly: R/`refcore.py`:49–54 versus `engines.py`:41–46.

5. **Selection rules — NEEDS-RESTATEMENT**

   **Correct:** the aggregate uncertainty rule in D:42–44, joint matching, complete-stage admissibility, and mesh-specific rematching follow the inherited design. These establish **ties-or-beats under conditional reference uncertainty**, not certified equality of true errors.

   **Problems:**

   - **Top-three screening can miss the fastest full-field candidate.** Restriction, interpolation, and PS synthesis have different costs from lattice output. D:96–98 does not establish that a candidate outside the lattice-cost top three cannot win at 4096².
   - **Cross-mesh populations are not fixed.** D:109 takes each target’s published population. With the P1 fallback, a crossover can therefore compare a 38-case accuracy target at one mesh with a 16-case target at another. A changing population can change the selected FOM and apparent crossover.
   - **A front member needs a cross-mesh identity.** D:150–155 does not specify whether curves track the same configuration or a separately selected front envelope. Missing front members at another mesh need an explicit outcome.
   - **Boundary labels omit the inherited cheapest-member flag.** D:121–122 checks coordinate edges. S/`fom_match.py`:103–106 also flags the cheapest admissible member, which need not occupy those coordinate edges.
   - **Same-mesh “most accurate” means lowest worst error only.** D:117–118 should name that criterion explicitly rather than imply joint accuracy optimality.

   Freeze case IDs, metric/store identities, configuration correspondence, and candidate sets before C2. Use a common cross-mesh population for crossover claims, with larger-population tables alongside.

6. **C2 timing and inference — WRONG as currently specified**

   The common reference A is a reasonable within-job normalization baseline. Burning, warm-up, drift checks, and output identity are also appropriate. However:

   - **Point estimate and bootstrap target differ.** D:145 defines a median over all valid case/repetition ratios. D:147 bootstraps per-case medians. A pooled median generally differs from a median of case medians, even with equal repetition counts. Choose one statistic and recompute that exact statistic in every bootstrap draw.
   - **Joint resampling is unspecified.** ROM and FOM costs must use the same resampled case IDs, with a declared common-valid-case policy. Independently bootstrapping numerator and denominator loses the paired structure.
   - **Comparator selection and confirmation reuse the same timing data.** D:114–115 chooses the minimum measured cost, then D:157–161 uses its interval for decisions. Selecting the observed minimum creates downward selection bias in FOM cost, potentially favoring a false K2 conclusion. Freeze selection on one timing block and infer on another, or specify an inference procedure that includes selection.
   - **Identity veto scope is ambiguous.** D:139–144 and T-id say triplets become invalid. P1 uses a subject-level identity veto, implemented in P/`p1stats.py`:300 onward. A mismatched execution must not simply disappear while the remaining timings support a claim.
   - **Crossover brackets need both endpoint classifications.** D:152–155 must distinguish a demonstrated FOM-fast/ROM-fast bracket from an unresolved lower endpoint, and report reversals rather than imply monotonicity.
   - **K2’s quantifier is narrowed.** D:160 cannot claim the governing K2 from front members alone without proving that omitted linear settings cannot change the conclusion. Unavailable settings must not disappear or make an empty universal test pass.

   H-comp’s lower/upper interval thresholds are otherwise sensible. The finite-ladder upper-bound direction is correct.

7. **Gates and controls — NEEDS-RESTATEMENT, including a WRONG false-match construction**

   | Gate/control | Verdict and finding |
   |---|---|
   | G-FD1/G-FD2 | **CORRECT in purpose.** Port reproduction and vendor upwind equivalence test the advertised implementation relationships. |
   | G-E0 | **CORRECT.** Both dev6 metadata files identify `a0_L2048` as central CN-R, \(L=2048\), \(\Delta t=0.00125\), `ntol=1e-11`, `ltol=1e-9`, 202 stages. See R/`store/2d/dev6/ST/c000.json`:14–50 and `c001.json`:14–50. Compare that named level, not the default finest reference. |
   | G-PS1 | **NEEDS-RESTATEMENT.** The parity gate is appropriate, but the two cases and step are unspecified. Halving a sufficiently small step need not change output by \(10^{-9}\); freeze a discriminating fixture. |
   | G-PS2 | **CORRECT target; make parameters explicit.** R/`store/2d/val32/PS/c000_a0.json`:21–34 gives \(N=1024,\Delta t=3.125\times10^{-5}\). The IC uses **256 panels × 32 Gauss points**, checked against 512 panels: R/`pscore.py`:177–198 and `refjob.py`:422–434. Use matrix lattice synthesis retaining all modes for this gate. |
   | G-INT | **CORRECT in purpose.** Exercise both axes, zero walls, normalization, and the actual production interpolation sizes. |
   | G-CT | **NEEDS-RESTATEMENT.** “Each family, \(N=1024\)” does not specify coverage of \(L<N\), \(L=N\), and \(L>N\), nor the largest output workload. |
   | K-loose | **NEEDS-RESTATEMENT.** One Newton iteration at loose tolerance is not intrinsically nonconverged. Establish independently that the fixed fixture fails convergence or output agreement, and record which predicate rejects it. |
   | K-syn | **NEEDS-RESTATEMENT.** Literal zero timing can cause both fixtures to fail timing admissibility before convergence or accuracy is tested. Use a small positive valid cost and assert the rejection reason. S/`fom_match.py`:70–80 does this with `1e-9`. |
   | K-false | **WRONG as a guaranteed construction.** Its perturbations do not necessarily cross the actual uncertainty-adjusted aggregate thresholds. |
   | K-swap | **NEEDS-RESTATEMENT.** Freeze the case pair and exact time permutation. A changed score is a sensitivity check; it does not establish rejection of inconsistent case/time metadata. |
   | T-dbl/T-fault | **NEEDS-RESTATEMENT.** Require two separately dispatched, synchronized executions. A duplicated expression inside one JIT can be eliminated. Validate the injected timing separation before interpreting its classifier. |
   | T-id | **CORRECT purpose, insufficient scope.** Require the subject-level veto, including all claims using that subject. |
   | G-RR/G-FR | **CORRECT scalar-score reproduction gates**, but equal scalar errors alone do not establish trajectory equality. Preserve the subsequent timed-output identity check. |

   **Why K-false is wrong:** the worst threshold is
   \[
   U_T=\max_i(e_{T,i}+\eta_i).
   \]
   Setting a case \(j\) to \(\max_i e_{T,i}+3\eta_j\) only produces a lower endpoint \(\max_i e_{T,i}+2\eta_j\), which need not exceed \(U_T\). The analogous median construction also fails in general for heterogeneous uncertainties.

   A guaranteed worst-failure fixture can set
   \[
   e'_j=U_T+\eta_j+\delta,\qquad \delta>0,
   \]
   starting from a target clone and choosing an upper-tail case so the median remains unchanged. Independently assert both expected aggregate outcomes before exercising the matcher.

   Also resolve D:163’s “every control … in the cluster smoke” against controls that require a selected real comparator and FRONT-2D, which arrive later.

8. **Resources and dependency handling — NEEDS-RESTATEMENT**

   - **Arithmetic is correct:** D:204–205’s allocations total 11 GPU-hours; adding 25% gives 13.75, approximately 14.
   - **The estimate is not yet a feasibility certificate.** Include all calibration levels, compilation, scoring runs, warm-ups, retries, confirmations, and input preparation in the pilot-derived accounting. The 122 configurations exclude those additional calls.
   - **68.7 GB is bank storage, not total peak.** P/`report.md`:204–207 reports a 113 GB pilot peak and a 96.6 GB process-lifetime peak. Co-resident FOM executables, interpolation/FFT workspaces, retained outputs, and cached ROM tables need a combined memory pilot.
   - **No inherent H200 impossibility is established.** A six-field 4096² output is about 0.81 GB. A shared bank with bounded subject caches may fit; one duplicated bank per subject will not. Specify object lifetimes and measure the actual combined workload.
   - **Defer order is correct.** Dropped same-mesh upwind cells must be labeled deferred, not “no match.”
   - **The bounded FRONT wait is operationally good.** The fallback must be labeled **partial P1-target evidence**, with FRONT-2D C2/H-comp and any unsupported K2 conclusion left outstanding.

**Ranked blocking fixes**

1. **Repair C2 inference:** one estimand, paired case resampling, explicit invalid-case policy, subject-level identity veto, and selection-aware or independent confirmation.
2. **Align comparator screening with full-field cost**, or narrow “best/cheapest” claims to the tested shortlist.
3. **Freeze crossover populations and configuration identities; restore K2’s full quantifier.**
4. **Repair K-false and K-syn**, and register fixed, demonstrably discriminating fixtures for the remaining negative controls.
5. **Resolve same-mesh PS/union scope and the FRONT fallback’s partial-completion status.**
6. **Add explicit ladder-boundary and PS stability/anchor policies**, plus a combined 4096² ROM–FOM memory and cost certificate.

The stored-reference gates and DST normalization do not require redesign. The main blockers concern what the experiment would be entitled to conclude from its measurements.