Read-only audit completed; no files changed or jobs submitted. Numbering follows round 2’s eight required fixes.

1. **NEEDS-RESTATEMENT — Chronology and frozen specification (A3.1).**  
   The chronology correction, diagnostic-guided selection disclosure, v4a name, corner and exclusion threshold address the substantive findings. However, `runs/descriptors.json` contains hashes and metadata, **not descriptor arrays**. Its acceptance rates are estimates from separate 20,000-centre probes, not observed acceptance rates while generating each cohort. Correct those two descriptions.

2. **CORRECT — Boundary interaction and family definition (A3.2).**  
   The common-IC box evolution separates boundary effects from initialization changes; conditional support and the narrow family’s changed location are disclosed. Validation plus untouched test confirmation, with no redesign, is appropriate. The supplied smoke4 interaction JSON establishes only the combined \(D_{\rm geo}\) contrast, as A3 correctly acknowledges; it does **not** establish the new \(D_{\rm bnd}\) gate.

3. **CORRECT — Reference eligibility (A3.3).**  
   Requiring all 32 validation/confirmation cases to qualify, with \(U\le\tau/5\), removes wins supported by unjustified last-difference allowances. The remaining reference estimates are explicitly conditional.

4. **WRONG — Decision table, attribution and selection (A3.4).**  
   The table is not yet exhaustive or internally consistent:
   - Three-way accuracy matching includes **unresolved** cases; these are not assigned an explicit branch distinct from genuine censoring, missing runs or qualification failure.
   - “Every competitor censored → candidate win, \(S=\infty\)” supplies neither a matched timing ratio nor a competitor finalist that can satisfy the stated confirmation rule.
   - An unqualified test reference becomes “not confirmed,” then a cell **loss**, conflating unavailable evidence with measured failure.
   - Attribution has an uncovered case: validation satisfies the separation threshold, but valid test results give \(S_{\rm box,test}\ge S_{\rm geom,test}/2\). This must withdraw attribution, especially when the test speed-ups are equal.
   - Calling the final outcome UNRESOLVED is useful scientifically, but the brief’s **“Otherwise withdraw the geometry claim”** still applies. UNRESOLVED must not preserve that claim.
   - Bank selection refers to the “best competitor,” whose identity depends on the selected bank. The paired-ratio selection needs an explicit optimization order and deterministic ties.

   **Concrete fix:** enumerate missing/unresolved/censored/invalid outcomes; require an actual matched competitor for a speed win; preserve reference failure as unresolved evidence; withdraw attribution on either valid cohort meeting the “same S” threshold; explicitly withdraw unsupported claims regardless of evidence label. Define selection as a deterministic max–min over the measured finalist-pair ratios, or another explicit frozen rule, including single-finalist cases.

   Requiring both levels to win and using the factor-two attribution separation are legitimate conservative lane choices.

5. **WRONG — Qualification certificates (A3.5).**  
   Competitor asymmetry is **not itself a blocker**: admitting additional valid, accuracy-matched competitors can only reduce the minimum speed-up for a fixed bank. COMMON-BRIEF §3 defines numerical qualification; it does not require every benchmark competitor to receive that label. Report failed Q1/Q2 honestly and distinguish benchmark eligibility from numerical qualification. Calling this a relaxation of the qualification definition is unnecessary.

   The bank certificates remain incomplete:
   - Correct ETDRK4 coefficients plus finite, bounded stages do not certify stage equations or the final update. A wrong stage argument or omitted nonlinear term can pass.
   - Allowing up to 1% non-finite stages conflicts with the binding **every run valid** requirement.
   - Linearised CN divides by \(\|R\|\) without defining the zero-residual case.
   - Q2’s “\(10^{-12}\max\)” leaves the population defining that maximum unspecified.

   **Concrete fix:** independently recompute normalized defects for every ETDRK4 stage and final update, with frozen tolerances and a stated explicit-stepper analogue of the tight/tighter check. Make any non-finite stage invalidate the run; define zero-safe CN normalization and the exact Q2 maximum/zero-state policy.

6. **WRONG — Detectable controls (A3.6).**  
   The supplied v4a geometry controls pass; L’s observed FD orders are approximately 1.99946 and 1.99987. That resolves the previous alignment demonstration failure. The Q1 fault now requires an independently established violation. However:
   - MF-Q2 still says detect failure **when** the fault exceeds the bar, without requiring that it actually does.
   - A perturbation of size \(2\tau\) alone does not establish that the resulting error exceeds the matcher’s uncertainty-adjusted rejection threshold; direction, normalization and existing error matter.

   **Concrete fix:** require independent recomputation proving \(e_{\rm fault}-U>\tau\) for MF-MATCH and \(\rho_{\rm fault}>0.116\) for MF-Q2 before testing detection. Require at least one demonstrated violating fixture for each. Final v4a fault-control evidence remains a release prerequisite.

7. **CORRECT — Search coverage and finer lattice (A3.7).**  
   Removing the size-stop heuristic fixes the coverage claim. Every matched method now receives the finer-lattice diagnostic, with FD interpolation explicitly identified.

8. **NEEDS-RESTATEMENT — Full-size cost certificate (A3.8).**  
   The release prerequisite is appropriately stated, but remains outstanding. Production-size smoke5 probe JSONs exist; `runs/cost/projection.json` does not. Probe timings alone do not establish total ladder cost, peak memory across required configurations, or fit within the remaining budget. This is a **production-release blocker**, not grounds to reject the prerequisite’s design.

**OVERALL: NOT-READY**

Blocking fixes only:

1. Complete A3.4’s evidence branches, confirmation, attribution withdrawal and deterministic finalist selection; align claim withdrawal with the binding brief.
2. Complete the bank’s ETDRK4 consistency certificate and invalid-run, CN normalization and Q2 denominator policies.
3. Require independently demonstrated violations for the remaining conditional fault controls, then verify the final v4a controls before release.
4. Supply and audit the full production time/memory projection against remaining budget before production release.