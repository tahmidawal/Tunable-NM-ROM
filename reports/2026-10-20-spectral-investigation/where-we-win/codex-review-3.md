Read-only review completed; no files modified, including the lab log.

Verification passed: in-memory execution of `make_report.py` reproduces **report.md and all 157 registered values byte-for-byte**. All **48 PCG cases** reproduce their iteration counts and convergence flags. L-shape and penalisation spot checks reproduce. The eight-hole scenario gives **2.23748 / 18.31476**, and the corrected partial-cost crossovers are **85,194 / 6,480 queries**.

Most round-2 fixes are present: flagship 2’s prediction is withdrawn; setup, flux and held-out checks are included; literature figures are registered; per-mesh amortisation arithmetic is repaired; coefficient contrast is normalized. Remaining findings follow.

1. **MAJOR — The summary still promotes incomplete scenarios into evidence for the ranking.**

   **Evidence:** [report.md:11](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-spectral-investigation/where-we-win/report.md:11) headlines the geometry speed-ups, and the ranked table repeats their endpoints. Yet section 3 correctly says these compare different delivered outputs, omit iteration/timestep sensitivity, and require common-contract measurements **before any payoff estimate**. Section 7.1 nevertheless calls them a “Prediction.” Likewise, line 13 still interprets the synthetic PCG comparison as evidence that warm-started FOM trajectories lose little, although section 3 explicitly disclaims that interpretation.

   **Fix:** Carry those limitations into the summary and ranked table. Label geometry numbers “arithmetic illustration with mismatched outputs; no performance forecast.” Replace the warm-start inference with “trajectory cost unknown.” Describe the ordering as a subjective research priority, rather than payoff-times-plausibility supported by these numbers.

2. **MAJOR — The execution gates remain logically inconsistent.**

   **Evidence:** [report.md:221](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-spectral-investigation/where-we-win/report.md:221) authorizes ROM training when **some FOM** costs twice the box ROM; line 224 stops only when **no FOM** does. One deliberately slow comparator can therefore release training despite a fast alternative.

   Flagship 1 also has conflicting objectives: its summary requires beating the tested **FOM** union, while line 205 requires beating every tested competitor—including POD-ECSW. Line 204 anticipates losing to POD and continuing with a mesh-free geometry story.

   **Fix:** Gate flagship 3 on the **fastest accuracy-admissible tested FOM**, under a common output contract. Explicitly separate FOM-superiority, POD-superiority and methodological research objectives for flagship 1, with consistent continuation rules. Spell out inclusion of `(a')` and `(a'')` in flagship 3’s final union.

3. **MINOR — The constant-coefficient control is numerically mislabelled.**

   **Evidence:** The PCG table reports contrast **1**, requiring **2 iterations**. `analysis/varcoef_pcg.py` actually runs **1.0001**; `values.py` rounds the key and calls it constant coefficient. My rerun gives **one iteration at exactly 1**, versus two at 1.0001.

   **Fix:** Either label the existing row “1.0001, nearly constant,” or run and register the exact constant-coefficient control. This does not affect the contrast-100 conclusions.

4. **MINOR — The glossary contains a mathematical misdefinition.**

   **Evidence:** [report.md:273](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-spectral-investigation/where-we-win/report.md:273) defines Galerkin as “as many tests as unknowns.” Equal dimensions alone do not make a method Galerkin; that description also admits square Petrov–Galerkin methods.

   **Fix:** Define Galerkin through its relationship between trial and test spaces, and distinguish it from least-squares minimisation of the tested residual. Avoid presenting a dimension count as the defining property.

5. **MINOR — Report-rule compliance and review closure are still overstated.**

   **Evidence:** The title, status, generated numerical registry, Mermaid diagram and expanded glossary pass. However, mathematical expressions such as `N = 24–48` and `r ≤ 512` remain outside LaTeX. The literal dated-filename rule is still unmet, although retaining the user-specified review target is reasonable. The review record already announces three completed rounds and points to `codex-review-3.md`, which was absent during this review.

   **Fix:** Finish the notation cleanup; document the filename exception explicitly; mark round 3 pending until its review is actually saved. Update dispositions to reflect the remaining summary and decision-rule issues.

The competitor coverage is now substantially stronger. Continuous-subspace geometry handling is supported by [LiCROM](https://arxiv.org/abs/2310.15907), and the added exponential-action alternatives are appropriate given the problem-dependent performance documented by [Deka, Tokman and Einkemmer](https://arxiv.org/abs/2211.08948). Optional Fourier continuation is acceptable only with the report’s explicit restriction to tested competitors; the remaining concern is enforcing the comparison set consistently.

**Verdict: ACCEPT-WITH-FIXES — as an exploratory research-direction report, not a validated forecast of wins.**

The top three are defensible research directions, but their exact ranking remains a judgment about potential scientific value, not an evidence-based ordering of likely speed-ups. The execution order—variable-coefficient FOM-only benchmark, then a matched-output 2D geometry pilot, then conditional 3D—is defensible and economical once the gates above are corrected. The geometry pilot should establish competitiveness against locally refined FOMs and continuous/POD alternatives before expensive geometry-conditioned training or 3D expansion.