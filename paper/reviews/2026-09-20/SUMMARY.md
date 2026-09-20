# Review of the compact FOM-comparison manuscript

Review findings for the frozen manuscript at `e6f07fe5`; these are provisional reviewer judgments, not new experimental results. The old reviewer comments were used as examples of scrutiny, not as evidence that old errors persist.

Three subagent reviewers worked independently on mathematical correctness, experimental evidence, and reproducibility. A fourth subagent could not be created because this session caps subagent threads; the coordinator's separate check is explicitly **not independent**, since the coordinator edited the manuscript. No manuscript changes were made during this review.

## Main assessment

The mathematical and experimental reviewers recommend substantial revision; the reproducibility reviewer also considers the paper not yet submission-ready. Both recognize that the earlier heat derivation, ambiguous projection language, unsupported universal speedups and timing disclosures have improved. The experiment reviewer recomputed the historical Burgers tight and relaxed speed ratios from the retained same-job calls and found the arithmetic consistent. The remaining concerns are scientific scope, missing method/configuration details, and evidence for the nonlinear head's contribution.

## Prioritized issues

### 1. Secondary success criterion is labelled the full preregistered criterion

**Must fix — verified overstatement.** Reproducibility review, independently checked by coordinator: main.tex:601; b-seeds DESIGN.md:188–200,415–432; PDF Table D.1.

**Action:** Replace full criterion with secondary knob criterion; define it and explicitly preserve failed sealed-ratio and universal-convergence checks.

**Work required:** Precise prose correction; no measured value changes.

### 2. Method scope and missing time-dependent equations

**Must fix.** Math review: PDF pp.3,14–17; wave curvature/RK4 arm differs from the general fully discrete residual; corrected heat equations absent.

**Action:** Add actual wave and corrected-heat formulas; qualify the general method and heat endpoint; specify the block-damped update.

**Work required:** Source audit and concise writing; no new benchmark implied.

### 3. Heat2D measurement provenance and reference convention

**Must fix — verified documentation mismatch.** T01_problems.tex:10 and T01b_spec.tex:11 name3511417; rewrite-provenance.json Heat rows name3529772.

**Action:** Regenerate configuration rows from the displayed run, separate lineage/measurement identifiers and label refined-reference error.

**Work required:** Source-driven table/caption correction.

### 4. Missing experimental definitions and stale coverage promises

**Must fix.** Experiment review: PDF pp.5–7,16–17; no complete 3D family/cohort/configuration table; NS2D and low-viscosity outcomes announced but not shown.

**Action:** Generate a compact configuration table from each displayed source and narrow the setup to supported cases.

**Work required:** Evidence integration and writing.

### 5. Historical Burgers comparison must identify its method/configuration

**Must fix.** Experiment reviewer independently verifies tight/relaxed ratios from original paired calls; PDF p.6 lower Table2 lacks full older-ROM configuration.

**Action:** Keep both named comparisons and their unequal-error/stall caveats; add the old checkpoint, q/M/m, and stopping-contract distinction.

**Work required:** Caption/source annotation, not replacement of measured numbers.

### 6. Nonlinear-head contribution exceeds what FOM-only evidence isolates

**Must fix.** Math and experiment reviews agree: reduction versus FOM does not identify benefit over linear enrichment inside the same bank.

**Action:** Keep FOM-only external tables; narrow the contribution, or choose a small same-bank internal ablation if demonstrating nonlinear necessity is essential.

**Work required:** Claim revision now; optional additional evidence requires a separate decision.

### 7. Fixed-rank causal study and multi-seed scheduled study support different claims

**Must fix.** Experiment review: main Table5 fixes M on one development checkpoint; sealed Table D.1 changes M with q.

**Action:** Define success rules and keep the two claims separate. Strong generalization of rank-only tuning needs prospectively frozen fixed-M replication.

**Work required:** Writing now; new run only if stronger claim is retained.

### 8. Timing uncertainty and comparator scope

**Important.** Experiment review: near-break-even ratios lack uncertainty; named CG is not the fastest applicable FOM guarantee.

**Action:** Add compact timing dispersion and case/repeat counts, identify tested CG/preconditioner controls, and limit speed claims to those baselines.

**Work required:** Generate uncertainty from retained repetitions where available; disclose unavailable counts.

### 9. Rank, skip and EQ guarantees need qualification

**Important.** Math review: QR rank assumptions omitted, linear skip does not guarantee descent, finite-state EQ check is not a global error certificate.

**Action:** State rank tests/fallbacks and empirical validation scope. Describe conditioning as motivation, not guarantee.

**Work required:** Method clarification from implementation.

### 10. Portable evidence and geometry/input dependencies

**Important.** Reproducibility review: absolute-path build, no artifact entry point, missing L-shape factor/tests and source/parameter arrow in diagram.

**Action:** Prepare relative-path evidence/build instructions; add concise geometry details and input dependency; publishing artifact needs separate authorization.

**Work required:** Packaging and compact method/diagram fixes.

## Scope-compatible revision

Keep the named-FOM external comparisons and compact appendix. Replace redundant prose with the actual wave/corrected-heat equations, a small source-generated configuration table, rank/stopping definitions, and timing uncertainty. Clarify the earlier Burgers configuration without changing its recorded values. A small same-bank ablation is an optional route to a stronger architectural claim; otherwise state explicitly that nonlinear necessity is not established. No reviewer request alone authorizes new experiments or changes the user's comparison preference.

The newly completed experimental lanes and the snapshots already printed in the paper are distinct. Integrating accepted final results is a separate source-controlled update; pending retention is not acceptance.

## Review files

- [Mathematics review](mathematics.md)
- [Experimental review](experiments.md)
- [Reproducibility review](reproducibility.md)
- [Coordinator format/provenance check](coordinator-check.json), not an independent review
- [Frozen source hashes](manifest.json)
- [Machine-readable issue list](issues.json)

## Glossary

**FOM:** full-order numerical solver used as a comparator. **NM-ROM:** nonlinear-manifold reduced-order model. **CG:** conjugate-gradient solver. **POD:** proper orthogonal decomposition, a way to construct a linear reduced space. **EQ:** empirical quadrature, a fitted rule for evaluating projected residual terms at selected points. **Correction rank:** number of added directions within the learned spatial bank. **Fixed test space:** unchanged residual projection modes across the comparison. **Scheduled study:** a study that changes test count along with correction rank. **Development cohort:** data available during model or setting selection. **Sealed/final cohort:** cases evaluated after choices are frozen. **Stationarity:** satisfying a numerical first-order stopping criterion; it does not prove global optimality. **Same-grid error:** error relative to the reference on the same discretization. **Refined-reference error:** comparison that also exposes discretization error. **Paired timing:** method and comparator measured within the same allocation. **Rank condition:** requirement that projected correction columns remain linearly independent. **Internal ablation:** changing a component of the same method to isolate its effect. **Weak reject:** a simulated reviewer assessment of the current manuscript, not a conference decision.
