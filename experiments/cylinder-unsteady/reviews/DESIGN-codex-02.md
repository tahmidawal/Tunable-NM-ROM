**DESIGN: RELEASE WITH CHANGES.** Freeze the clarifications below before R1b; preserve every existing tolerance and failure.

**CODE: NOT READY.** Before R1b, fix the gate evaluator, enforce complete comparison windows, and verify the common-state embedding. The trained-bank projection bug must be fixed before R3/R4.

Read-only review completed; no files, Git state, jobs or installations were changed. References below use `D` = coordinator `DESIGN-G5.md`, `REF` = reference experiment directory, and `ROM` = ROM experiment directory.

1. **Blocker — R1b can falsely pass without mandatory evidence.**  
   [`REF/r1_gates.py:25`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-cylinder-unsteady-ref/experiments/cylinder-unsteady-ref/r1_gates.py:25) accepts missing ladder comparisons. Missing order estimates leave `order_ok=True`; startup results are merely listed, never incorporated into `R1b_time_gate`. Startup denominator 800 is not loaded at all. Thus one passing comparison can release a run without an order or startup check. Space results likewise lack an aggregate admission decision.  
   **Required:** fail closed on missing/nonfinite evidence; require the selected-dt startup check; compute explicit combined R1b/R1c decisions. Freeze which three-level stencil qualifies each candidate, especially 1/3200, and whether a coarse non-asymptotic stencil disqualifies finer candidates. The hardcoded metric floor `1e-12` at line 14 needs justification: an algebraic residual tolerance does not establish a field/force noise floor.

2. **Major — a partial comparison window can satisfy the reference gate.**  
   [`REF/compare.py:79`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-cylinder-unsteady-ref/experiments/cylinder-unsteady-ref/compare.py:79) requires only two common snapshots. It does not require completed runs, coverage through \(S^*+0.35\), the expected save grid, or complete eligible force times. Interrupted runs could yield small, plausible maxima over a shorter interval.  
   **Required:** validate status, hashes, initial-state identity, endpoints and expected cadence before calculating gate metrics.

3. **Major — the one-period certificate does not establish full-horizon reference accuracy.**  
   `D:163–171` fixes \(W=0.35\) s, while `D:328–338` ranks phase-sensitive errors over 1.7 s. A short-window lift/velocity difference can accumulate substantially over five periods. The parent design §7 requires reference uncertainty below one fifth of the comparison tolerance.  
   **Required:** retain W unchanged, but predeclare a full-horizon uncertainty check or a justified propagation criterion; mark comparisons unresolved when reference uncertainty is too large. Also specify whether R1d compares every complete cycle or an aggregate—currently “statistics over complete cycles” leaves that choice open.

4. **Blocker for R4 — trained-bank projections remain frozen-bank projections.**  
   [`ROM/rom_export.py:278`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-cylinder-unsteady-rom/experiments/cylinder-unsteady-rom/rom_export.py:278) exports reference inner products only against FE functions plus the frozen four-mode bank. `rom_r4.py:54–57` replaces bank evaluations but does not regenerate those products. `rom_core.py:185–188` uses the same column offsets for every bank.  
   **Required:** export separately identified, hash-bound projections for the trained bank and select them by bank identity. An in-memory check reproduced an out-of-bounds access at bank column 4; therefore every registered retraining rank ≥8 fails. A four-mode replacement could instead silently use incorrect products.

5. **Major — initial ROM pressure is diagnostic but enters primary errors.**  
   [`ROM/rom_core.py:158`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-cylinder-unsteady-rom/experiments/cylinder-unsteady-rom/rom_core.py:158) estimates initial pressure by least squares without the inertial term or the differentiated continuity constraint. `rom_r4.py:80,96–97` includes that pressure in the reported error series. In unsteady flow this is not generally the consistent initial pressure.  
   **Required:** compute consistent pressure and acceleration from the constrained reduced equations, and report the initial continuity defect. Freeze any alternative treatment of the initial pressure sample before results.

6. **Major — R5 does not fully implement the parent fairness contract.**  
   `D:317–324` compares a 200-step FE segment with a full-horizon ROM. Normalizing by simulated duration does not remove differing startup, factorization-refresh and output overheads. The release also omits the parent’s explicit accuracy-matched FOM ladder and same-invocation accuracy/checksum/work witnesses.  
   **Required:** freeze equivalent output workloads and initialization/cache accounting; require seven retained paired repetition arrays, immediate CPU warm-up, result identities and work counts. Either test the fastest qualifying FOM settings or limit the claim explicitly to speed versus the production reference configuration. No R5 timing implementation was present among the supplied files.

7. **Major — quadrature fairness is shared, but quadrature accuracy is unverified.**  
   `D:375` justifies order 12 using polynomial degree. Curved mappings and neural modes prevent that argument from establishing exact integration. Shared quadrature can still bias bank and POD differently.  
   **Required:** predeclare a higher-order quadrature comparison with numerical acceptance criteria for projections, reduced operators and force functionals before ranking either method.

8. **Major — “exact embedding” needs a numerical witness.**  
   [`REF/transient.py:373`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-cylinder-unsteady-ref/experiments/cylinder-unsteady-ref/transient.py:373) uses `GridFunction.Set(..., bonus_intorder=6)`, rather than a demonstrated coefficient injection. Current code has the bonus-order argument missing from the earlier Grok-reviewed version, but that alone does not prove exactness on curved geometry.  
   **Required:** verify velocity, pressure and essential traces before/after embedding against a fixed numerical bound, and ensure both R1c arms start from the original P3 S*. A settled P4 state cannot be loaded into P3 by the current loader.

9. **Major — scientific retention remains incomplete.**  
   [`REF/cluster/collect.sh:13`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-cylinder-unsteady-ref/experiments/cylinder-unsteady-ref/cluster/collect.sh:13) deliberately leaves snapshot chunks remotely, retaining only their hashes locally. Hashes cannot reconstruct arrays or support R1f after remote deletion.  
   **Required:** establish durable, checksum-verified snapshot retention before R1f acceptance or cluster cleanup. Freeze an exact accounting ceiling and extension trigger: A3’s “about 23 GPU-h” is insufficient as a hard stop rule.

10. **Minor — reference extraction and phase alignment need precise definitions.**  
    `sources/extract_featflow.py:24–39` uses sampled extrema from the first complete cycle; `REF/cycles.py:22–55` refines extrema parabolically. Applying the latter to the finest published logs changes \(C_{L,\max}\) by approximately \(1.75\times10^{-5}\), below the gate tolerance but contrary to “same code.” The source JSON also labels the 1996 ranges as requiring verification. `ROM/rom_r4.py:105–114` compares different overlap lengths for different phase lags.  
    **Required:** use one frozen extraction procedure, verify the historical ranges, and define a common phase-search support.

The core design is otherwise credible: the split isolates held-out Re105; the floor and R3 rank rules are deterministic; bank/POD share lift, pressure basis, supremizers and rollout scheme. I found no sign/indexing error in the inspected convection tensors, lift terms, divergence coupling, inertia-inclusive reaction force, or saved-time-to-rollout mapping. Pressure construction removes each snapshot’s spatial mean as the parent A4 rule requires. Training value/gradient, supervised-divergence and traction objectives match the stated terms; the SciPy optimizer substitution should be explicitly acknowledged in the release.

A1/A2 are documented post-smoke method development, **not untouched preregistration**, but preserve failures and numerical gates. A3 follows the declared stop after P3 divergence failure; it does not relax that gate. Saved evidence shows P3 final divergence \(0.002922\), versus \(0.0003159\) after P4 settling. That supports investigating P4, not admitting R1e yet. Making the 1996 lift interval non-gating before project results is defensible against the pinned modern reference; it should remain unchanged.