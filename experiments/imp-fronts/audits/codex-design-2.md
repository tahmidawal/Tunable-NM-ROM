**Overall: WRONG — Amendment 1 fixes several findings, but the design is not ready for production.** Blocking issues remain in qualification, adaptive-policy termination, cost certification and smoke coverage.

I made no file changes, submitted no jobs, created no scratch files, and accessed only allowed dev6/val32 references. The disclosed debug outputs remain exploratory; rerunning them does not restore pre-data status.

1. **A1 — NEEDS-RESTATEMENT.** Empirical POD labels, clipping, SVD tolerance and the explicitly unmatched 256-node sensitivity resolve the earlier ambiguities. However, fitting a case’s own upwind fields guarantees a small upwind floor, **not** \(D\ge10\): reference fields could occupy the same span. The reverse construction has the same problem, especially after clipping. **Fix:** retain these as empirical self-fit checks; add synthetic orthogonal components with analytically prescribed, nonzero floors giving the required \(D\) values at both ranks.

2. **A2, PDE compatibility — CORRECT.** The positive-time, smooth-wall compatibility argument remains valid.

3. **A2, “analytic curvature” — NEEDS-RESTATEMENT.** Differencing an analytic gradient is still a finite-difference approximation to curvature. A fixed \(\delta=10^{-4}\), a polynomial fixture and low-frequency sines do not bound its error for the learned bank. **Fix:** call it a gradient-difference estimate; compare \(\delta,\delta/2,\delta/4\), or evaluate analytic Hessians. Freeze the random-sine fixture, normalization and convergence tolerance.

4. **A2, wall-share controls — CORRECT under the literal \(h_\pm\)-only definition.** I computed DST-I high-pass fields on the specified 257² lattice, retaining the first 4096 Laplacian-ordered modes and using the eight interior-node wall band:
   
   | Control | Median \(\omega\), dev6+val32 | Required | Outcome |
   |---|---:|---:|---|
   | \(h_+\) alone | 0.81303544 | ≥0.5 | 38/38 pass |
   | \(h_-\) alone | 0.04688363 | ≤0.25 | 38/38 pass |
   
   These are essentially shape tests: reference amplitude cancels. **They do not demonstrate separation after adding the perturbations to real references.** For \(u_{\rm ref}+h_-\), median \(\omega=0.69214567\), which fails the negative-control bar. Also, the pure Gaussian’s relative high-pass norm is only \(6.15\times10^{-11}\); compute retained tail coefficients directly, not by subtracting nearly equal total energies.

5. **A2, arm-(c) decision — NEEDS-RESTATEMENT.** The heuristic label, removal of the impossibility claim, and RUN/DEFER/INCONCLUSIVE branches fix the substantive earlier finding. However, \(\omega\) remains undefined for exactly zero tail energy; clipping \(\tau\) does not define this energy ratio. **Fix:** freeze a relative tail-energy floor and return an explicitly defined unavailable/INCONCLUSIVE result below it. State explicitly that controls use \(h_\pm\), not \(u_{\rm ref}+h_\pm\).

6. **A3 — CORRECT for the actual registered references.** Primary population and empirical \(K=16\) labels are fixed. I checked the fixture’s potential cancellation: the largest reference tail beyond rank 9216 is \(8.704\times10^{-5}\), so the triangle inequality gives a fixture residual of at least \(0.004913\), comfortably exceeding 0.001. Freeze the fixture seed in implementation. The deferred timing label is appropriate.

7. **A4 — CORRECT.** Duplicate-column enrichment, rank-revealing orthogonalization and the numerical strict-improvement tolerance supply the missing null control. The high-frequency hybrid is correctly relegated to an empirical comparator. Carry the disclosed prior inspection of A4 results into reporting.

8. **B0, copied rules — CORRECT.** The binding factors, AND/OR logic and strict safety guard remain intact.

9. **B0, operational qualification — WRONG.** The finite-control test prevents the previous infinity success. But assigning failed arm cases infinity can still produce a finite, passing median despite unqualified trajectories, contrary to the common brief’s configuration-level “every run valid/every-case distance” contract. Arm-(c)’s tail route explicitly requires only finite valid runs, not Q1+Q2, and lacks complete missing-control semantics. **Fix:** distinguish diagnostic infinity-penalized medians from qualified success; require the applicable full-population certificate before success. Define finite, available control prerequisites for both arm-(c) comparisons and a complete outcome/guard precedence table.

10. **B1, K1 reuse and seed claim — CORRECT.** Reuse remains legitimate; “passes against all three incumbent retrains” correctly replaces intervention seed robustness.

11. **B1(a), fidelity adapter — CORRECT at design level.** Output cadence, mode convention, synthesis grid, initial-state convention, both data streams and replacement fidelity gates are now specified. Code audit must verify that the adapter actually changes every generator call.

12. **B1(b), coverage adapter — CORRECT.** Both populations, unchanged state picks and all-state finite/residual checks address the earlier omission.

13. **B1(c), architecture — NEEDS-RESTATEMENT.** The causal limitation and independent symmetry tests are fixed. Initialization, normalization, optimizer masking and rotation are merely promised to be frozen later. **Fix:** register their actual choices before implementation; define the relative derivative-test denominator and zero-scale behavior.

14. **B2, identity/evaluator controls — CORRECT as control requirements.** Deterministic mismatch rejection replaces unjustified decode degradation; exceptional-input tests are included. Their expected outcomes still depend on repairing B0.

15. **B2, fidelity/coverage controls — CORRECT.** All-dev6 aggregate fidelity and complete-table coverage checks fix the sample-size errors.

16. **B2, freshness — CORRECT at design level.** Explicit descriptor allow-lists, training/arm comparisons and within/across-split checks repair the unsafe traversal. The existing \([{\rm IMPF},g,j]\) seeds distinguish the registered splits. Implement comparisons without discovering files outside the allow-list.

17. **B2, Q1 certificate — NEEDS-RESTATEMENT.** A seeded \(10^{-3}\) coefficient perturbation does not guarantee \(\gamma>10^{-3}\); the seed/direction remains unspecified. More seriously, “prescribed subset” still does not identify fresh-cohort tighter anchors; the inherited default is `dev6`, absent from fresh populations. **Fix:** freeze nonempty fresh anchor IDs and a reproducible perturbation fixture demonstrated to fail before production.

18. **B3 — NEEDS-RESTATEMENT.** Counts, seed/index mapping, draw order, deterministic subsets and reference-name allow-list fix the original split defect. But dev/val are declared reference-free while B9 builds FC2 dev case 0’s reference. **Fix:** explicitly permit and budget that development reference; keep its outcome separate from test scoring.

19. **B4 — CORRECT semantically.** Incorporating the actual selector fixes successor mapping, target certification, undetermined evidence and the false lower-bound claim. Preserve those semantics when adapting its incumbent-specific bank registry and attempt-name assertions.

20. **B5 — WRONG.** Factors and final-state Q2 rechecking are repaired, but the state machine is still incomplete:
   - No transition exists for missing/diagnostic initial selection, invalid runs or unavailable comparisons.
   - Four rounds cannot exhaust the stated ladder plus step refinements; even starting at gauss160 can require more.
   - Factor 0.25 has no finer check, so cannot establish step convergence; its qualification label remains ambiguous.
   
   **Fix:** enumerate every state and exceptional transition, including round-cap exhaustion; require valid comparison solves; define terminal numerical and step-convergence labels separately. Either fund the complete path or preregister an explicit budget-limited terminal failure.

21. **B6 — CORRECT.** Rank 384 remains primary; rank 512 remains supplementary.

22. **B7 — NEEDS-RESTATEMENT.** Descriptive status, absolute target, percentile convention and direct-routing distinction are fixed. But an in-range accuracy-matched FOM timing does not establish fallback cost for flagged steep cases. B8 also removes K0 scoring while the original B7 still requests it. **Fix:** explicitly score E9 on K1; tune and pair fallback timings on the relevant flagged families at the declared target, including adaptive-policy overhead and burn-in.

23. **B8 — NEEDS-RESTATEMENT.** Arithmetic is correct: **14.7 GPU-h**. S2 genuinely saves six references plus timing; S3 removes a retrain. However, no complete solve inventory certifies 0.6 GPU-h for policy rounds, tight/tighter anchors and successors. Arm-(c) substitution has no complete costed schedule, and S2 drops a required E9 deliverable. **Fix:** cost the repaired B5 worst-case paths and development reference; freeze a complete arm-(c) replacement schedule or defer it. Label E9 incomplete under S2, and count spent smoke separately from remaining work.

24. **B9 — WRONG.** Both generators and full training data are improvements. Nevertheless, fib4181 is not guaranteed to fail on the two smoke cases; “round 1” cannot exercise every transition or terminal state. C-B2–C-B5 execution is not fully scheduled, and one FC2 reference does not bound all refinement paths. **Fix:** enumerate every control and expensive phase, use deterministic injected branch failures alongside real solves, exercise terminal/unavailable states, and certify reference paths and smoke cost explicitly.

| Audit item | Re-audit verdict |
|---|---|
| A1 | NEEDS-RESTATEMENT |
| A2 compatibility | CORRECT |
| A2 curvature | NEEDS-RESTATEMENT |
| A2 literal wall-share controls | CORRECT |
| A2 decision/numerical edge cases | NEEDS-RESTATEMENT |
| A3; A4 | CORRECT |
| B0 copied rules | CORRECT |
| B0 qualification | WRONG |
| B1 reuse; seed wording; adapters (a)/(b) | CORRECT |
| B1(c) | NEEDS-RESTATEMENT |
| B2 identity; fidelity/coverage; freshness | CORRECT |
| B2 Q1 fixture/anchors | NEEDS-RESTATEMENT |
| B3 | NEEDS-RESTATEMENT |
| B4 | CORRECT |
| B5 | WRONG |
| B6 | CORRECT |
| B7; B8 | NEEDS-RESTATEMENT |
| B9 | WRONG |