1. **PARTIALLY — Plateau validation (A1.1).** Independent anchors and global selection fix the automatic endpoint “plateau.” The requested tail-stability check remains absent. Require agreement of multiple terminal rungs with the anchors; otherwise withhold plateau-dependent claims. This blocker remains open.

2. **RESOLVED — Query contracts (A1.2).** The primary lattice-output contract aligns products and specifies preparation, projection and caching. Separate full-field reporting and the “operational estimate” label address the remaining timing distinctions.

3. **RESOLVED — Selection and matching (A1.3).** Selection now uses validated ST/global thresholds, explicit ties and empty sets, and matches the FOM against the selected ROM’s achieved errors. Provisional timing is labelled.

4. **PARTIALLY — Failure semantics (A1.4).** Fixed case sets, invalid-rung suffix breaks and whole-cohort FOM convergence resolve aggregation ambiguity. However, both inherited solvers can return reason **2**, “tiny step”; A1 omits it from non-convergence. Count reasons 0 and 2 toward the limit, with explicit 2D/3D mappings.

5. **PARTIALLY — Reference uncertainty (A1.5).** Envelopes and multiple-crossing reporting help, but “± … on both sides” is ambiguous. Specify independent adverse endpoints: compare the left upper bound with the right lower bound, and vice versa; shifting both sides together can conceal uncertainty.

6. **RESOLVED — Gates and controls (A1.6).** Unexplained parity failures now withhold results, empirical controls are distinguished from integrity checks, and explicit fault injections restore scoring/integrity validation.

7. **PARTIALLY — Timing noise (A1.7).** Paired summaries, variability and all-block drift checks improve the protocol. Confirmation is measured, but the selected minimum’s main-panel speed-up remains primary without a disagreement rule. Use confirmation for the primary selected-pair estimate or make disagreement render that claim provisional.

8. **PARTIALLY — Budget (A1.8).** Precomputed rules, deterministic mesh priority and reservations for outstanding jobs resolve important gaps. The component ranges still lack per-job totals covering both contracts, compilation/warmup and confirmation, and bounded pilot limits. Add those totals and limits before submission.

9. **RESOLVED — Operator authorization (A1.9).** The documented JCP campaign authorizes inherited off-mesh implementations, and A1 explicitly identifies continuum weak advection and its departure from the deployed upwind prescription.

**New major introduced by A1.9:** It attributes the reduced-solution discrepancy to the stencil’s \(O(h)\) consistency gap. Operator consistency alone does not establish an \(O(h)\) discrepancy between these computed trajectories: projection, finite test spaces, residual weighting, time stepping and solver termination also matter. **Fix:** describe an advection-operator consistency difference; withhold a solution-error rate or causal attribution unless separately demonstrated.

No new blocker identified; the original plateau-validation blocker remains.

Verdict: **NOT-READY**