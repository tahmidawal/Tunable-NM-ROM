# B8 frozen-parent span discriminator

Prospective scope: one allocated GPU for at most 30 minutes; driver soft boundary 1,740 seconds between case/layout/time rows. An individual row may exceed the soft boundary; scheduler walltime remains the hard limit. Incremental completed rows and their hashed code payloads survive interruption. No final-test fields, training, model changes or online refinement.

Entry point: `analyze_residual_span.py --out OUTPUT --model-job COMPLETED_MULTIDEPTH_JOB`. Default time index0. Optional `--time-indices 0,5,10` is the same finite protocol, subject to the same cap; do not infer that uncompleted rows passed. No optional training-case sample is currently implemented.

Primary development cases are dev000, dev003, dev013 and align004, plus all eight independently seeded expanded development trajectories. This is development evidence, not untouched final-test evidence. All inputs and fitted parent initializations are hash checked against the selected model's provenance. Expanded references retain their existing qualification status; this analysis cannot admit them independently.

For each case, use coarse4, uniform20 and initial-field prescribed20 layouts, the same frozen feature/head checkpoint and the same parent weights AND parent code. Fit on128-by-128 Gauss points and score independently on256-by-256 Gauss points using current-time reference norms. Models are scalar float64 throughout. No reconstructed4096 admission claim follows from this diagnostic.

Controls:

1. Direct coefficients in all learned local features (64 per child for the registered model): offline capacity oracle, more variables than the actual method.
2. Direct coefficients in the zero-code child tangent (16 per child): same local variable count but a linearized model.
3. Actual nonlinear child codes, with fixed parent code: zero, tangent least-squares and norm-clipped tangent starts;60 LM iterations with120 more only if unfinished. Choose by fit-grid error, retain every start's independent score and convergence status.

Linear fits use column-scaled thin SVD with fixed relative cutoff1e-10, retaining all singular values, rank and retained condition number. This is a numerical rank-limited span diagnostic, not an unconditional mathematical lower bound. Nonlinear weights and parent remain untouched. The nonlinear fit and direct span use identical local learned features. Parent-only scores accompany each row. Joint-parent assembled results from the training driver are a separate experiment and must not be interpreted as apples-matched controls here; no joint-parent direct oracle is implemented in this first t0 diagnostic.

Spatial localization reports parent residual energy and remaining nonlinear residual energy in every retained node's core. Parent/descendant cores overlap; do not sum these entries as a partition. Saved code payloads permit independent plots and re-scoring. Timing includes JIT compilation and diagnostic scoring where stated; it is not a PDE runtime benchmark.

Decision: if the full span fails, investigate spatial features/geometry; if it succeeds but a converged nonlinear manifold fails, investigate child dimension/head expressivity; if nonlinear outcomes depend strongly on starts or remain unfinished, address fitting first. A successful tangent with poor nonlinear fitting is particularly strong evidence against a feature-capacity explanation. Good representation on these selected development fields does not establish generalization, PDE stability, online refinement or complete-runtime superiority.

## Prospective selected-joint parent-code variant

Optional `--root-code-source selected-joint` retains the same30-minute single-GPU cap and current-time error criteria. Default `initial` is unchanged. Register this option explicitly before running it; no automatic submission. Require a completed, locally/remotely verified retained source job and committed manifest, with matching selected model, result and allocation-comparison hashes.

For all12 selected development cases and each requested time/layout, take the exact stored cells and jointly fitted parent code from allocation_comparison.pkl. Require unique, complete requested coverage, expected code/cell dimensions and finite status/codes. Full20 rows must exactly match selected-checkpoint evaluation rows. All requested times for a given case/layout must retain identical cells. Never silently regenerate allocation geometry in this mode.

Within each row, freeze the identical retained parent code for full64-feature coefficients, zero16-tangent coefficients and nonlinear children. Add the stored selected child code as a fourth nonlinear start. Before interpreting results, reconstruct its untouched fit-grid and independent256 scores and require absolute agreement<=1e-10 with retained scores. Record untouched status, convergence flag, errors and original codes, including unfinished fits. This score-consistency threshold is a provenance/numerical check, not a relaxed scientific accuracy threshold.

This removes the comparison mismatch between initial-parent span controls and selected jointly fitted tree fields. It does not establish a globally optimal parent code: retained fits may still be unfinished. Parent codes can differ across layouts, so comparisons isolate conditional within-layout feature/manifold restrictions, not allocation superiority. If extended joint fits later change parent codes materially, their results motivate a separately registered stronger comparison. Full-span failure at a fixed unfinished parent remains conditional evidence, not an absolute decoder-capacity proof. No final-test fields or online truth-assisted refinement.
