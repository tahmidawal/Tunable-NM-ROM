# Burgers paired-query rerun after stationarity instrumentation repair

This is an instrumentation-only rerun of `accuracy07`. The original allocation failed at a charged-versus-posthoc normalized-gradient assertion during the middle mesh; its incomplete coverage, full fields, internal states and logs are checksum-preserved. No completed three-mesh accuracy claim is accepted from that attempt.

The failure diagnostic is `runs/accuracy07/STATIONARITY-DIAGNOSTIC.json`. A focused local GPU replay reproduced the archived failing field and internal states to roundoff. Independent NumPy analytic gradients agree with the charged and posthoc computations; cancellation and compiler fusion affect normalized gradients near stationarity. The original failing A100 charged-gradient array was not saved before the assertion, so its exact discrepancy cannot be recovered or claimed. The replay is a mechanism diagnostic, not benchmark timing.

The new `accuracy08` allocation retains the original stationarity threshold, solver iteration budgets, network dimensions, terminal checkpoint, candidate libraries, weak operators, quadrature refitting procedure and all six development cases. It persists every charged and posthoc gradient array with each complete query before acceptance checks. Their differences are checked independently after collection. Any disagreement in classification around the unchanged $10^{-6}$ threshold is flagged conservatively as unconfirmed convergence. A numerical failure retains raw output arrays and failure metadata before exiting.

The terminal training checkpoint, training targets, training log and cluster-regenerated refined references are inherited by content hash from the checksum-verified closed original attempt, with its job, source, GPU, seed and archive lineage recorded explicitly. Files are copied directly between private paralab directories; they are not regenerated unnecessarily or moved through login-node temporary storage. Actual supplied query inputs are regenerated from the same declared seeds in the new allocation. The two additional development cases were first opened in `accuracy07`; they remain part of this same development round and are not advertised as a new independent confirmation set. No endpoint is selected or trained using those results.

All four ROM arms and both iterative FOM controls are rerun together on one new GPU allocation at all three meshes and all repetitions. No time from the failed allocation is spliced into the new comparison. The original full-field timing, finite-data, f64/highest, burn-in, retention and independent operator/field-gradient audit requirements remain in force. The final paper cohort remains sealed. Source and artifact lineage are frozen before submission into the existing approved namespace in the private `accuracy08` directory.

## Glossary

- **Charged / posthoc gradient:** objective gradient retained inside the timed solver / independently recomputed at its saved coordinates after timing.
- **Stationarity:** normalized first-order objective gradient below the declared threshold; physical accuracy remains a separate measurement.
- **Instrumentation:** recording and checking the solver's diagnostics, without changing its update equations.
- **Lineage:** source, seed, machine and content hashes identifying the original generation of reused artifacts.
- **Paired-query rerun:** all model and full-grid controls measured again within the same new GPU allocation.
- **Development cohort:** cases available for method development, distinct from the still-sealed final paper cohort.
