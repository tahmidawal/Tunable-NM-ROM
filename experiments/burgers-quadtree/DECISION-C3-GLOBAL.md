# C3 frozen global-space decision — 2026-11-18

Close the tested global neural/POD diffusion-compression recipes as complete-cost negatives. Both finalists pass physical/reference, quadrature, time and finer-space gates on all registered development cases, but neither beats its paired eligible classical comparator. This is not a proof against all global or nonlinear architectures.

| Method | Worst actual4096 error | Error + uncertainty | Median complete query (ms) | Paired time ratios |
|---|---:|---:|---:|---|
|neural|0.69624%|0.70517%|9.367|1.525 / 1.523|
|POD|0.92596%|0.94492%|7.783|1.268 / 1.265|

The eligible paired baseline is spectral48, median 6.180ms. All cases are slower for both ROMs in both blocks. No positive break-even follows from adding nonnegative model preparation/training cost to slower warm queries.

Small global spaces failed projection; the bounded capacity extension recovered accuracy only with dense integration. The physical finer-feature check, separate192→256 quadrature check, half-step check and independent NumPy actual-field audit are retained. Final-only queries retain reduced states and decode only the final field, so unnecessary intermediate physical reconstruction is not charged against these finalists.

Reusable cached-array payload is 350500864 bytes (arrays only, not peak allocator/executable memory); preparation took 8.947s. Cold first queries and every timed state are retained.

Artifacts: `configs/c3-finalists.json`, `evidence/c3cost-assessment.json`, `evidence/c3cost-retention.json`, `evidence/c3cost-witness-audit.json`, all matched global checkpoints and raw owned job outputs. No test population was opened.

This closes C3 global compression only. The same owned campaign continues the explicitly registered family-B quadrature/time efficiency screen because existing controls justify cheaper local operators. The overall physical-accuracy plus substantial-cost objective remains unmet.
