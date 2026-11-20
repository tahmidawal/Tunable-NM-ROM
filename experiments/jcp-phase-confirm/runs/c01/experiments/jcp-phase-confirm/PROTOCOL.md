# Frozen head dynamics and physical phase/shape confirmation

This protocol was fixed before new truth generation. It adds physical-centroid reliability checks and a fresh in-family confirmation to the accepted matched head/span study. It does not train or choose a new model.

## Fixed scientific comparison

The primary configuration is the accepted frozen learned bank and head with the original periodic NS3D family, N64, R64, k8, M584, time step 0.01, horizon 0.2. Head and span both include three free moving-frame unknowns. Common-initial-state head/span comparisons isolate evolution; the native span initialization is supplementary. All tolerances, fit starts, solver safeguards, exact weak operator, row preconditioner, and reference qualification are inherited unchanged. No cross-job timing claim is made.

The new whole-trajectory cohort uses seed 2026111901 and the first 16 row-wise parameter draws. Its parameter hash and exact/rounded tuple non-overlap with the recorded training and development groups are in COHORT_REGISTRATION.json. This opens no historical reserved cohort. The two solver tolerances and refined CNAB2 reference are run for every case. Failed qualification is retained and interpreted as failure of confirmation; no fresh-accuracy-driven hyperparameter repair, extra seed search, or case removal is allowed. The optional same-previous-state objective-containment probes and tangent diagnostics retain their prior role as local numerical observations.

## Phase and shape diagnostics

For each periodic coordinate, energy weights are summed along the other coordinates. The circular first-moment magnitude is divided by total energy to obtain concentration. Values below 0.1 flag a potentially ambiguous centroid; this declared descriptive cutoff is not a universal theorem. Report all coordinates, cases and times. Phase displacement is the wrapped difference between physical energy centroids, measured in fractions of the unit domain length. It is not the learned frame coordinate.

Ordinary and centroid-aligned field errors both use the norm of the original physical initial field. Alignment applies a Fourier shift determined by the reference centroid. It is a truth-assisted diagnostic, not an online correction or an optimized registration. The panel title must not attribute the whole head advantage to transport when the aligned-shape gap persists.

Saved development coefficients are replayed on a GPU and their metrics compared to the original saved values before concentration is used. This is postprocessing, not a new trajectory experiment. Recomputed raw reference hashes are reported, but scalar reproduction within 1e-10 is the cross-hardware check. Fresh and development panels stay separate. Cohort medians and full case ranges accompany individual-case traces; no representative case is selected by favorable method comparison.

## Execution and retention

One exclusive branch/worktree and namespace; GPU partition only, f64/highest, GPU preflight, code provenance, no data synchronization. Initial job reservation is 20 minutes and lane ceiling is 1.5 allocated GPU hours, including failed attempts. Code and protocol are committed and mirrored before submission. Results, solver histories, reconstructible states, per-coordinate concentrations, job metadata and checksums are retained before removing owned remote job duplicates. No merge is authorized.

## Glossary

A bank is a set of spatial functions. A head restricts their coefficients through a nonlinear map; a span allows arbitrary coefficients. A trajectory is a time sequence of fields. Weak tests measure equation error against smooth functions. Qualification means passing declared numerical checks. Concentration measures how well a periodic centroid is defined. Phase describes physical position; aligned shape describes the field after the stated translation. A fresh cohort is a set of new physical cases evaluated after the method is frozen. Training cases fit model weights; development cases inform model choices. A paired ratio compares methods on the same case. A code-only mirror backs up eligible source and result files without inherited archive blobs.
