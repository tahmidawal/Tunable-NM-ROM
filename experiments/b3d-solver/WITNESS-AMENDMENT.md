# Adaptive check of the finer quadrature witness

Pre-data design: one case, number5 from the unchanged seed923801 validation cohort,
W1024/default4097 tests, CN step0.01, gl80. The original first-eight finer-rule check
has independently passed, closing legacy gl56 qualification. Its gl80 FS trajectory
has one mild fresh-stationarity miss among25 steps; gl80 itself is not qualified.

Run only original recording adaptive LM at tight1e-7 and tighter1e-8, retaining every
state and fresh step gamma. Reuse the accepted FS80 payload by checksum and the existing
verified gl64 tight CN anchor, whose complete provenance is copied in the small historical
inventory. Save FS80-to-AD80, AD64-to-AD80 and tight-to-tighter field distances with the
original full-grid Gram matrix and E0 initial norm. Verify finite states, nfail0 and each
fresh gamma at the requested tolerance; mark the anchor usable only when both runs pass
and their field distance is at most2.5e-5.

This strengthening control does not change the frozen original Q2 predicate, the already
admitted gl56 verdict, or the physical-error scoring definition. Keep failed anchors and
forbid interpretation as a converged witness if the checks fail. Budget15 minutes on
one H200, memory peak120GB; no additional quadrature or parameter sweep.
