# Native ROM interface independent audit

Independent reviewer numerical_2d_run accepts job 6613813 for low-level integration only. The audit verified 812 files, 43 staged hashes, exact runtime source cb9db577ec96375007e7929befb2980105ae7857, GPU/f64/highest preflight and empty stderr. Archive SHA256: b35b31000bdbdcdd7ed5e9b96be679353448c5826618ab19b7b449c772a97924.

All thirteen volume/boundary replay discrepancies are below 6.68e-16, including actual boundary stress gradients. Component maps are complete and disjoint; metric block discrepancies are below 7.08e-17. Saved float64 eigenbanks have full column ranks 48/32, ordered eigenvalues and positive cutoff gaps. Source-reviewed native eigen residual checks pass below 3.51e-15; sparse metric matrices were not retained, so the independent reviewer did not recompute those residuals or Gram matrices.

Retention 4667682a7 / mirror 660f59847 and exact duplicate cleanup bd6d1f91c / mirror 570887700 are complete. This audit admits neither physical references, training data nor actual reduced-solve accuracy.
