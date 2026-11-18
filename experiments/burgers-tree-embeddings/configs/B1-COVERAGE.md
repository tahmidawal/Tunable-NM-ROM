# B1 prefix-stable coverage expansion

The normalized 16-variable head dramatically improves its training fits while its optimized development fits remain poor. The training-only global spectral span also retains a development floor even at rank 512. This diagnoses inadequate global generalization/coverage in addition to the known parent and local-span limitations; it does not prove that data alone will fix them.

Before generation, register the protocol's training expansion from 64 to 128 trajectories and development expansion from 16 to 32. Generate only new indices 64–127 and 16–31, respectively, from the unchanged cohort seeds. Retain the original development and alignment results as a separate panel for before/after comparisons. Final-test seeds and fields remain unopened. Every new case gets all five original space/time reference levels, current-time normalization and the unchanged 0.1% reference allowance. Independent cross-checks on the new largest-uncertainty cases are required before admitting expanded data to claims.

Reserve at most 90 GPU-minutes for the added references. This is within B1's six-hour allocation, not a new compute expansion. No new architecture recipe is charged for data generation; any retraining on the larger cohort will be registered and counted separately.
