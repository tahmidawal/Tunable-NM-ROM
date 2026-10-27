# Independent results audit: failed frozen fit

Reviewer `/root/independent_review` marked the failed-fit result and retained-input
evidence READY. The reviewer independently verified all final output hashes,
including the full contribution matrix, then recomputed the terminal residual and
the complete candidate-gradient vector in NumPy. These agree with the saved
training residual and the failed global KKT certificate. The support target was
not reached, the earlier optimum was uncertified, and the banned-node quality
limit was exceeded. The training permutation matches its frozen seed.

The terminal result correctly rejects construction. No new-support validation,
retry or alternate fit is released. A2 was not reached: this is a failed frozen
constructor, not a new-support rollout failure. The qualified full-residual POD
fallback remains valid and a qualified HR comparison remains unresolved.

The reviewer independently confirmed successful scheduler termination and clean
GPU/provenance preflight. Source, descriptors, exact matrix/candidate/target inputs
and terminal arrays remain retained. The matrix is intentionally outside Git and
must be preserved before any future worktree archival.
