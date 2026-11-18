# B1 shared local-span diagnostic

The parent/local attribution still fails at 1.20% even after releasing the root and replacing local features with 36 polynomial channels on the old trees. Before training more compressed heads, test whether a richer, better allocated direct local span is adequate.

Use the training-only spectral POD128 global control, fixed C2 windows, retained parents, and shared tensor Legendre local features. Whiten within each cell with its physical Gram matrix; do not mix compact supports across cells. Solve the assembled ridge problem with coefficients in all active cells, comparing penalties 1e-10 and 1e-8. Count actual coefficients, not the old 128-variable cap.

Registered six diagnostic configurations (side polynomial count, retained local nodes, maximum depth, allocation): (4,28,3,initial), (6,28,4,initial), (6,28,4,swept), (8,28,4,initial), (8,28,4,swept), (6,52,4,swept). The swept indicator translates the known initial callable over a speed envelope based only on its sampled maximum, over nine time offsets through 0.25. It receives no evolved truth or privileged pulse descriptor. No deployable adaptation claim.

Screen all eleven snapshots of dev005, align000, align004, align005. Then evaluate the best configuration on all original 16 development and 12 alignment trajectories. Screening is explicitly incomplete coverage; only the full evaluation can guide advancement. Integration starts at 128-by-128, and any promising candidate requires doubled integration and exact 4096 reconstruction before qualification. Training compressed child heads is conditional on an adequate direct span. This diagnostic is analytical, not a replacement claim for a neural method. Reserve at most 120 GPU-minutes within B1.
