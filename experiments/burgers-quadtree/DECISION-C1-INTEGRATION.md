# C1 full512 integration branch decision

2026-11-18, after c1cost/c1fast/c1cubature, before further space changes.

The initial-field four-patch hybrid is physically accurate: worst actual4096 error0.5271%, or0.5460% including conditional reference uncertainty. Its full-integration complete query is slower than tuned spectralN32/dt0.01: primary median time ratios2.529 and2.524 in independent blocks, zero cases faster. Setup7.61s is additional. No speed success.

Two finite attempts to reduce integration work for this frozen512-coordinate space failed:

- Smaller tensor quadrature: atdt0.0125, worst errors plus uncertainty are46.37%,32.28%,12.72%,4.61% for orders48/64/80/96. Order128 retains0.5494%. Coarse128 and the three registered input-width policies do not pass all physical/integration/time gates.
- Positive TRAIN-only empirical quadrature: support128/256 gives worst PDE errors61.35%/59.81%, with zero of32 cases below1%. All trajectories remain finite; this is numerical inaccuracy, not a hardware failure. The unsketched training flux errors also expose the poor rule fit.

Stop developing those smaller rules for the unchanged512-coordinate bank. Preserve every checkpoint, positive rule, trajectory and timing. The larger qualifieddt0.0125 alone reduces prepared solve time to~8ms, but a complete-output speed advantage is unproven and further classical time tuning remains relevant.

Next distinct architectural hypothesis: compress the trial space itself in projected-diffusion eigenvalue order, then separately assess field projection and PDE accuracy. This targets highly oscillatory, stiff directions that make the full bank expensive to integrate. It does not assert that deleting them is harmless. Genuine complete quadtree leaf partitions are the other already registered architectural branch. Do not reopen unseen tests or reinterpret this branch's negative speed result as success.
