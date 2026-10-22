No blocking issues found.

- Shortley–Weller fractions feed the RK4 start correctly. Only H’s \(n=64\) RK4 ladder fits the registered 20,480-step cap; finer grids are appropriately skipped.
- Curved SEM, PCG/eig eligibility, `phi_H`, curved quadrature, and the box evolution with H’s initial condition are consistent with the registered implementation. Existing H geometry controls pass.
- No concrete memory blocker identified for 80 GB GPU / 160 GB host. The 10.9-hour projection remains assumption-based; ×1.5 gives 16.31 hours, within 16:30.
- Step requests total 24.5 GPU-hours; including both planned 2-hour test jobs gives 28.5, below 30. Retries must fit the remaining allowance.
- Outputs support A10’s S1 comparison after the separate frozen test-once job; they cannot establish a geometry WIN without the additional registered evidence.

Read-only audit completed; no files changed, jobs submitted, or large arrays copied.

JOB(H_wide_S1): READY