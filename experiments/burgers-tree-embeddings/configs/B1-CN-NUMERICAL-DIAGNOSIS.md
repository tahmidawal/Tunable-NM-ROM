# Separate CN time and test-space sensitivities

The first CN pilot materially improves pulse accuracy. The global control now clears the physical pilot accuracy target, while the tree has an early-time excess error despite small projection error and stationary weak solves. This difference prevents attributing all remaining error to shared spatial discretization. Diagnose temporal stiffness and insufficient test visibility separately before a decoder-capacity claim.

Use exactly the same three exposed cases, models, initial-callable allocation, L2048 fine-upwind physics and dense integration. Two finite changes from the retained CN dt0.0025/4K-test pilot:

- Halve dt to0.00125 with the same test rule, maximum60 GPU-minutes (already conditionally registered in B1-TEMPORAL-REPAIR.md).
- Retain dt0.0025 and start the common test ladder at8K rather than4K, maximum45 GPU-minutes. This enriches both methods together, with the same normalized visibility safeguard. No quadrature change in this diagnostic.

Compare full physical returned states to the original CN pilot at every requested time. A physical error improvement without numerical convergence is not admission. Record projection/dynamics separation, all solver status, test visibility and actual4096 final outputs. The original BE/CN failures remain. These are independent one-factor numerical diagnostics, not an architecture sweep or performance benchmark. Submit only within the current finite accounting ceiling and two-job cap. Use their results to choose the next finite integration or solver check; do not combine settings retrospectively and call them prequalified.
