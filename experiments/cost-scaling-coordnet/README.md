# Coord-net ROM online-cost skeleton (2026-08-13)

The coordinate-decoder arm of the online-cost scaling experiment. The full
write-up, protocol note, and the figure live in the sibling worktree
`2026-08-13-cost-scaling-cp/experiments/cost-scaling-cp/` (branch
`exp/2026-08-13-cost-scaling-cp`).

This arm times the GN/EQ arithmetic the future coordinate-decoder ROM will do:
FD 5-point residual at m=100 EQ nodes via 5m pointwise FiLM-decoder
evaluations (dx enters as a value, never a shape), jacfwd Jacobian over
z in R^k, k x k Levenberg-damped solve. Random weights (cost depends on
shapes, not values). n-freedom is asserted in code two ways: jaxpr shape
multisets identical between min and max N, and XLA `cost_analysis()` FLOPs
per GN iteration bit-identical across all N at fixed k.

Result (`coordnet_timing_pax106.json`, Slurm job 2345393, A100 pax106,
median of 30 solves, 10 GN iterations fixed): per-iteration time varies
<= 0.44% across N = 32 -> 512 at every k; 0.22 ms (k=2) -> 0.84 ms (k=32)
per iteration. FLOP CHECK PASSED across all 25 cells.

`coordnet_timing_spark-d69e.json` is the local GB10 smoke (provenance only,
not figure data).
