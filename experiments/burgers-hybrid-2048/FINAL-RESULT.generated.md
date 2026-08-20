# Burgers-2D hybrid extension at N=2048

These are final, independently audited N=2048 results. The primary candidate is a classical residual-plus-exact-Helmholtz warm start, not a learned method or NM-ROM; its speedup is supported at tolerance 1e-6 only.

## Authoritative classical panel

| FOM tolerance | Linear tolerance | Cubic (ms) | Classical warm start (ms) | Speedup | Paired saving (ms) | Trajectory-cluster 95% CI (ms) | Supported |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1e-06 | 1e-02 | 532.817 | 343.673 | 1.550358x | 178.175 | [54.945, 195.984] | yes |
| 1e-08 | 1e-04 | 923.747 | 927.397 | 0.996064x | 5.293 | [-27.191, 67.302] | no |
| 1e-10 | 1e-05 | 1191.344 | 1209.532 | 0.984962x | 45.404 | [-20.894, 266.999] | no |

Only tolerance 1e-6 is a supported speedup. The 1e-8 and 1e-10 point paired-saving medians are positive while their aggregate speed ratios are below one; these noncommuting reducers disagree, and both clustered intervals cross zero, so no speedup is claimed. All timing outliers were retained.

The candidate is charged for 50 full-grid exact-upwind residual evaluations and 50 exact Helmholtz inverses per trajectory. Timing is warmed compiled online latency; reference generation, compilation, loading, and first-query latency are excluded.

## Reference gate

| Trajectories | Maximum actual residual | Maximum step disagreement | Maximum trajectory disagreement | Flags/breakdowns | Pass |
|---:|---:|---:|---:|---:|:---:|
| 4 | 9.537719e-13 | 4.682753e-14 | 1.793184e-14 | 0 / 0 | yes |

This reference uses two prospectively gated exact-Helmholtz routes on untouched seed 20260830. The fixed public-JAX truth path is excluded after the development diagnostic reproduced its nonfinite-update freeze.

## Genuine weak FiLM NM-ROM sensitivity

The learned sensitivity used a separate preregistered job and seed, so its absolute times are compared only within that job, never against the primary job above.

| FOM tolerance | Comparison | Control (ms) | FiLM NM-ROM (ms) | Control / FiLM | Paired saving (ms) | 95% CI (ms) | Supported | Guard accepts |
|---:|:---|---:|---:|---:|---:|---:|:---:|---:|
| 1e-06 | cubic vs FiLM | 756.309 | 862.278 | 0.877106x | -100.433 | [-111.816, -91.028] | no | 4.0 / 50 |
| 1e-06 | classical vs FiLM | 713.903 | 862.420 | 0.827790x | -190.550 | [-267.018, -70.379] | no | 4.0 / 50 |
| 1e-08 | cubic vs FiLM | 1418.694 | 1497.729 | 0.947231x | -104.314 | [-117.931, -49.726] | no | 4.0 / 50 |
| 1e-08 | classical vs FiLM | 1331.101 | 1497.767 | 0.888724x | -211.999 | [-552.563, 30.375] | no | 4.0 / 50 |
| 1e-10 | cubic vs FiLM | 2213.739 | 2333.497 | 0.948679x | -119.626 | [-135.276, -104.982] | no | 4.0 / 50 |
| 1e-10 | classical vs FiLM | 2012.698 | 2333.343 | 0.862581x | -225.143 | [-433.529, -157.890] | no | 4.0 / 50 |

The FiLM arm is supported in 0 of 3 cells against cubic and 0 of 3 against the classical warm start. It performs 100 reduced Jacobians per trajectory but its exact-residual guard accepts a median 4 of 50 steps. Conversely, cubic is supported faster than FiLM at all three tolerances; the classical arm is supported faster at 1e-6 and 1e-10, while 1e-8 is inconclusive.

## Audit and exclusions

The primary bundle contains 288 raw timing records and 144 immediate burn records. The generated audit and an independent recomputation of paired medians, bootstrap intervals, outliers, reference histories, work, accuracy, and record grids both pass.

Excluded from claims: fixed-eight and fixed-25 public-JAX reference attempts (zero timing rows), the first unbatched diagnostic replay, the negative repaired diagnostic, and the accidental seed-20260829 N=32 implementation smoke. The authoritative primary uses seed 20260830 and job 2680178.
