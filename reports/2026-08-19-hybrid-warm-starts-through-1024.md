# NM-ROM warm starts followed by FOM correction through N=1024

This final report covers the autonomous Poisson-2D and Burgers-2D architecture, hyperparameter, and solver push. Every numeric statement and table is generated from the checksummed run JSONs; superseded timing claims are identified inline.

## Outcome

- **Poisson:** the optimized genuine K=8 NM-ROM warm start has exactly 1 supported crossover in the balanced confirmation: N=1024, FOM tolerance 1e-06, 211.996 ms versus 217.578 ms, or 1.026x with clustered interval [1.014, 1.077]. It is a modest win against counting CG, not the best Poisson solver.
- **Burgers:** the calibrated cubic-history FOM is faster than linear history in 18 of 18 conditions. The guarded weak FiLM NM-ROM is slower than cubic in 18 of 18 conditions. The learned warm start does not win.

## What was optimized

Poisson tested the original full-grid K=8 FiLM path, a trust-region K=8 path decoded at fixed N=64, and a cached nonlinear GroupFiLM with K=16 and 58419 parameters. The train-only RBF latent predictor was stopped after its selected standardized latent MSE was 0.838. The optimized K8 path uses weak-form empirical quadrature with M=64 and m=256; construction at N=1024 is 4.255 ms.
Burgers first tuned the FOM itself: the selected inner tolerances are 1e-02, 1e-04, 1e-05 for outer tolerances 1e-06, 1e-08, 1e-10, with the exact Helmholtz preconditioner. A rank-32 correction basis still left 0.528 of validation correction, and the best translated/scaled deployable surrogate left 0.620; both missed their gate. The retained genuine FiLM control therefore uses K=8, M=64, m=256, a fixed coarse decode, two weak Jacobians per step, an exact-residual guard, and a charged live cubic fallback.

## Poisson: learned warm start across resolution

Rows through the last unbalanced mesh are screening evidence from the all-arm panel; their large losses are retained for the requested resolution ladder. The two finest meshes use the fresh-seed, position-balanced AB/BA confirmation and are authoritative near parity.

| design | N | FOM tau | hybrid ms | zero-CG ms | speedup | interval | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| screening panel | 32 | 1e-06 | 5.719 | 2.644 | 0.462x | [0.387, 0.547] | slower |
| screening panel | 32 | 1e-08 | 5.993 | 2.979 | 0.497x | [0.424, 0.578] | slower |
| screening panel | 32 | 1e-10 | 6.438 | 3.375 | 0.524x | [0.452, 0.605] | slower |
| screening panel | 64 | 1e-06 | 8.052 | 5.009 | 0.622x | [0.549, 0.704] | slower |
| screening panel | 64 | 1e-08 | 8.842 | 5.854 | 0.662x | [0.590, 0.734] | slower |
| screening panel | 64 | 1e-10 | 9.828 | 6.535 | 0.665x | [0.598, 0.734] | slower |
| screening panel | 128 | 1e-06 | 12.788 | 9.812 | 0.767x | [0.712, 0.827] | slower |
| screening panel | 128 | 1e-08 | 14.898 | 11.945 | 0.802x | [0.751, 0.854] | slower |
| screening panel | 128 | 1e-10 | 16.582 | 13.331 | 0.804x | [0.755, 0.854] | slower |
| screening panel | 256 | 1e-06 | 22.386 | 20.119 | 0.899x | [0.861, 0.940] | slower |
| screening panel | 256 | 1e-08 | 25.612 | 23.536 | 0.919x | [0.884, 0.955] | slower |
| screening panel | 256 | 1e-10 | 29.490 | 26.490 | 0.898x | [0.865, 0.932] | slower |
| balanced AB/BA | 512 | 1e-06 | 62.006 | 62.269 | 1.004x | [0.984, 1.058] | inconclusive / tie |
| balanced AB/BA | 512 | 1e-08 | 73.868 | 73.751 | 0.998x | [0.973, 1.018] | inconclusive / tie |
| balanced AB/BA | 512 | 1e-10 | 85.432 | 82.404 | 0.965x | [0.941, 0.981] | supported slower |
| balanced AB/BA | 1024 | 1e-06 | 211.996 | 217.578 | 1.026x | [1.014, 1.077] | supported faster |
| balanced AB/BA | 1024 | 1e-08 | 253.531 | 257.030 | 1.014x | [0.993, 1.031] | inconclusive / tie |
| balanced AB/BA | 1024 | 1e-10 | 290.348 | 288.587 | 0.994x | [0.980, 1.010] | inconclusive / tie |

### Balanced paired evidence at the two finest meshes

| N | tau | hybrid - zero ms [clustered interval] | favorable cases | status |
|---:|---:|---:|---:|---|
| 512 | 1e-06 | -0.799 [-2.691, 1.243] | 5/8 | inconclusive / tie |
| 512 | 1e-08 | +0.460 [-1.324, 2.046] | 3/8 | inconclusive / tie |
| 512 | 1e-10 | +3.300 [1.946, 4.989] | 0/8 | supported slower |
| 1024 | 1e-06 | -5.874 [-14.642, -2.418] | 7/8 | supported faster |
| 1024 | 1e-08 | -4.040 [-8.976, -0.573] | 7/8 | inconclusive / tie |
| 1024 | 1e-10 | +2.469 [-1.534, 6.457] | 3/8 | inconclusive / tie |

### Strong Poisson controls at the largest mesh

These rows are same-job comparisons from the all-arm panel. The spectral/direct margin is large enough that the timing-order issue affecting the small learned crossover cannot change the conclusion.

| FOM tau | selected spectral arm | spectral ms | zero-CG ms | speedup | FFT-DST ms |
|---:|---|---:|---:|---:|---:|
| 1e-06 | `spectral_q512` | 0.759 | 228.212 | 300.9x | 1.095 |
| 1e-08 | `spectral_q1024` | 0.814 | 269.776 | 331.3x | 1.096 |
| 1e-10 | `spectral_q1024` | 0.931 | 302.598 | 325.2x | 1.095 |

The learned-plus-spectral arm is not evidence for learning: at the largest mesh its total is

| FOM tau | GroupFiLM + q8 ms | matched selected spectral ms | excess ms |
|---:|---:|---:|---:|
| 1e-06 | 61.921 | 0.759 | 61.163 |
| 1e-08 | 81.103 | 0.814 | 80.288 |
| 1e-10 | 99.834 | 0.931 | 98.903 |

## Burgers: all final resolution/tolerance conditions

Times are medians across trajectory-level repetition medians. Intervals resample whole trajectories; timing repetitions are not treated as independent cases. Positive savings mean the named candidate is faster.

| N | FOM tau | inner tau | linear ms | cubic ms | cubic saving [interval] | guarded FiLM ms | FiLM saving vs cubic [interval] | accepted steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1e-06 | 1e-02 | 42.758 | 30.855 | 13.757 [7.994, 16.989] | 162.304 | -128.223 [-135.358, -125.091] | 0.5/50 |
| 32 | 1e-08 | 1e-04 | 67.709 | 45.098 | 22.869 [19.896, 26.964] | 177.013 | -128.577 [-136.136, -126.251] | 0.5/50 |
| 32 | 1e-10 | 1e-05 | 83.540 | 58.159 | 23.072 [18.658, 27.711] | 183.511 | -124.880 [-131.139, -122.274] | 0.5/50 |
| 64 | 1e-06 | 1e-02 | 35.262 | 24.723 | 11.304 [6.591, 13.952] | 156.777 | -130.403 [-134.719, -119.936] | 1.0/50 |
| 64 | 1e-08 | 1e-04 | 54.654 | 36.202 | 19.471 [16.477, 21.190] | 165.794 | -129.420 [-134.067, -120.671] | 1.0/50 |
| 64 | 1e-10 | 1e-05 | 64.665 | 45.692 | 16.411 [12.607, 21.586] | 178.242 | -130.943 [-134.286, -120.864] | 1.0/50 |
| 128 | 1e-06 | 1e-02 | 57.225 | 41.634 | 17.071 [10.985, 23.485] | 175.656 | -135.281 [-146.448, -124.674] | 2.0/50 |
| 128 | 1e-08 | 1e-04 | 91.813 | 59.339 | 33.441 [27.624, 37.453] | 193.722 | -136.883 [-144.113, -122.621] | 2.0/50 |
| 128 | 1e-10 | 1e-05 | 108.026 | 76.970 | 27.213 [17.227, 34.474] | 218.670 | -136.871 [-149.988, -126.045] | 2.0/50 |
| 256 | 1e-06 | 1e-02 | 49.594 | 36.399 | 13.949 [9.173, 18.544] | 163.926 | -126.731 [-132.385, -121.684] | 2.5/50 |
| 256 | 1e-08 | 1e-04 | 82.307 | 53.422 | 27.909 [25.172, 32.711] | 185.463 | -130.465 [-135.869, -128.032] | 2.5/50 |
| 256 | 1e-10 | 1e-05 | 98.572 | 71.787 | 23.055 [11.695, 31.044] | 202.962 | -131.194 [-138.861, -127.138] | 2.5/50 |
| 512 | 1e-06 | 1e-02 | 149.004 | 103.393 | 48.043 [30.225, 56.540] | 247.771 | -153.022 [-168.177, -126.975] | 4.0/50 |
| 512 | 1e-08 | 1e-04 | 226.070 | 141.080 | 78.998 [72.192, 97.854] | 282.273 | -145.404 [-161.792, -130.549] | 4.0/50 |
| 512 | 1e-10 | 1e-05 | 287.799 | 205.768 | 69.871 [28.686, 97.816] | 356.451 | -157.787 [-168.883, -132.847] | 4.0/50 |
| 1024 | 1e-06 | 1e-02 | 587.983 | 390.409 | 201.565 [131.581, 232.578] | 535.287 | -149.234 [-165.194, -135.892] | 4.0/50 |
| 1024 | 1e-08 | 1e-04 | 963.710 | 586.169 | 354.413 [317.928, 462.599] | 728.815 | -153.113 [-163.246, -125.555] | 4.0/50 |
| 1024 | 1e-10 | 1e-05 | 1204.624 | 851.428 | 312.525 [125.364, 419.855] | 1016.865 | -168.800 [-184.467, -158.268] | 4.0/50 |

## Accuracy, measurement, and provenance

Poisson pair confirmation: job 2664551 on NVIDIA A100 80GB PCIe, seed 20260820, 8 cases and 12 repetitions per case. Every method occupied first and second position 6 times per case. All 6 rows meet their named true-residual tolerance.
Burgers confirmation: job 2664725 on NVIDIA A100-PCIE-40GB, seed 20260819, 4 trajectories and 7 repetitions each. It contains 1512 timed records across 54 rows, with 0 nonzero flags and 0 breakdowns. The maximum returned residual/tolerance ratio is 0.997803.
Both jobs report `jax_backend=gpu`, x64=True, and matmul precision `highest`. Checksums and staged source hashes match; stderr/warning audits pass; both explicit remote job directories were deleted after the results were pulled.

## Retractions and final interpretation

- The earlier Poisson multi-arm claim of a learned crossover at every largest-mesh tolerance is retracted. Its cyclic schedule did not balance timing position. The balanced fresh-seed run supports only the single row reported above.
- The provisional Burgers selected-cohort panel is not the headline result. The final table uses the untouched confirmation seed and trajectory-clustered uncertainty.
- Architecture tuning improved the Poisson learned path enough for a small loose-tolerance crossover, but operator-aware classical structure wins by orders of magnitude. On Burgers, solver and history tuning dominate; the tested learned manifolds cannot supply enough accepted warm-start improvement to repay their construction cost.

## Source artifacts

- `worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/runs/pairfinal1/out/pairfinal1.json`
- `worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/runs/final1/out/final1.json`
- `worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/runs/confirm2/out/final_summary.json`
- `worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/runs/confirm2/out/confirm2.json`
- `worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/runs/corr_gate/out/corr_gate.json`
- `worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/runs/shift_corr_gate/out/shift_corr_gate.json`
