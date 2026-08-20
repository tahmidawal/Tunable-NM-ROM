# Hybrid speed push through N=1024

This final report covers the bounded follow-on architecture, objective, and solver search for Poisson-2D and Burgers-2D. All numbers and tables are generated from the audited run JSONs; the learned and classical outcomes are kept explicitly separate.

## Final outcome

- **Poisson learned hybrid:** no new one-update candidate passed. The strongest genuine NM-ROM result remains the earlier N=1024, tolerance 1e-06 row: 211.996 ms versus 217.578 ms for counting CG, 1.026x [1.014, 1.077]. It is not the fastest Poisson solver.
- **Burgers practical winner:** the learned trajectory NM-ROM failed its scaling gate. A classical full-grid residual-plus-Helmholtz warm start is supported in 10/18 cells. At N=1024 and tolerance 1e-06, it takes 226.403 ms versus 308.432 ms for optimized cubic-history FOM, 1.362x, with paired saving 81.583 ms [19.396, 95.499].

## Why the learned routes stopped

The Poisson conditional candidates started from a direct physical-parameter surrogate and performed one online weak Gauss--Newton update. Alpha=1 minimized a truncated field-error objective; the independent follow-on fixed alpha=0.5 to target energy/A-error. Every update was accepted and reduced its projected objective, but both global A-error and CG work became worse. The table shows the final solver-aligned round.

| N | direct construct ms | direct A-error | direct CG | direct total ms | NM-ROM construct ms | NM-ROM A-error | NM-ROM CG | NM-ROM total ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.210 | 3.084e-02 | 142.94 | 5.125 | 0.418 | 3.916e-02 | 145.19 | 5.391 |
| 256 | 0.215 | 4.770e-02 | 604.00 | 20.312 | 0.421 | 5.346e-02 | 610.81 | 20.801 |

For Burgers, the one-shot trajectory representation looked compact at N=64 but left 0.372 of the correction at N=256, above the frozen 0.25 gate. It therefore never advanced to a weak online wrapper or final timing.

## Poisson: strongest learned result and production controls

| N | tolerance | NM-ROM+FOM ms | counting CG ms | speedup [95% CI] | verdict |
|---:|---:|---:|---:|---|---|
| 512 | 1e-06 | 62.006 | 62.269 | 1.004x [0.984, 1.058] | inconclusive / tie |
| 512 | 1e-08 | 73.868 | 73.751 | 0.998x [0.973, 1.018] | inconclusive / tie |
| 512 | 1e-10 | 85.432 | 82.404 | 0.965x [0.941, 0.981] | supported slower |
| 1024 | 1e-06 | 211.996 | 217.578 | 1.026x [1.014, 1.077] | supported faster |
| 1024 | 1e-08 | 253.531 | 257.030 | 1.014x [0.993, 1.031] | inconclusive / tie |
| 1024 | 1e-10 | 290.348 | 288.587 | 0.994x [0.980, 1.010] | inconclusive / tie |

At N=1024 the fastest eligible structured controls are:

| tolerance | control | total ms | counting-CG ms | speedup vs counting CG |
|---:|---|---:|---:|---:|
| 1e-06 | `dense DST direct` | 0.665 | 228.212 | 343.0x |
| 1e-08 | `dense DST direct` | 0.673 | 269.776 | 401.0x |
| 1e-10 | `spectral_q1024` | 0.931 | 302.598 | 325.2x |

## Burgers: fresh-seed full-resolution panel

The successful candidate is **classical and nonlearned**. Before each fine FOM step it performs one full-grid residual evaluation and one exact Helmholtz inverse. The warmed online wall time includes all 50 residual evaluations and all 50 inverses. Finish Newton and BiCGStab counters do not include those extra operations, so wall time is the authoritative speed metric. Compilation, module loading, data generation, and reference generation are excluded.

| N | FOM tol | cubic ms | corrected ms | speedup | paired saving ms [95% trajectory CI] | supported |
|---:|---:|---:|---:|---:|---|:---:|
| 32 | 1e-06 | 29.911 | 23.187 | 1.290x | 6.692 [3.976, 9.958] | yes |
| 32 | 1e-08 | 43.327 | 46.896 | 0.924x | -3.501 [-4.264, -1.819] | no |
| 32 | 1e-10 | 59.720 | 55.324 | 1.079x | 5.709 [0.944, 6.693] | yes |
| 64 | 1e-06 | 23.236 | 17.909 | 1.297x | 5.401 [2.215, 7.580] | yes |
| 64 | 1e-08 | 35.575 | 37.235 | 0.955x | -1.032 [-2.677, -0.164] | no |
| 64 | 1e-10 | 44.706 | 43.255 | 1.034x | 3.006 [-0.415, 7.417] | no |
| 128 | 1e-06 | 36.164 | 27.286 | 1.325x | 8.881 [4.675, 11.362] | yes |
| 128 | 1e-08 | 53.464 | 56.245 | 0.951x | -2.228 [-4.607, 1.316] | no |
| 128 | 1e-10 | 74.415 | 70.152 | 1.061x | 7.015 [-0.396, 20.733] | no |
| 256 | 1e-06 | 34.274 | 25.501 | 1.344x | 8.757 [2.012, 11.419] | yes |
| 256 | 1e-08 | 46.178 | 46.525 | 0.993x | -0.234 [-1.933, 4.120] | no |
| 256 | 1e-10 | 61.897 | 59.734 | 1.036x | 3.801 [0.114, 14.164] | yes |
| 512 | 1e-06 | 88.885 | 65.113 | 1.365x | 23.699 [5.939, 27.897] | yes |
| 512 | 1e-08 | 134.486 | 137.913 | 0.975x | -2.715 [-8.394, 8.364] | no |
| 512 | 1e-10 | 175.443 | 170.617 | 1.028x | 9.681 [0.292, 35.868] | yes |
| 1024 | 1e-06 | 308.432 | 226.403 | 1.362x | 81.583 [19.396, 95.499] | yes |
| 1024 | 1e-08 | 473.612 | 490.791 | 0.965x | -13.944 [-36.474, 26.855] | no |
| 1024 | 1e-10 | 631.715 | 621.222 | 1.017x | 26.786 [1.470, 156.630] | yes |

All 6 tolerance-1e-6 rows are supported wins; none of the 6 tolerance-1e-8 rows is supported. At tolerance 1e-10, 4/6 rows are supported. The N=1024 tight-tolerance interval is positive but very wide because the final population contains four trajectories.

## Measurement and audit

Burgers final job 2667698 ran on NVIDIA A100 80GB PCIe from clean commit `9cdeca5aabc9` with GPU backend, f64/x64, and highest matmul precision. It contains 1,728 timed records and 864 burns in exact AB/BA order. All solver flags and breakdowns are zero; all named residual tolerances pass. Tukey outliers are counted and retained, never removed.

All referenced cluster job directories were checksum-verified after pull and explicitly deleted. The Poisson branch closes at `57329c0`; the Burgers branch closes at `559583c`. Neither branch was merged automatically.

## Input artifacts

- `worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/runs/paramlmgate1/out/paramlmgate1.json`
- `worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/runs/paramritzg1/out/paramritzg1.json`
- `worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/runs/pairfinal1/out/pairfinal1.json`
- `worktrees/2026-08-19-poisson-hybrid-1024/experiments/poisson-hybrid-1024/runs/final1/out/final1.json`
- `worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/runs/trajectory_representation/out/trajectory_quality.json`
- `worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/runs/dynamic_final/out/dynamic_final_summary.json`
- `worktrees/2026-08-19-burgers-hybrid-1024/experiments/burgers-hybrid-1024/runs/dynamic_final/out/dynamic_final.json`
