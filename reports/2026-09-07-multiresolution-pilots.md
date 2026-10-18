# Multiresolution development pilots: accuracy and complete-query cost

This report covers frozen-network mesh-transfer pilots, bounded solver changes and controlled training refinements of the current separable NM-ROM against efficient FOM solvers. The numbers are provisional development evidence; independent confirmation and the full resolution study remain open.

The tested frozen decoders produce solutions on new meshes, but the completed nonlinear-ROM studies have not established a complete-query advantage over efficient FOMs. Increasing resolution does not reliably reduce ROM error in these pilots. Further work must address representation, initialization or reduced-solver cost, according to the PDE.

The older ViT + CP architecture is excluded. Waves use only the fresh verified lineage. The final cohorts remain unopened. Different rows use different physical error definitions, stated below; they must not be ranked as a common cross-PDE accuracy score.

## Latest audited development results

These rows show the finest requested mesh from each latest completed study. Heat, Burgers and Poisson use their declared development cost-to-accuracy selections. Wave rows show the first predeclared larger-head seed at the largest repeated step; both seeds and all tested steps are retained in the detailed findings. Different physical norms and cohorts are explicit and must not be ranked as one cross-PDE accuracy score.

| PDE | Output intervals | Cases | Error norm | Worst error (%) | ROM / FOM query (ms) | Paired FOM/ROM | Timing outliers, ROM / FOM | FOM comparison |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Heat | 1024 | 12 | Current L2, full grid | 4.5555 | 27.733 / 21.4106 | 0.76673 | 0 / 0 | Tested envelope |
| Burgers | 1024 | 4 | Initial L2, full grid | 4.6351 | 49.463 / 24.9034 | 0.52388 | 0 / 0 | Tested envelope |
| Poisson | 512 | 30 | Steady L2, common grid | 6.8016 | 4.01824 / 2.26715 | 0.56129 | 0 / 0 | Tested envelope |
| Reflective wave | 512 | 2 | Initial wave state, common grid | 6.2029 | 1217.24 / 61.584 | 0.050593 | 0 / 0 | Same grid only |
| Absorbing wave | 512 | 2 | Initial wave state, common grid | 5.9132 | 1336.64 / 277.79 | 0.20786 | 0 / 0 | Same grid only |

The cost-to-accuracy targets shown are Heat 5%, Burgers 5%, Poisson 10%, with explicitly empirical reference allowances. Wave costs are raw same-grid comparisons, without a coarse-FOM envelope. Ratios above one would favor ROM; none of these rows does. The absorbing wave still has large late error relative to its remaining field. These are development results, not independent final confirmation.

Query costs are medians of per-case repetition medians, including supplied host input and requested host outputs. Paired ratios are medians of per-case FOM/ROM ratios. Timing outliers count repetitions longer than twice their own case/configuration median; none is discarded. All comparisons pair methods within the same GPU job.

The earlier pilot table and the controlled changes below preserve how these conclusions were obtained; earlier rows are not the current best settings.

## Primary configurations from the first pilots

These are explicit representative settings, not a claim that every row is the cheapest configuration at a qualified accuracy target. Burgers uses a same-mesh FOM here; its native report also includes a coarse-mesh output envelope. Poisson uses the development-selected stopping setting shown. Raw cost ratios are separate from physical-accuracy qualification.

| PDE | Intervals | ROM setting | Error metric | Cases | ROM median / worst error (%) | FOM worst error (%) | ROM / FOM query (ms) | Raw paired FOM/ROM |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Heat | 64 | CN dt=0.025 | Current L2 | 4 | 5.858 / 9.736 | 0.07092 | 27.775 / 1.221 | 0.04396 |
| Heat | 128 | CN dt=0.025 | Current L2 | 4 | 5.855 / 9.736 | 0.01772 | 27.738 / 1.3691 | 0.04925 |
| Burgers | 256 | dt=0.005; stall=0.001 | Initial L2 | 4 | 1.954 / 3.961 | 2.226 | 39.284 / 18.416 | 0.4917 |
| Burgers | 512 | dt=0.005; stall=0.001 | Initial L2 | 4 | 2.88 / 5.274 | 1.582 | 42.379 / 27.875 | 0.6954 |
| Poisson | 256 | tau=0.01 | Steady L2 | 6 | 0.8927 / 7.363 | 0.01546 | 4.8257 / 2.0607 | 0.395 |
| Poisson | 512 | tau=0.01 | Steady L2 | 6 | 0.8928 / 7.363 | 0.003677 | 4.8859 / 2.4406 | 0.4995 |
| Reflective wave | 256 | dt=0.0025 | Initial wave state | 2 | 46.17 / 55.37 | 0.5047 | 3300.4 / 14.024 | 0.004249 |
| Reflective wave | 512 | dt=0.0025 | Initial wave state | 2 | 46.01 / 55.2 | 0.1445 | 3354.3 / 64.534 | 0.01924 |
| Absorbing wave | 256 | dt=0.0025 | Initial wave state | 2 | 7.576 / 7.655 | 0.08748 | 3221.3 / 94.984 | 0.02948 |
| Absorbing wave | 512 | dt=0.0025 | Initial wave state | 2 | 7.573 / 7.654 | 0.01951 | 3274.3 / 297.2 | 0.09076 |

All timings include the supplied host field, initialization/source projection, solve or evolution, and requested host field outputs. Each cost is the cohort median of per-case repetition medians. The paired ratio is the median of per-case FOM/ROM cost ratios. Each pair was measured in one job on one GPU; raw wall times must not be compared across PDE jobs as a hardware-normalized ranking.

**Heat:** errors are maxima over output times relative to the current reference norm, evaluated on the common observation grid. This is a newly verified, restricted single-bump development family. It does not yet cover the broader heat use case.

**Burgers:** errors use the initial reference norm and the common observation grid. The first reference refinement estimate leaves target qualification unresolved. Cold-fit budget exits are retained in the native records and independent audit; an observed small error is not proof of a stationary initial fit. The follow-ups below test more starting guesses, a finer reference and the initial fitting objective.

**Poisson:** errors are relative solution norms for each steady source on the common observation grid. Tighter stationary solves still leave a worst-case error floor. The reference has empirical refinement evidence; development qualification is not an independent final-cohort result.

**Waves:** the reported metric is the maximum of displacement, velocity and energy-state errors on the common observation grid. Displacement is divided by the initial displacement L2 norm; velocity and energy-state error use the initial phase-energy scale $\sqrt{2E(0)}$. Here $E(0)$ is the initial wave energy, including displacement gradients and velocity; the initial velocity alone can be zero. Absorbing errors relative to the small remaining field are substantially larger and must also be reported. The time-step pair is resolved for these cases, so a smaller step does not remedy the observed error.

## Evidence and limitations

| PDE | Numerical source commit | GPU job | Native findings |
|---|---|---|---|
| Heat | `ad882df3ecb8614d4cafa6a25761e920f3074e3d` | 3349961 | [Findings / source records](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/HEAT-PILOT-NOTES.md) |
| Burgers | `96befb8815a4c20c8d9f2862502e454ab8402032` | 3350012 | [Findings / source records](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/pilot01/out/pilot.json) |
| Poisson | `82d2c3261126cb150bb83220ec3edbcf0dd5f489` | 3350079 | [Findings / source records](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot01/FINDINGS.md) |
| Waves | `a02aacd9578f46a9bee0dcb093acd35e62d10739` | 3349951 | [Findings / source records](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/pilot01/analysis/FINDINGS.md) |

All first-pilot outputs were checksum-collected and their exact remote job directories removed. Numerical checks cover GPU execution, precision, relevant operators, reference refinement and state advancement. Root independently recomputed the preserved heat, Burgers and Poisson field errors. The wave root review checks source design, paired accounting and artifact hashes; it is not an independent full-grid regeneration of every wave metric.

The rigorous reference-bound field remains unspecified where only empirical refinement is available. No paper-wide speedup, optimal capacity, optimized training cost, broad-family robustness, or independent confirmation is established.

## Bounded improvements with unchanged network weights

These follow-ups retain the first-pilot physical cases and checkpoints. They provide development evidence about specific numerical changes; no new training or final-cohort evaluation is included.

### Heat: less stringent stopping reduces cost

Source `06f8ed98653e95bbf094c42212171d0db3519541`, job `3350258`. The native audit verifies 504 timed invocations; compiled and modular fields, latent states and iteration counters agree at each matched tolerance. The original strict control is measured again in the same job, so the improvement does not compare clocks across jobs.

| Intervals | Strict modular ms | Relaxed compiled ms | Paired improvement | FOM ms | ROM worst current error (%) | Maximum field drift |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 28.323 | 14.177 | 1.954 | 1.1972 | 9.7356 | 0.000347272 |
| 128 | 28.821 | 14.268 | 1.919 | 1.3757 | 9.7356 | 0.000347323 |

The relaxed normalized-gradient tolerance is `1e-05`; the strict tolerance is `1e-09`. Field drift is relative to the current reference norm and stays below the predeclared `0.001` ceiling. Fewer nonlinear iterations account for most of the gain; compiling the query alone gives a smaller improvement. The FOM remains faster, and the head reconstruction error remains.

[Complete heat runtime findings](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/HEAT-RUNTIME-NOTES.md).

### Poisson: more weak modes do not remove the bank limitation

Source `0175877d40c96bc4c0e28cd2c8e2e3b0f6cbc966`, job `3350408`. The native audit recomputes 672 timed errors and verifies the compiled-query parity controls. The following table uses stationary solves with the early residual-reduction stop disabled.

| Intervals | Requested / retained weak modes | Worst physical error (%) | Compiled query ms | Direct FOM ms |
|---|---:|---:|---:|---:|
| 256 | 64 / 64 | 7.3187 | 4.9668 | 1.4808 |
| 256 | 128 / 129 | 7.2546 | 4.6038 | 1.4808 |
| 256 | 256 / 257 | 7.2542 | 4.4342 | 1.4808 |
| 512 | 64 / 64 | 7.3189 | 5.1563 | 1.9171 |
| 512 | 128 / 129 | 7.2546 | 5.0417 | 1.9171 |
| 512 | 256 / 257 | 7.2542 | 4.8622 | 1.9171 |

| Intervals | Worst full-bank projection error (%) | Worst best-recorded head-fit error (%) |
|---|---:|---:|
| 256 | 5.7465 | 7.2635 |
| 512 | 5.7388 | 7.2564 |

These reference-only reconstruction diagnostics use the same-grid field norm and cannot initialize a deployed query. The unrestricted bank already has a difficult-source error larger than the next all-case target; more weak modes cannot repair missing spatial directions. Compiling the unchanged solver gives only small timing changes. A stronger bank/training study and a specialized small-system linear solve remain distinct next tests.

[Complete Poisson follow-up findings](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot02/FINDINGS.md).

### Burgers: a finer reference and a resolution-dependent initial-fit defect

Full-query source `9a2025c1f0624db91fb8ecbcfef0d0d433373187`, job `3350134`. The native audit independently recomputes 828 invocation errors from preserved observation fields. The largest empirical reference estimate is `0.003736669`, allowing development checks at the targets shown; stricter targets remain unresolved. The additive margin uses the larger of the spatial-plus-time difference and the Richardson estimate for each case. The initial analytic norm is fixed. These are empirical development checks, not rigorous reference bounds.

| Output intervals | Target (%) | Selected ROM | Selected FOM envelope | ROM / FOM ms | Paired FOM/ROM |
|---|---:|---|---|---:|---:|
| 256 | 10 | `rom_L256_dt0.005_stall0.01_starts1` | `fom_L128_out256_dt0.01_ntol0.01` | 36.547 / 9.7213 | 0.2755 |
| 256 | 5 | `rom_L256_dt0.005_stall0.01_starts1` | `fom_L256_out256_dt0.01_ntol0.01` | 36.547 / 10.121 | 0.2871 |
| 512 | 10 | `rom_L512_dt0.005_stall0.01_starts1` | `fom_L128_out512_dt0.01_ntol0.01` | 39.98 / 12.164 | 0.3162 |
| 512 | 5 | Target unattained | `fom_L256_out512_dt0.01_ntol0.01` | — / 12.308 | — |
| 1024 | 10 | `rom_L1024_dt0.005_stall0.01_starts1` | `fom_L128_out1024_dt0.01_ntol0.01` | 55.073 / 24.702 | 0.4558 |
| 1024 | 5 | Target unattained | `fom_L256_out1024_dt0.01_ntol0.01` | — / 24.866 | — |

Selections minimize the cohort median of case-median query times, including coarse-FOM interpolation to the requested output. The native Burgers report also supplies selections using pooled repetition medians; closely timed FOM choices can differ. All 36 cold-fit budget exits remain visible across the full pilot search. Configured budget or small-improvement exits are explicit early stops, not proof of stationary fitting. More initial guesses did not remove the error increase with resolution, and no tested envelope establishes a ROM advantage.

Cold-fit-only source `0fb42607811425d639ba58714da2210f152cf463`, job `3350594`. The audit checks 240 declared fits, retained fields, unchanged cases and checkpoint bytes. The following full-grid initial-error diagnostics are provisional: finer-grid full norms and bank floors were computed in the GPU job, but only the common-grid fields were retained for independent reconstruction. These fits do not include a PDE rollout.

| Intervals | Edge Gram (%) | Edge QR (%) | Fixed midpoint (%) | Fixed Gauss (%) | Full-grid QR (%) | Unrestricted bank floor (%) |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 2.6005 | 2.6005 | 2.5467 | 2.5629 | 2.5447 | 0.1898 |
| 512 | 5.4672 | 5.4672 | 3.2958 | 3.2836 | 3.2828 | 1.4953 |
| 1024 | 8.6951 | 8.6951 | 3.8737 | 3.8562 | 3.8549 | 2.2593 |

The table uses a budget of `180` iterations and `4` training-code starting guesses, reporting the worst physical case. The same edge samples give essentially the same result under Gram and QR fitting; fixed physical midpoint or Gauss sampling greatly reduces the finer-grid fitting error. The remaining bank floor also grows with resolution. Thus both the changing sampled objective and a real frozen-bank representation loss matter. The earlier explanation based only on optimizer starting guesses is withdrawn. The clipped initial-condition family has not changed. These are best-recorded nonlinear fits, not stationary or globally optimal certificates. Fixed physical field sampling with charged interpolation is an initialization method, not strong-form PDE collocation. Corrected-initializer rollouts were unmeasured at this stage; the continued-development section below now reports them.

[Complete Burgers follow-up findings](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/reports/2026-09-07-burgers2d-multiresolution.md).

### Burgers: larger steps save some cost but introduce solver-limit effects

Source `ffd57ee68cd1a1a61503556135d74220a0ca2fa0`, job `3353574`. This keeps the frozen decoder, larger-budget Gauss initialization, physical cases and small-improvement stopping rule fixed. Every returned initial field agrees across the step-size controls. Both ROM and efficient same/coarse-grid FOM candidates are measured again within this job.

| Output intervals | ROM step | Worst complete-grid error (%) | Complete query (ms) | Evolution / first-step budget exits |
|---|---:|---:|---:|---:|
| 512 | 0.005 | 3.7133 | 40.857 | 0 / 0 |
| 512 | 0.01 | 4.6194 | 35.73 | 9 / 6 |
| 512 | 0.025 | 8.585 | 31.94 | 12 / 12 |
| 512 | 0.05 | 21.841 | 27.438 | 21 / 12 |
| 1024 | 0.005 | 3.9076 | 54.57 | 0 / 0 |
| 1024 | 0.01 | 4.6351 | 49.463 | 6 / 6 |
| 1024 | 0.025 | 8.8998 | 46.457 | 12 / 12 |
| 1024 | 0.05 | 21.913 | 42.425 | 18 / 12 |

Exit counts include all timing repetitions. Fewer time steps do not produce a proportional reduction in nonlinear work: larger steps require more solver trials, and several hit the configured budget. The existing predictor already extrapolates previous latent states when that reduces the weak residual; it has no previous-step history at startup. The larger-step accuracy changes therefore combine time discretization and incomplete nonlinear solves. They do not establish the error of a fully converged time integrator.

| Output intervals | Empirical target (%) | Selected ROM step | ROM / FOM envelope query (ms) | Paired FOM/ROM |
|---|---:|---:|---:|---:|
| 512 | 10 | 0.025 | 31.94 / 12.147 | 0.39096 |
| 512 | 5 | 0.01 | 35.73 / 12.147 | 0.33999 |
| 1024 | 10 | 0.025 | 46.457 / 24.903 | 0.55971 |
| 1024 | 5 | 0.01 | 49.463 / 24.903 | 0.52388 |

The efficient FOM envelope remains faster at the qualified development targets. Physical target qualification uses the explicit empirical reference allowance and is not a stationarity certificate. The native report retains initial/later errors, complete-grid and common-grid margins, staged component diagnostics, every FOM candidate and every configured-stop count.

The owner reconstructs 172 dense output artifacts and verifies repeat hashes for all 516 timed calls; the largest complete-grid metric disagreement is `7.21645e-16`. Root independently checks their common-grid errors, with maximum disagreement `8.326673e-17`. There are 0 nonfinite calls and 0 FOM tolerance failures. Independent final confirmation remains open.

[Complete Burgers timestep findings](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/reports/2026-09-07-burgers2d-multiresolution.md).

All cost ratios in this report use ratios of per-case timing medians before the cohort median. Some native exploratory reports also retain medians of per-repetition ratios or select configurations by the pooled median across all repetitions under an explicit different label; those statistics are not interchangeable.

## Continued development: controlled training and solver changes

### Heat: broader training coverage improves accuracy with the unchanged spatial bank

Source `0597f0dd505de537ebbda459d9db330132126d44`, job `3352849`. The spatial bank is exactly unchanged. Each refined head receives 8000 updates, with the same architecture and per-snapshot relative squared-error loss. The original and expanded training cohorts are crossed with the recorded minibatch seeds. All final checkpoints are retained; none was selected by a validation training loss.

| Head endpoint | Training trajectories | Initial median / worst relative error (%) | Later worst reconstruction error (%) | Initial-fit gate |
|---|---:|---:|---:|---|
| `frozen` | 32 | 5.0379 / 9.7356 | 6.2355 | Fail |
| `original_seed790714` | 32 | 4.6531 / 9.4113 | 5.898 | Fail |
| `original_seed790715` | 32 | 4.6769 / 9.4312 | 5.9152 | Fail |
| `expanded_seed790714` | 160 | 2.6559 / 3.9531 | 2.1682 | Pass |
| `expanded_seed790715` | 160 | 2.6697 / 4.0303 | 2.1124 | Pass |

The predeclared gate requires every initial fit to be stationary and below 5% error. Both expanded-cohort endpoints pass; continuing optimization on only the original cohort does not. Each endpoint uses its own training-code initialization library. This demonstrates an improvement in the combined training-and-initialization procedure; it does not isolate better weights from better starting codes. A stationary fit is a local convergence result, not a proof of global optimality.

Full rollouts below use the already tested relaxed solver tolerance. Errors are relative to the current reference on the common observation grid; the same restricted physical family and development cases are retained.

| Intervals | Head endpoint | Worst rollout error (%) | ROM / direct FOM query (ms) | Paired FOM/ROM | Nonstationary initial fits / steps |
|---|---|---:|---:|---:|---:|
| 64 | `frozen` | 9.7356 | 13.734 / 1.138 | 0.08166 | 0 / 0 |
| 64 | `expanded_seed790714` | 3.9531 | 12.382 / 1.138 | 0.09205 | 0 / 0 |
| 64 | `expanded_seed790715` | 4.0303 | 12.171 / 1.138 | 0.09363 | 0 / 0 |
| 128 | `frozen` | 9.7356 | 13.716 / 1.2259 | 0.08978 | 0 / 0 |
| 128 | `expanded_seed790714` | 3.9531 | 12.214 / 1.2259 | 0.09486 | 0 / 0 |
| 128 | `expanded_seed790715` | 4.0303 | 12.011 / 1.2259 | 0.09656 | 0 / 0 |

The accuracy gain survives evolution on both query meshes, but the efficient direct FOM remains faster. The root independently recomputed all 280 timed outputs; the largest metric disagreement was `0`. The empirical spectral-reference difference is `1.810033e-10`; no rigorous bound is supplied. Saved offline refinement durations include compilation and host work without a dedicated warm-timing protocol; they are observed costs, not a training-speed comparison.

[Complete heat training findings](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/HEAT-HEAD-NOTES.md). [Heat refinement figure](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/figures/heat-head.png).

### Poisson: a guarded small-system solver gives a modest query improvement

Source `20f96592b14c2d9727eb40341d2c6cbcb568f464`, job `3352868`. The unchanged decoder and nonlinear objective are tested with a specialized Gauss–Jordan solve. Its numerical guard and generic-solver fallback are inside the timed query. The table pairs it with the generic compiled solver and the direct FOM in the same job.

| Intervals | Requested / retained modes | tau | Generic / specialized / FOM query (ms) | Paired generic/specialized | Worst physical error (%) |
|---|---:|---:|---:|---:|---:|
| 256 | 64 / 64 | 0.01 | 4.1684 / 4.0364 / 1.9248 | 1.0417 | 7.3626 |
| 256 | 64 / 64 | 0 | 5.2462 / 5.1026 / 1.9248 | 1.0459 | 7.3187 |
| 256 | 256 / 257 | 0.01 | 4.5543 / 4.0984 / 1.9248 | 1.1084 | 7.2542 |
| 256 | 256 / 257 | 0 | 4.7529 / 4.3824 / 1.9248 | 1.119 | 7.2542 |
| 512 | 64 / 64 | 0.01 | 5.0262 / 4.7197 / 2.435 | 1.0581 | 7.363 |
| 512 | 64 / 64 | 0 | 5.9294 / 5.422 / 2.435 | 1.1063 | 7.3189 |
| 512 | 256 / 257 | 0.01 | 5.2577 / 4.8576 / 2.435 | 1.0551 | 7.2542 |
| 512 | 256 / 257 | 0 | 5.7968 / 5.3563 / 2.435 | 1.0625 | 7.2542 |

The specialized solver preserves physical accuracy but does not beat the FOM. The residual-reduction stop may finish before stationarity; the stationary controls disable that stop.

Root independently checks 672 invocation metrics and 1200 saved linear systems. The largest metric difference is `1.387779e-16` and the largest linear backward error is `1.534149e-16`. There are 0 timed fallbacks, 3 solver-counter changes and 0 stop-reason changes. The 3 instrumented-replay counter mismatches are retained; exact trajectory identity is not claimed. These replay checks are numerical evidence, not timing measurements.

[Complete Poisson solver findings](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot03/FINDINGS.md).

### Burgers: the corrected initial fit improves complete rollouts

Source `d73fb4119057d4826c783d02829dd08835fe7a24`, job `3352857`. The network weights, physical initial-condition family and efficient FOM candidates are unchanged. Fixed physical Gauss sampling charges interpolation from the supplied input field. Both complete-grid and common-grid output errors are preserved, with separate reference-refinement margins for each grid.

| Output intervals | Target (%) | Selected ROM | Worst common / full-grid error (%) | ROM / FOM envelope query (ms) | Paired FOM/ROM | IC budget stops |
|---|---:|---|---:|---:|---:|---:|
| 256 | 10 | `rom_L256_edge_ic60_dt0.005_stall0.01_starts1` | 4.5399 / 4.5399 | 36.676 / 10.277 | 0.29294 | 3 |
| 256 | 5 | `rom_L256_edge_ic60_dt0.005_stall0.01_starts1` | 4.5399 / 4.5399 | 36.676 / 10.665 | 0.30413 | 3 |
| 512 | 10 | `rom_L512_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | 3.7132 / 3.7133 | 39.676 / 12.098 | 0.31227 | 3 |
| 512 | 5 | `rom_L512_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | 3.7132 / 3.7133 | 39.676 / 12.261 | 0.32117 | 3 |
| 1024 | 10 | `rom_L1024_edge_ic60_dt0.005_stall0.01_starts1` | 8.4972 / 8.6953 | 54.392 / 24.634 | 0.45953 | 3 |
| 1024 | 5 | `rom_L1024_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | 3.9109 / 3.9076 | 54.904 / 25.549 | 0.48201 | 0 |

At 1024 intervals, the matched primary rollout's worst complete-grid error changes from 8.6953% with edge initialization to 3.9076% with the larger-budget Gauss fit. The efficient FOM envelope remains faster at every reported target. The envelope permits a cheaper coarse solve and charges interpolation to the requested output. The native report retains every selected FOM configuration.

Budget exits and configured small-improvement stops remain admissible only under the stated empirical physical-error check; they are not stationary-fit certificates. All such counts remain in the native records. The tighter target with an unresolved reference budget is not promoted to qualified evidence.

The owner independently reconstructs 240 complete-grid artifacts representing all 720 timed calls; the largest complete-grid metric disagreement is `4.163336e-17`. Root separately recomputes all common-grid errors, with maximum disagreement `2.775558e-17`. There are 0 nonfinite outputs and 0 FOM tolerance failures. The reference margin is empirical and independent final confirmation remains open.

[Complete corrected Burgers rollout findings](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/reports/2026-09-07-burgers2d-multiresolution.md).

### Fresh waves: compression and autonomous dynamics both matter

Source `a0c4d0a7d7b59b05615846e61df0499ef5de4f62`, job `3353136`. The learned bank and existing nonlinear head remain frozen. The saved affine control has the same displacement and velocity dimensions as the nonlinear head; the larger affine and full-bank controls use additional coordinates. They all remain inside the learned neural bank. The speed-dependent affine propagator includes the nonzero-offset forcing and is charged inside the query.

| Boundary | Intervals | Method | Displacement / phase dimensions | Median / worst time-max error, initial normalization (%) | Method / same-grid FOM query (ms) | Raw paired FOM/method |
|---|---:|---|---:|---:|---:|---:|
| dirichlet | 256 | rom | 16 / 32 | 46.165 / 55.367 | 3334.92 / 13.6384 | 0.0040895 |
| dirichlet | 256 | affine16 | 16 / 32 | 26.332 / 30.205 | 13.8045 / 13.6384 | 0.98861 |
| dirichlet | 256 | affine32 | 32 / 64 | 4.3226 / 5.1579 | 13.9311 / 13.6384 | 0.979 |
| dirichlet | 256 | full64 | 64 / 128 | 3.0506 / 3.2126 | 13.8164 / 13.6384 | 0.98807 |
| dirichlet | 512 | rom | 16 / 32 | 46.014 / 55.204 | 3380.99 / 62.3308 | 0.018436 |
| dirichlet | 512 | affine16 | 16 / 32 | 26.332 / 30.205 | 60.6831 / 62.3308 | 1.0272 |
| dirichlet | 512 | affine32 | 32 / 64 | 4.3395 / 5.1888 | 60.3147 / 62.3308 | 1.0335 |
| dirichlet | 512 | full64 | 64 / 128 | 3.1116 / 3.2896 | 61.9178 / 62.3308 | 1.0067 |
| absorbing | 256 | rom | 16 / 32 | 7.5763 / 7.6554 | 3257.96 / 94.7872 | 0.029089 |
| absorbing | 256 | affine16 | 16 / 32 | 16.826 / 18.676 | 11.0369 / 94.7872 | 8.5845 |
| absorbing | 256 | affine32 | 32 / 64 | 5.6362 / 5.9237 | 11.4512 / 94.7872 | 8.2736 |
| absorbing | 256 | full64 | 64 / 128 | 2.8477 / 3.0206 | 11.4077 / 94.7872 | 8.3062 |
| absorbing | 512 | rom | 16 / 32 | 7.5729 / 7.6543 | 3300.85 / 279.53 | 0.084669 |
| absorbing | 512 | affine16 | 16 / 32 | 16.817 / 18.666 | 61.6481 / 279.53 | 4.5348 |
| absorbing | 512 | affine32 | 32 / 64 | 5.6276 / 5.9127 | 54.9659 / 279.53 | 5.1376 |
| absorbing | 512 | full64 | 64 / 128 | 2.8469 / 3.0201 | 62.5465 / 279.53 | 4.468 |

At equal dimension, the affine control improves reflective rollout error but worsens absorbing rollout error. The larger linear spaces improve both, while the nonlinear head can reconstruct individual snapshots better than its matched affine control. Snapshot reconstruction therefore does not by itself establish accurate autonomous dynamics. These observations do not establish an irreducible error for every nonlinear manifold of the same dimension.

Some linear controls are faster than the listed same-grid FOM, especially for the absorbing problem. Those are classical linear-control results, and the benchmark has not yet swept cheaper coarse FOMs with charged output interpolation. They do not establish a nonlinear-decoder or cost-to-tolerance speed advantage.

| Absorbing output intervals | Method | Final energy-state error / current reference norm, case range |
|---|---|---:|
| 512 | rom | 2.92808–4.10705 |
| 512 | affine16 | 21.4504–46.2361 |
| 512 | affine32 | 2.69765–3.65008 |
| 512 | full64 | 1.32915–2.15488 |

The absorbing reference becomes small. Even the full-bank linear control has a final energy-state error larger than the remaining reference; its small initial-normalized error must not be described as accurate relative prediction at late times. Zero-field fits and weak normal-force diagnostics are retained, without attributing the entire failure to either mechanism. The full-bank control represents zero exactly and has no normal-force residual within the bank by construction, yet still has late absorbing error; neither diagnostic alone certifies physical accuracy.

All 8 nonlinear time-step comparisons pass, with maximum required difference `1.04107e-06`. The selected diagnostic fits have 0 nonstationary exits, and maximum longer-budget objective change `0`. The owner independently reconstructs the full-grid ROM fields, affine propagators, coordinate bank, head derivatives, curvature and diagnostic fits. Root separately checks all 156 declared timed calls on the common observation grid, with maximum metric difference `3.108624e-15`. The audit reuses the saved reference trajectories; full-grid FOM metrics at nonreference time steps remain outside its reconstruction scope. Rigorous reference bounds remain unspecified.

[Complete wave dynamics findings](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/FINDINGS.md). [Wave error evolution](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/dynamics-error-evolution.png).

### Absorbing waves: a missing conservation property is a concrete diagnostic

Read-only postprocessing of the archived dynamics run checks a global moment preserved by its full discrete equations:

$$I(t)=\langle 1,v(t)\rangle_M+c\langle 1,u(t)\rangle_B,\qquad \dot I=0.$$

Here $M$ contains area integration weights, $B$ contains outgoing-boundary weights including both corner contributions, and $c$ is the supplied speed. The native diagnostic derives this identity from the actual discrete Laplacian and damping. At 512 intervals, the relative error in projecting the spatial constant into the learned bank is `0.004093844`. The reduced moment-generator rows are nonzero: this bank does not preserve the original invariant for general states.

| Intervals | Case | Method | Initial moment error | Maximum drift from own start | Final moment error |
|---|---:|---|---:|---:|---:|
| 512 | 0 | frozen MLP16, dt=0.0025 | 0.006142049 | 0.01738408 | -0.001512215 |
| 512 | 0 | affine16 | 0.001467529 | 0.02290407 | 0.006978501 |
| 512 | 0 | affine32 | -9.881988e-05 | 0.001557765 | -0.0007796877 |
| 512 | 0 | full64 | 0.0007440567 | 0.0008986995 | -3.089903e-05 |
| 512 | 1 | frozen MLP16, dt=0.0025 | 0.01649773 | 0.01823518 | -0.001492485 |
| 512 | 1 | affine16 | 0.0006066674 | 0.02146288 | 0.01165329 |
| 512 | 1 | affine32 | 0.0006058239 | 0.001310659 | 0.0002725378 |
| 512 | 1 | full64 | -0.0009765562 | 0.0007404492 | -0.001598021 |

The saved full references have maximum moment drift `4.336809e-17` on this mesh. The table separates initialization error from subsequent drift; its entries are signed moment errors or absolute drift, not relative field errors. Even unrestricted bank projection changes the input moment. The native findings retain that quantity separately from nonlinear fitting error.

This establishes a missing discrete conservation property. It does not establish how much of the late physical error it causes. Adding a constant test direction and enforcing the initial moment are proposed separate interventions; neither has been tested here, and preserving this moment alone would not certify local-field accuracy.

[Moment formulas, checks and complete traces](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/dynamics02/analysis/ABSORBING-MOMENT.md).

### Poisson: training coverage and relative loss are distinct changes

Source `0ed8384c30ff150d0f25adfe1dead41a8a739c26`, job `3353137`. The factorial crosses original versus expanded training coverage with global versus per-field relative squared-error normalization. The prescribed budget is 10000 updates per continuation, with unchanged architecture and identical initial weights. Evaluation uses 6 previously inspected and 24 fresh development sources, outside training. The fresh development cohort is now available for method selection; it is not sealed final confirmation.

| Intervals | Development cohort | Endpoint | Training sources / completed updates | Median / worst physical error (%) | Failures at target | Invalid / nonstationary cases |
|---|---|---|---:|---:|---:|---:|
| 256 | existing_development | original_frozen | 512 / 0 | 0.76019 / 7.2542 | 1 / 6 at 5% | 0 / 0 |
| 256 | existing_development | original_global | 512 / 10000 | 0.75819 / 7.1918 | 1 / 6 at 5% | 0 / 0 |
| 256 | existing_development | expanded_global | 2048 / 10000 | 0.90515 / 7.4612 | 1 / 6 at 5% | 0 / 0 |
| 256 | existing_development | original_relative | 512 / 10000 | 1.3192 / 6.0846 | 1 / 6 at 5% | 0 / 0 |
| 256 | existing_development | expanded_relative | 2048 / 10000 | 1.3618 / 5.7675 | 1 / 6 at 5% | 0 / 0 |
| 256 | fresh_development | original_frozen | 512 / 0 | 1.9545 / 7.6688 | 6 / 24 at 5% | 0 / 0 |
| 256 | fresh_development | original_global | 512 / 10000 | 1.9465 / 7.7635 | 6 / 24 at 5% | 0 / 0 |
| 256 | fresh_development | expanded_global | 2048 / 10000 | 1.8541 / 8.5 | 5 / 24 at 5% | 0 / 0 |
| 256 | fresh_development | original_relative | 512 / 10000 | 2.0831 / 6.8016 | 5 / 24 at 5% | 0 / 0 |
| 256 | fresh_development | expanded_relative | 2048 / 10000 | 1.8473 / 7.4745 | 3 / 24 at 5% | 0 / 0 |
| 512 | existing_development | original_frozen | 512 / 0 | 0.76017 / 7.2542 | 1 / 6 at 5% | 0 / 0 |
| 512 | existing_development | original_global | 512 / 10000 | 0.75817 / 7.1918 | 1 / 6 at 5% | 0 / 0 |
| 512 | existing_development | expanded_global | 2048 / 10000 | 0.90513 / 7.4612 | 1 / 6 at 5% | 0 / 0 |
| 512 | existing_development | original_relative | 512 / 10000 | 1.3192 / 6.0846 | 1 / 6 at 5% | 0 / 0 |
| 512 | existing_development | expanded_relative | 2048 / 10000 | 1.3618 / 5.7675 | 1 / 6 at 5% | 0 / 0 |
| 512 | fresh_development | original_frozen | 512 / 0 | 1.9545 / 7.6688 | 6 / 24 at 5% | 0 / 0 |
| 512 | fresh_development | original_global | 512 / 10000 | 1.9465 / 7.7635 | 6 / 24 at 5% | 0 / 0 |
| 512 | fresh_development | expanded_global | 2048 / 10000 | 1.8541 / 8.5 | 5 / 24 at 5% | 0 / 0 |
| 512 | fresh_development | original_relative | 512 / 10000 | 2.0831 / 6.8016 | 5 / 24 at 5% | 0 / 0 |
| 512 | fresh_development | expanded_relative | 2048 / 10000 | 1.8473 / 7.4745 | 3 / 24 at 5% | 0 / 0 |

These are the tighter generic stationary-control solves. The failure count applies the physical-error target with its empirical reference adjustment and solver/endpoint gates. Original and expanded sets share an update/batch budget; the expanded set receives fewer visits per source. Both the spatial bank and nonlinear head are refined in this Poisson study. At query time each endpoint uses its own mean training code. Thus these results compare complete training procedures under the stated budget, without isolating a universal effect of more data.

| Intervals | All-development target (%) | Selected ROM endpoint / solver / tau | ROM / eligible FOM query (ms) | Paired FOM/ROM |
|---|---:|---|---:|---:|
| 256 | 10 | original_global / rom_gj / 0.01 | 4.3097 / 1.7825 | 0.41362 |
| 256 | 5 | Target unattained | — / 1.7825 | — |
| 256 | 1 | Target unattained | — / 1.7825 | — |
| 256 | 0.1 | Target unattained | — / 1.7825 | — |
| 512 | 10 | original_frozen / rom_gj / 0.01 | 4.6255 / 2.1976 | 0.46878 |
| 512 | 5 | Target unattained | — / 2.1976 | — |
| 512 | 1 | Target unattained | — / 2.1976 | — |
| 512 | 0.1 | Target unattained | — / 2.1976 | — |

The complete-query envelope uses every development case and includes the charged coarse-grid FOM option. Per-case timing repetition arrays, component medians, outliers, failed gates and paired factorial error contrasts are retained in the native findings. Model selection for any later speed experiment must use all cases and both meshes rather than an inspected example. Offline training durations include compilation and diagnostics and are not a paired warm training-speed comparison.

Root independently recomputes 7680 invocation metrics, checks declared source draws and endpoint accounting, and finds maximum metric disagreement `2.359224e-16`. The owner additionally verifies source/checkpoint history and training normalization samples against an independent CPU reference. There are 0 specialized-solver agreement failures and 0 timed fallbacks. Reference adjustment remains empirical; no continuum certificate or global nonlinear optimum is claimed.

[Complete Poisson training findings](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot04/FINDINGS.md).

### Fresh waves: larger nonlinear heads improve accuracy, with a remaining cost gap

Source `0351e9865ac4559c9e9dbd7bfe4b8a52596e143c`, job `3353701`. Each boundary receives 2 fixed reconstruction-only endpoints with 32 displacement coordinates, each trained for 10000 updates on the original 64 regenerated training trajectories. The spatial bank stays fixed with 64 weak equations. The equations-to-coordinate ratio changes from 4 to 2; the nonlinear system remains overdetermined. No velocity or rollout loss is added.

The table shows the finer requested mesh (512 intervals). The native findings retain both meshes, every repeated step-size setting and both seeds.

| Boundary | Method | Step | Median / worst time-max error, initial normalization (%) | Method / same-grid FOM query (ms) | Raw paired FOM/method |
|---|---|---:|---:|---:|---:|
| dirichlet | frozen_mlp16_seed691200 | 0.0025 | 46.014 / 55.204 | 3373.82 / 61.584 | 0.018253 |
| dirichlet | new_mlp32_seed691200 | 0.01 | 5.2687 / 6.2029 | 1217.24 / 61.584 | 0.050593 |
| dirichlet | new_mlp32_seed691200 | 0.0025 | 5.2698 / 6.2044 | 4603.88 / 61.584 | 0.013377 |
| dirichlet | new_mlp32_seed691201 | 0.01 | 5.5303 / 6.7742 | 1216.63 / 61.584 | 0.050619 |
| dirichlet | new_mlp32_seed691201 | 0.0025 | 5.5339 / 6.7794 | 4605.2 / 61.584 | 0.013373 |
| dirichlet | affine32 | 0 | 4.3395 / 5.1888 | 59.5704 / 61.584 | 1.0338 |
| absorbing | frozen_mlp16_seed691200 | 0.0025 | 7.5729 / 7.6543 | 3288.12 / 277.79 | 0.084483 |
| absorbing | new_mlp32_seed691200 | 0.01 | 5.5122 / 5.9132 | 1336.64 / 277.79 | 0.20786 |
| absorbing | new_mlp32_seed691200 | 0.0025 | 5.5121 / 5.9129 | 5094.42 / 277.79 | 0.054531 |
| absorbing | new_mlp32_seed691201 | 0.01 | 5.5495 / 6.1811 | 1328.15 / 277.79 | 0.20919 |
| absorbing | new_mlp32_seed691201 | 0.0025 | 5.5495 / 6.1811 | 5050.13 / 277.79 | 0.055013 |
| absorbing | affine32 | 0 | 5.6276 / 5.9127 | 51.6608 / 277.79 | 5.4807 |

At the same step, the larger nonlinear heads improve initial-normalized rollout error but increase query cost. Using their larger tested step recovers some cost, while the efficient same-grid FOM remains faster. At the finer mesh shown, the larger nonlinear heads do not improve the all-case worst error of their matched affine control. A finer time integrator alone does not close the physical-error gap.

All 48 adjacent nonlinear step comparisons pass the unchanged `0.01` development criterion; their maximum required difference is `0.0003036077`. The 16 finest-step queries are separately audited accuracy controls, excluded from every timing summary and selection. Their saved single-call latencies may include uncached compilation.

| Absorbing method | Step | Final energy-state error / current reference norm, case range |
|---|---:|---:|
| frozen_mlp16_seed691200 | 0.0025 | 2.92808–4.10705 |
| new_mlp32_seed691200 | 0.01 | 2.92709–3.22814 |
| new_mlp32_seed691201 | 0.01 | 3.12151–4.66729 |
| affine32 | 0 | 2.69765–3.65008 |

Late absorbing errors remain larger than the remaining reference field. Improved error relative to the initial state does not establish accurate relative prediction at late times. The separate moment diagnostic has not been corrected in this experiment.

The native audit verifies 228 timed calls and 16 fine controls, with 0 failed timed trajectories and 0 failed fine trajectories. It reconstructs full-grid ROM fields, head derivatives and fitting diagnostics, verifies training/checkpoint lineage, and checks the affine generator. Root independently recomputes all common-field metrics and accounting roles, maximum disagreement `3.108624e-15`. Reference bounds remain empirical; the full-grid timed FOM reconstruction limitation from the prior wave audit remains. No coarse-FOM envelope, new development cohort or sealed-final confirmation is included.

[Complete larger-head wave findings](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/k32heads03/analysis/FINDINGS.md). [Larger-head wave error evolution](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/k32heads03/analysis/heads32-error-evolution.png).

### Poisson: field-based initialization saves some cost; the FOM remains faster

Source `20893525a16603a99f527dbd9e10affcdc206b9d`, job `3354845`. Both checkpoints are frozen: the original model and the relative-loss continuation selected using all development sources on both meshes. The factorial crosses two equivalent sine-source projections with mean-code versus nearest-training-code initialization. Nearest lookup compares supplied-field weak coefficients with cached decoder predictions from training codes; it uses no Gaussian descriptors or evaluation answers.

| Intervals | Checkpoint | Projection | Initialization | Median / worst physical error (%) | Query (ms) | Median solver attempts |
|---|---|---|---|---:|---:|---:|
| 256 | original_frozen | sine products | mean code | 1.905 / 7.6688 | 4.36125 | 12 |
| 256 | original_frozen | sine products | nearest code | 1.905 / 7.6688 | 3.79992 | 8 |
| 256 | original_frozen | DST gather | mean code | 1.905 / 7.6688 | 4.17404 | 12 |
| 256 | original_frozen | DST gather | nearest code | 1.905 / 7.6688 | 3.90369 | 8 |
| 256 | original_relative | sine products | mean code | 1.8495 / 6.8016 | 4.36739 | 12 |
| 256 | original_relative | sine products | nearest code | 1.8495 / 6.8016 | 3.93937 | 7.5 |
| 256 | original_relative | DST gather | mean code | 1.8495 / 6.8016 | 4.34453 | 12 |
| 256 | original_relative | DST gather | nearest code | 1.8495 / 6.8016 | 3.73487 | 7.5 |
| 512 | original_frozen | sine products | mean code | 1.905 / 7.6688 | 4.68915 | 12 |
| 512 | original_frozen | sine products | nearest code | 1.905 / 7.6688 | 4.18059 | 8 |
| 512 | original_frozen | DST gather | mean code | 1.905 / 7.6688 | 4.59367 | 12 |
| 512 | original_frozen | DST gather | nearest code | 1.905 / 7.6688 | 4.32107 | 8 |
| 512 | original_relative | sine products | mean code | 1.8494 / 6.8016 | 4.60523 | 12 |
| 512 | original_relative | sine products | nearest code | 1.8494 / 6.8016 | 4.01824 | 7.5 |
| 512 | original_relative | DST gather | mean code | 1.8494 / 6.8016 | 4.8363 | 12 |
| 512 | original_relative | DST gather | nearest code | 1.8494 / 6.8016 | 4.20458 | 7.5 |

Paired mean-start/nearest-start cost ratios range from 1.08629 to 1.15832; ratios above one favor nearest starts. Paired sine-product/DST-gather ratios range from 0.950756 to 1.00211, so there is no consistent projection-cost gain. A projector selected by aggregate median latency can differ from the one favored by the median paired ratio; the native report retains both statistics.

The stopping threshold stays $\tau\|r(z_0)\|_2$ for each call's own initial code. A nearer start can require a tighter absolute residual, and may finish by stationarity rather than the relative-reduction stop. The target is never re-anchored. Cost changes therefore describe initialization together with the existing stopping procedure, not an isolated iteration-count effect.

| Intervals | All-development target (%) | Selected checkpoint / projection / initialization | ROM / FOM envelope query (ms) | Paired FOM/ROM |
|---|---:|---|---:|---:|
| 256 | 10 | original_relative / DST gather / nearest code | 3.73487 / 1.75097 | 0.45678 |
| 256 | 5 | Target unattained | — / 1.75097 | — |
| 256 | 1 | Target unattained | — / 1.75097 | — |
| 256 | 0.1 | Target unattained | — / 1.75097 | — |
| 512 | 10 | original_relative / sine products / nearest code | 4.01824 / 2.26715 | 0.56129 |
| 512 | 5 | Target unattained | — / 2.26715 | — |
| 512 | 1 | Target unattained | — / 2.26715 | — |
| 512 | 0.1 | Target unattained | — / 2.26715 | — |

All 480 single-call stationary controls are retained separately, with 0 invalid case/configuration outputs. They never enter the speed envelope. Projection parity compares the same initialization; different minima between mean and nearest starts are allowed and judged by their physical errors and solver checks.

Timed outputs include the full field plus returned stop reasons, counters, latent states, guard results and initialization metadata. Stationarity is independently recomputed afterward from the same invocation, alongside physical-error and parity diagnostics. Offline cache/operator construction and compilation remain outside the query.

The owner audits all 2280 outputs and reconstructs the coordinate bank, decoder, cache, residuals and gradients. Root recomputes their saved field errors, maximum disagreement `3.469447e-16`, and independently replays 120 source-field lookup panels. There are 0 lookup mismatches, 0 projection-gate failures and 0 timed guarded-solver fallbacks. The efficient tested FOM envelope remains faster. Reference allowance is empirical and sealed-final confirmation remains unopened.

[Complete Poisson projection and initialization findings](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot05/FINDINGS.md).

### Heat: the refined heads transfer across the wider mesh ladder

Source `88ae5905d3f1ec424edbc4d7b8027e19eebe296b`, job `3354958`. Both expanded-coverage heads and their training-code libraries remain byte-identical to the prior endpoints. This run repeats 4 original development inputs and adds 8 fresh draws from the same restricted family. The requested intervals are `[64, 128, 256, 512, 1024]`, with shared observations at 64 intervals and complete-grid errors also checked. There is no new training or final-cohort evaluation.

| Output intervals | Cohort | Frozen endpoint | Full-grid median / worst current-relative error (%) | Common-grid worst error (%) | Complete query (ms) |
|---|---|---|---:|---:|---:|
| 64 | fresh | expanded_seed790714 | 3.4811 / 4.56 | 4.56 | 12.7703 |
| 64 | fresh | expanded_seed790715 | 3.4745 / 4.5593 | 4.5593 | 12.7522 |
| 64 | original | expanded_seed790714 | 2.6559 / 3.9531 | 3.9531 | 12.3313 |
| 64 | original | expanded_seed790715 | 2.6697 / 4.0303 | 4.0303 | 11.737 |
| 128 | fresh | expanded_seed790714 | 3.4795 / 4.56 | 4.56 | 12.4394 |
| 128 | fresh | expanded_seed790715 | 3.4728 / 4.5563 | 4.5563 | 12.5188 |
| 128 | original | expanded_seed790714 | 2.6559 / 3.9531 | 3.9531 | 12.0604 |
| 128 | original | expanded_seed790715 | 2.6697 / 4.0303 | 4.0303 | 11.7758 |
| 256 | fresh | expanded_seed790714 | 3.4792 / 4.56 | 4.56 | 13.2836 |
| 256 | fresh | expanded_seed790715 | 3.4726 / 4.5557 | 4.5557 | 13.733 |
| 256 | original | expanded_seed790714 | 2.6559 / 3.9531 | 3.9531 | 12.7622 |
| 256 | original | expanded_seed790715 | 2.6697 / 4.0304 | 4.0303 | 12.4077 |
| 512 | fresh | expanded_seed790714 | 3.4791 / 4.56 | 4.56 | 15.0555 |
| 512 | fresh | expanded_seed790715 | 3.4725 / 4.5555 | 4.5555 | 16.2106 |
| 512 | original | expanded_seed790714 | 2.6559 / 3.9531 | 3.9531 | 14.9218 |
| 512 | original | expanded_seed790715 | 2.6697 / 4.0304 | 4.0303 | 15.5772 |
| 1024 | fresh | expanded_seed790714 | 3.4791 / 4.56 | 4.56 | 28.1776 |
| 1024 | fresh | expanded_seed790715 | 3.4725 / 4.5555 | 4.5555 | 27.8552 |
| 1024 | original | expanded_seed790714 | 2.6559 / 3.9531 | 3.9531 | 28.1483 |
| 1024 | original | expanded_seed790715 | 2.6697 / 4.0304 | 4.0303 | 27.1513 |

Every FOM returns the supplied initial field exactly, then evolves a restricted initial field and interpolates later outputs where required. Host input restriction, GPU work, interpolation and complete contiguous host output are charged. The ROM returns its actual fitted initial field and charges full-field projection, fitting, evolution and readout. The FOM envelope tests solver intervals `[16, 32, 64, 128]` where no larger than the requested mesh, plus the same-grid solve. Exact duplicate choices are timed once with explicit aliases.

| Output intervals | Union full-grid current-relative target (%) | Selected ROM / FOM solver | ROM / FOM query (ms) | Paired FOM/ROM |
|---|---:|---|---:|---:|
| 64 | 5 | expanded_seed790715 / fom_dst_64 | 12.5659 / 0.509903 | 0.040695 |
| 128 | 5 | expanded_seed790714 / fom_dst_64 | 12.2602 / 0.6732 | 0.056624 |
| 256 | 5 | expanded_seed790714 / fom_dst_64 | 13.2078 / 1.16413 | 0.087957 |
| 512 | 5 | expanded_seed790714 / fom_dst_16 | 14.9861 / 4.51582 | 0.29134 |
| 1024 | 5 | expanded_seed790715 / fom_dst_16 | 27.733 / 21.4106 | 0.76673 |

At the union-cohort 5% target with empirical reference adjustment, a head qualifies on 5 of 5 meshes and beats the tested FOM envelope on 0. The full-output contract limits how much spatial compression alone can save; the native figure separates input, device and output cost components. This study measures frozen-weight mesh transfer, not per-resolution retraining.

The native audit checks 1152 invocations and independently reproduces FOM propagation/interpolation to `6.446909e-16` relative disagreement, and spectral reference trajectories to `1.104404e-15`. Root recomputes every saved norm and checks source draws, frozen weights, repetitions and the initial-output policy, maximum metric difference `0`. There are 0 nonstationary selected initial fits and 0 nonstationary evolution steps across repetitions. The observed spectral reference discrepancy is `1.628109e-12`, with no rigorous bound. Both cohorts, all candidates, raw timing arrays and outliers remain in the native audit.

[Complete wider heat-transfer findings](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/HEAT-TRANSFER-NOTES.md). [Heat accuracy and cost across resolution](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/heat-transfer.png). [Heat query cost components](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/transfer04/analysis/heat-transfer-components.png).

## Next experiments justified by the diagnostics

- **Burgers:** larger steps partly reduce cost but also expose iteration caps, especially at startup. A later bounded startup-solve or time-formula control should separate those effects before further architectural changes. The existing latent extrapolator is already present; adding it again is not a new intervention.
- **Heat:** both frozen expanded-coverage endpoints now pass the wider mesh ladder and fresh inputs in the restricted family. Complete output and initialization costs still matter. Broader inputs, tighter accuracy and separately measured per-resolution training are the remaining scientific controls; any runtime change must keep the charged coarse-FOM envelope.
- **Poisson:** relative loss improves the worst development sources, and field-based initialization gives a modest cost reduction. Equivalent source projections give no consistent gain. The remaining accuracy floor and nonlinear solve cost require further representation or solver controls; expanded coverage at matched updates is not uniformly beneficial.
- **Waves:** the larger nonlinear heads greatly improve reflective accuracy but remain costly and leave late absorbing error. Separate constant-test and initial-moment interventions are justified as later controls by the conservation diagnostic, without assuming they solve the full problem. A coarse-FOM resolution envelope is still needed before promoting any raw linear-control timing ratio.

These are directions for a later bounded round; every GPU job in this report is complete. The complete study still needs wider mesh ladders for the remaining PDEs, separately labeled per-resolution training, independent data/training repeats, validation-selected settings and sealed final evaluation.

## Visual artifacts

[Heat accuracy and complete-query cost](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign/heat-pilot01-accuracy-cost.png). [Reflective wave evolving](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign/dirichlet-wave-evolution-case1.png). [Absorbing wave evolving](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign/absorbing-wave-evolution-case1.png).

The wave still sequences show reference displacement, predicted displacement and absolute difference with fixed scales. They use saved fields from an actual finer-mesh ROM solve; the display resolution is labeled. Displacement pictures do not replace the velocity, energy or vanishing-field diagnostics. PDF exports and figure source/provenance are beside the images.

[Larger-head reflective wave evolving](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign/dirichlet-wave-k32-evolution-case1.png). [Larger-head absorbing wave evolving](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign/absorbing-wave-k32-evolution-case1.png). These updated sequences compare the reference, the earlier smaller head and the first declared larger-head seed within the same audited run. The same development-case index is used as in the original still sequences. Small displacement discrepancies can coexist with larger velocity or gradient-based energy errors; fixed scales can also hide late absorbing relative error.

## Reproduction

Run `/home/tahmid/Dev/.venv/bin/python reports/generate_multiresolution_pilots.py` from a checkout with the recorded experiment worktrees. The adjacent JSON manifest identifies every source artifact by content hash. All numerical table values are generated; none are hand-entered.

## Plain-language glossary

- **PDE / FOM / NM-ROM / ROM:** partial differential equation / full spatial solver / nonlinear-manifold reduced solver / reduced solver.
- **Frozen / intervals / cases:** unchanged network weights / grid cells along one axis / distinct physical inputs. Timing repetitions are not additional cases.
- **ROM setting / dt / CN / stall / tau:** chosen solver configuration / time step / Crank–Nicolson time formula / relative improvement stopping rule / requested weak-residual reduction.
- **Current L2 / initial L2 / steady L2:** field error divided by the current reference norm / initial reference norm / steady reference solution norm.
- **Initial wave state / energy-state error:** displacement normalized by its initial L2 norm and velocity/energy error normalized by the initial phase-energy scale / error combining displacement gradients with velocity.
- **Common observation grid / same-grid norm:** fixed physical sampling locations shared by different query resolutions / error measured over the full grid at the stated resolution. These measures need not coincide.
- **Median / worst / query ms / paired ratio:** middle case result / largest case error / complete input-to-output milliseconds / per-case FOM time divided by ROM time, then a cohort median.
- **Bank / head / latent / weak test mode:** learned spatial features / their nonlinear coefficient map / compressed state coordinates / smooth function averaging the PDE equation.
- **Cold start / stationary / budget exit:** initial reduced-state fitting / meeting a local derivative convergence check / exhausting allowed iterations.
- **DST / coarse-grid envelope:** direct sine-transform solution / least-cost qualifying full solve allowing fewer cells and charging interpolation to the requested output.
- **Reference refinement / uncertainty / qualification:** comparing finer trusted solves / remaining reference error / meeting accuracy and numerical-validity requirements with that uncertainty included.
- **Development / validation / sealed final / provisional:** preliminary experiment data / data used to select settings / untouched independent confirmation data / evidence with the stated limitations.
- **Compiled / segmented / projection / compression:** one prepared executable / separately launched stages / representing a field in a spatial span / constraining that representation through fewer latent coordinates.
- **Strict / relaxed / gradient tolerance / field drift:** more demanding stopping control / less demanding stopping setting / required smallness of the objective derivative / change from the strict-control trajectory divided by the current reference norm.
- **Paired improvement / requested modes / retained modes:** strict-control time divided by changed-method time, summarized across cases / desired minimum number of weak tests / actual number when tied sine eigenmodes are retained together.
- **Edge Gram / edge QR / midpoint / Gauss / full-grid QR:** fitting on grid-dependent edge-inclusive samples using normal equations / a stable factorization on those same samples / fixed equally spaced physical midpoints / fixed weighted Gaussian quadrature points / least-squares fitting using every grid value.
- **Bank projection floor / best-recorded fit / Richardson estimate:** smallest error possible in the unrestricted linear feature span / best fit found by the tested nonlinear searches, without proving optimality / remaining reference error estimated by assuming observed refinement rates continue.
- **Envelope / additive empirical margin / passed reference budget:** least-cost tested method meeting the development target / estimated reference error added to measured ROM or FOM error / that estimate also lies below the predeclared fraction of the target. Empirical passage is not a rigorous certificate.
- **Commit / manifest / checksum:** saved source revision / inventory of source artifacts / content fingerprint checking exact file bytes.
- **Endpoint / minibatch seed / per-snapshot relative squared-error loss:** saved weights after the declared training budget / random seed selecting training examples per update / squared reconstruction error divided by that snapshot's squared field norm.
- **Initialization library / gate / factorial:** stored training codes used to start a solve / a predeclared requirement for continuing an experiment / crossing independently varied choices to distinguish their effects.
- **Gauss–Jordan / backward error / fallback / solver counter / replay:** elimination for the small latent linear system / residual of the computed linear solution relative to its data / guarded use of the original solver / count of optimization steps or evaluations / instrumented recomputation of a saved solve.
- **Affine / phase dimension / tangent velocity / normal force / curvature:** linear map plus a constant offset / displacement and velocity coordinate count / velocity representable by local decoder derivatives / weak acceleration outside those derivative directions / acceleration contributed by the bending decoder map.
- **Moment / invariant / constant test / drift:** weighted global combination of displacement and velocity / quantity the discrete equations preserve / spatially constant function used to test those equations / change from a method's own initial value.
- **First-step budget exit / predictor:** iteration limit reached at the first evolution step / proposed next latent state extrapolated from earlier states and checked by the weak residual.
- **Timing outlier:** a repetition longer than twice its own case/configuration median in the latest-results table. It remains in all reported calculations.
