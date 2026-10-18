# Replaying the earlier Burgers tensor and Poisson comparison

This report contains new GPU measurements under the user's requested historical device-resident protocol. Results are provisional development evidence from the historical cohorts; they are not independent final paper confirmation or replacements for the separately scoped modern benchmark.

Under this protocol, the Burgers tensor beats its named FOM at nodes per axis 512, 1024. Poisson QF beats the original CG comparator at nodes per axis 128, 256, 512, 1024. The direct Poisson solver remains faster at every tested mesh.

The input field starts on the GPU. The timer includes initialization or source projection, solution and full-field reconstruction on the GPU. Poisson returns all interior unknowns with prescribed zero boundary values; Burgers returns full grid snapshots. Host transfers, offline training/setup and compilation are outside this primary timer. Both solvers use the same discrete mesh; no coarse-grid FOM envelope is substituted. The [replay protocol](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/HISTORICAL-COMPARISON-REPLAY.md) records this user-selected change in comparison scope.

The restored families use single Gaussian peaks with varying position, width and amplitude. Burgers evolves a scalar viscous field; Poisson solves a steady constant-coefficient equation. Both use unit-square grids with zero Dirichlet boundary values. Inputs to the ROM are supplied fields, with viscosity also supplied for Burgers; the Gaussian generating parameters are not substituted for field-based initialization.

![Device-query runtime and relative-error scaling](<2026-09-10-historical-burgers-poisson-replay-scaling.png>)

[Download the scaling figure as PDF](<2026-09-10-historical-burgers-poisson-replay-scaling.pdf>). Each PDE has its own GPU allocation, named below; compare scaling within each panel. The lower panels retain the distinct error summaries defined in their tables. Lines connect measured meshes and do not imply an asymptotic fit or error bound.

## Burgers tensor and original same-grid FOM

Every mesh runs in allocation 3492130 on NVIDIA A100 80GB PCIe, using its frozen historical checkpoint and the restored trained sampled-field initializer. The original tensor, sampled and full residual controls are retained in the raw results. The table uses the tensor's paired complete-query measurements against the original dense-sine-preconditioned, tolerance-stopped Newton FOM, on 8 historical test trajectories.

The original rollout uses 50 steps of size 0.005 to final time 0.25 and reconstructs every step, including the initial field.

| Nodes/axis | Latent solve ms | ROM / FOM query ms | FOM/ROM | ROM mean / median case / worst error (%) | FOM mean error (%) | Newton / linear tolerance |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 27.695 | 29.402 / 12.366 | 0.421 | 2.309 / 2.379 / 17.911 | 0.037 | 0.003 / 0.0015 |
| 256 | 27.394 | 29.638 / 17.224 | 0.581 | 2.451 / 2.536 / 16.398 | 0.035 | 0.003 / 0.0015 |
| 512 | 27.856 | 29.634 / 40.158 | 1.355 | 2.456 / 2.566 / 15.578 | 0.035 | 0.003 / 0.0015 |
| 1024 | 26.713 | 33.815 / 118.731 | 3.511 | 6.381 / 3.740 / 33.725 | 2.975 | 0.01 / 0.005 |

Times are cohort medians of per-trajectory repetition medians, and FOM/ROM divides those medians. The latent-solve column is the original pooled median of evolution-only repetitions; it excludes initialization and decoding and must not be substituted for complete-query cost. Errors use the current-time same-grid reference norm. Mean averages over trajectories and times, median case is the median of time-averaged trajectory errors, and worst is the maximum over every trajectory and time. The FOM tolerance is selected using cohort mean accuracy, not a worst-case physical-error guarantee.

The polynomial tensor matches the backward-upwind operator algebraically. Its equivalence to sign-dependent upwinding depends on decoded signs; the retained native gates and undershoot measurements remain part of its scope. Loading the archived trained checkpoint replaces the original in-job training at the intermediate mesh. The initializer is still fitted offline from the recorded seed.

| Nodes/axis | Tensor step stopping reasons | Blowups |
| --- | --- | --- |
| 64 | stalled: 400 | 0 |
| 256 | stalled: 400 | 0 |
| 512 | stalled: 400 | 0 |
| 1024 | stalled: 400 | 0 |

The historical stall stopping rule is retained. A stall exit means the configured progress criterion stopped the iteration; it does not certify a stationary latent minimizer. The reported accuracy belongs to the fields returned at those stops.

## Poisson quadrature-free solve and original CG baseline

Every mesh runs in allocation 3492398 on NVIDIA A100-PCIE-40GB. This table uses the preselected stopping threshold `tau=0.001` and 16 held-out sources from the historical seed. The unpreconditioned CG row is the fastest recorded same-grid tolerance with mean error no larger than the ROM's. The original direct spectral solver is separately named in the same table.

| Nodes/axis | QF ROM ms | CG ms | CG/ROM | Direct ms | ROM mean / median / worst error (%) | CG tol / mean error (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 3.268 | 4.057 | 1.241 | 0.123 | 3.063 / 2.252 / 8.080 | 0.1 / 1.726 |
| 256 | 2.971 | 8.458 | 2.847 | 0.136 | 3.141 / 2.625 / 7.586 | 0.1 / 1.065 |
| 512 | 3.217 | 25.688 | 7.984 | 0.218 | 3.154 / 2.528 / 8.057 | 0.1 / 0.710 |
| 1024 (extension) | 4.005 | 101.342 | 25.302 | 0.722 | 3.476 / 2.900 / 8.240 | 0.1 / 0.491 |

The largest quadrature-free mesh is an extension: the historical QF ladder stopped earlier, while the old largest-mesh CG comparison used the sampled cached decoder. A ROM win against CG is a claim about that iterative baseline; the direct column records the available alternative on this constant-coefficient rectangular problem. No cross-job old/new timing ratio is used to claim an algorithm improvement.

The full-grid and QF paths use the same frozen decoder, initialization, trust-LM solver and weak equation. The QF path evaluates a preassembled reduced matrix; the full path evaluates its mathematically equivalent full-grid contraction. Empirical quadrature fitting is omitted from this replay. The additional threshold and historical fresh-seed cohort are retained below rather than selecting a headline configuration after seeing timings.

| Nodes/axis | Cohort | Stopping threshold | Full / QF query ms | Mean / worst QF error (%) | Censored (%) |
| --- | --- | --- | --- | --- | --- |
| 128 | heldout_seed0 | 0.001 | 4.445 / 3.268 | 3.063 / 8.080 | 100.0 |
| 128 | fresh_seed1 | 0.001 | 4.873 / 3.631 | 2.892 / 15.768 | 100.0 |
| 128 | heldout_seed0 | 0.01 | 4.023 / 2.887 | 3.114 / 8.080 | 68.8 |
| 128 | fresh_seed1 | 0.01 | 4.213 / 3.116 | 2.925 / 15.768 | 68.8 |
| 256 | heldout_seed0 | 0.001 | 6.185 / 2.971 | 3.141 / 7.586 | 100.0 |
| 256 | fresh_seed777 | 0.001 | 6.509 / 2.997 | 3.807 / 9.483 | 100.0 |
| 256 | heldout_seed0 | 0.01 | 5.429 / 2.529 | 3.204 / 7.586 | 62.5 |
| 256 | fresh_seed777 | 0.01 | 5.848 / 2.827 | 3.855 / 9.467 | 68.8 |
| 512 | heldout_seed0 | 0.001 | 15.419 / 3.217 | 3.154 / 8.057 | 100.0 |
| 512 | fresh_seed20260823 | 0.001 | 14.703 / 2.982 | 4.597 / 11.361 | 100.0 |
| 512 | heldout_seed0 | 0.01 | 13.443 / 2.848 | 3.227 / 8.057 | 68.8 |
| 512 | fresh_seed20260823 | 0.01 | 13.812 / 2.728 | 4.625 / 11.361 | 68.8 |
| 1024 | heldout_seed0 | 0.001 | 56.003 / 4.005 | 3.476 / 8.240 | 100.0 |
| 1024 | fresh_seed20260823 | 0.001 | 59.417 / 4.144 | 5.057 / 12.268 | 100.0 |
| 1024 | heldout_seed0 | 0.01 | 47.464 / 3.528 | 3.500 / 8.240 | 68.8 |
| 1024 | fresh_seed20260823 | 0.01 | 49.936 / 3.709 | 5.074 / 12.268 | 75.0 |

Censored means the solver stopped without satisfying its accepted convergence reason. These runs still return fields with measured errors; censoring is not a convergence certificate and is retained explicitly.

## Measurement audit and limits

The initial Poisson attempt completed the smaller meshes but stopped before timing the largest mesh: it inherited the smaller-grid reference-residual guard. The archived largest-grid experiment explicitly used a different guard. That recorded setting was restored, an independent CG/direct field-agreement check was added, and the entire ladder was rerun in one allocation. Only that complete rerun enters these tables; the initial attempt remains in the owner's diagnostic archive. The reference CG solver tolerance itself was unchanged.

| Poisson nodes/axis | Recorded / NumPy reference residual | Historical residual limit | Independent CG/direct field discrepancy |
| --- | --- | --- | --- |
| 128 | 2.171e-12 / 2.171e-12 | 1.0e-10 | See measured direct error in source rows |
| 256 | 1.245e-11 / 1.245e-11 | 1.0e-10 | See measured direct error in source rows |
| 512 | 6.963e-11 / 6.963e-11 | 1.0e-10 | See measured direct error in source rows |
| 1024 | 3.884e-10 / 3.884e-10 | 1.0e-09 | 1.136e-14 |

The generator checked 456 timing/error identities and 60 groups of full captured fields. Poisson errors were independently recomputed from the last timed output of every method and source, and a separate NumPy stencil calculation checked the reference equation against every saved supplied source. Burgers full-grid reconstruction checks cover the predeclared first trajectory at every mesh; remaining trajectories retain native full-grid errors and sampled output fields, so an independent full-grid audit of the entire Burgers cohort is not claimed.

The following accuracy comparison checks reproduction of the archived configuration. Absolute differences use the dimensionless relative-error fraction; the evidence JSON additionally retains every Burgers time/case error's maximum departure from its archive. No archived largest-mesh Poisson QF result exists to compare with its extension.

| PDE / mesh | Archived mean error (%) | Replay mean error (%) | Absolute mean-error difference |
| --- | --- | --- | --- |
| Burgers / 64 | 2.30941423 | 2.30941423 | 9.085e-13 |
| Burgers / 256 | 2.45121445 | 2.45121445 | 1.514e-14 |
| Burgers / 512 | 2.45623504 | 2.45623504 | 4.348e-14 |
| Burgers / 1024 | 6.38139030 | 6.38143104 | 4.074e-07 |
| Poisson QF / 128 | 3.06261967 | 3.06261967 | 1.735e-17 |
| Poisson QF / 256 | 3.14149303 | 3.14149303 | 7.085e-13 |
| Poisson QF / 512 | 3.15380739 | 3.15380739 | 1.132e-14 |

Every outlier below exceeds twice its own case's repetition median. All repetitions remain in the aggregates. Poisson's three counts are ROM / selected CG / direct; Burgers has ROM / paired FOM.

| PDE / mesh / cohort / threshold | ROM / FOM outliers | Repetitions per method |
| --- | --- | --- |
| Burgers / 64 | 0 / 0 | 24 |
| Burgers / 256 | 0 / 0 | 24 |
| Burgers / 512 | 0 / 0 | 24 |
| Burgers / 1024 | 0 / 0 | 24 |
| Poisson / 128 / heldout_seed0 / 0.001 | 0 / 0 / 0 | 192 |
| Poisson / 128 / fresh_seed1 / 0.001 | 0 / 0 / 0 | 192 |
| Poisson / 128 / heldout_seed0 / 0.01 | 0 / 0 / 0 | 192 |
| Poisson / 128 / fresh_seed1 / 0.01 | 0 / 0 / 0 | 192 |
| Poisson / 256 / heldout_seed0 / 0.001 | 0 / 0 / 0 | 192 |
| Poisson / 256 / fresh_seed777 / 0.001 | 0 / 0 / 0 | 192 |
| Poisson / 256 / heldout_seed0 / 0.01 | 0 / 0 / 0 | 192 |
| Poisson / 256 / fresh_seed777 / 0.01 | 0 / 0 / 0 | 192 |
| Poisson / 512 / heldout_seed0 / 0.001 | 0 / 0 / 0 | 192 |
| Poisson / 512 / fresh_seed20260823 / 0.001 | 0 / 0 / 0 | 192 |
| Poisson / 512 / heldout_seed0 / 0.01 | 0 / 0 / 0 | 192 |
| Poisson / 512 / fresh_seed20260823 / 0.01 | 0 / 0 / 0 | 192 |
| Poisson / 1024 / heldout_seed0 / 0.001 | 0 / 0 / 0 | 192 |
| Poisson / 1024 / fresh_seed20260823 / 0.001 | 0 / 0 / 0 | 192 |
| Poisson / 1024 / heldout_seed0 / 0.01 | 0 / 0 / 0 | 192 |
| Poisson / 1024 / fresh_seed20260823 / 0.01 | 0 / 0 / 0 | 192 |

Mesh-specific historical checkpoints are restored; this is not frozen-weight mesh transfer. GPU backend, double precision, source/checkpoint hashes, stopping records and raw timings are retained with the owner artifacts. Different historical hardware and instrumentation prevent interpreting an absolute runtime difference from the archive as an isolated numerical-method change. The [historical audit](2026-09-10-historical-poisson-and-burgers-cost-audit.md) documents those prior comparisons and the modern protocol separately.

| PDE / nodes per axis | Latent coordinates | Spatial bank size | Weak modes | Archived training steps / maximum snapshots |
| --- | --- | --- | --- | --- |
| Burgers / 64 | 16 | 64 | 64 | 40000 / 8192 |
| Burgers / 256 | 16 | 64 | 64 | 60000 / 8192 |
| Burgers / 512 | 16 | 64 | 64 | 60000 / 8192 |
| Burgers / 1024 | 16 | 64 | 64 | 40000 / 2048 |
| Poisson / 128 | 16 | 96 | 64 | Frozen checkpoint; see source configuration |
| Poisson / 256 | 16 | 64 | 64 | Frozen checkpoint; see source configuration |
| Poisson / 512 | 16 | 64 | 64 | Frozen checkpoint; see source configuration |
| Poisson / 1024 | 16 | 64 | 64 | Frozen checkpoint; see source configuration |

The checkpoint training settings change across the ladder. An accuracy change between meshes therefore cannot be attributed to resolution alone. The replay performs no new spatial-bank or nonlinear-head training.

- [worktrees/2026-09-07-mr-burgers2d/experiments/historical-burgers2d-replay/runs/historical_20260910_r1/out/n64/sep_b2d_tensor_n64.json](../worktrees/2026-09-07-mr-burgers2d/experiments/historical-burgers2d-replay/runs/historical_20260910_r1/out/n64/sep_b2d_tensor_n64.json)
- [worktrees/2026-09-07-mr-burgers2d/experiments/historical-burgers2d-replay/runs/historical_20260910_r1/out/n256/sep_b2d_tensor_n256.json](../worktrees/2026-09-07-mr-burgers2d/experiments/historical-burgers2d-replay/runs/historical_20260910_r1/out/n256/sep_b2d_tensor_n256.json)
- [worktrees/2026-09-07-mr-burgers2d/experiments/historical-burgers2d-replay/runs/historical_20260910_r1/out/n512/sep_b2d_tensor_n512.json](../worktrees/2026-09-07-mr-burgers2d/experiments/historical-burgers2d-replay/runs/historical_20260910_r1/out/n512/sep_b2d_tensor_n512.json)
- [worktrees/2026-09-07-mr-burgers2d/experiments/historical-burgers2d-replay/runs/historical_20260910_r1/out/n1024/sep_b2d_tensor_n1024.json](../worktrees/2026-09-07-mr-burgers2d/experiments/historical-burgers2d-replay/runs/historical_20260910_r1/out/n1024/sep_b2d_tensor_n1024.json)
- [worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay01/attempt-status.json](../worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay01/attempt-status.json)
- [worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay02/out/poisson_n128.json](../worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay02/out/poisson_n128.json)
- [worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay02/out/poisson_n256.json](../worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay02/out/poisson_n256.json)
- [worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay02/out/poisson_n512.json](../worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay02/out/poisson_n512.json)
- [worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay02/out/poisson_n1024.json](../worktrees/2026-09-07-mr-poisson2d/experiments/historical-poisson-replay/runs/replay02/out/poisson_n1024.json)

## Glossary

- **N / nodes per axis:** grid points including boundary nodes; the interior has fewer points along each axis.
- **Gaussian peak / Dirichlet boundary:** a localized bell-shaped field / a prescribed field value at the boundary, zero here.
- **ROM:** reduced-order model, solving for latent coordinates and reconstructing the field.
- **FOM:** full-order model, solving the discretized field equations.
- **Tensor:** preassembled quadratic coefficients used for Burgers advection.
- **Latent solve:** reduced time evolution after initialization, before field decoding.
- **Query / ms:** the declared input-to-output computation and its elapsed milliseconds.
- **FOM/ROM or CG/ROM:** baseline time divided by ROM time; values above one favor the ROM against that named baseline.
- **QF:** quadrature-free; the repeated weak equation uses an exact preassembled reduced matrix.
- **CG:** conjugate gradient, the historical iterative Poisson solver.
- **Unpreconditioned:** CG uses the original equation directly, without an auxiliary solver transforming its conditioning.
- **Direct:** the original dense sine-transform Poisson solve.
- **Newton / linear tolerance:** stopping settings for Burgers nonlinear corrections and their linear subproblems.
- **Weak residual:** the PDE mismatch projected onto smooth test functions.
- **Trust-LM:** the historical trust-limited Levenberg–Marquardt nonlinear fitting algorithm.
- **Threshold / tau:** the configured stopping threshold, not a bound on solution error.
- **Mean / median / worst error:** specified summaries of relative field errors over the recorded cases and, for Burgers, times.
- **Cohort / held-out / fresh seed:** the named set of sources; held-out sources were excluded from fitting, while the historical fresh cohort uses its separately recorded seed. Neither is newly sealed final evidence in this replay.
- **Censored:** a solve terminating outside the accepted convergence reasons.
- **Stalled / blowup:** a solve stopped by its configured lack-of-progress rule / a trajectory returning nonfinite fields before completion.
- **Checkpoint / initializer:** saved trained weights / the procedure producing the initial latent guess.
- **Latent coordinates / spatial bank size / weak modes:** the number of solved reduced variables / learned spatial features / projected test equations.
- **Training steps / maximum snapshots:** archived optimizer updates / the cap on training fields, not work charged to a query in this replay.
- **Outlier / repetitions:** a slow recorded invocation under the stated threshold / repeated timings of the same query.
- **Absolute mean-error difference:** the magnitude of replay mean error minus archived mean error, expressed as a fraction rather than a percentage.
- **Reference residual / limit:** the discretized equation mismatch for the reference field / its recorded acceptance threshold.
- **CG/direct field discrepancy:** the relative difference between independently computed iterative and direct reference solutions.
- **Allocation:** a single scheduled GPU job, keeping the mesh ladder on one device.
