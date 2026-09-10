# Historical CP heat optimizations and transfer to the current NMROM

Recovered source and archived July timing records, reviewed in September. The table is a recomputation of saved timing/error summaries, not a new GPU experiment or independent full-field validation; all rows remain historical and provisional for present-day paper claims.

## Sources recovered

The local NeurIPS directory contains a later heat cost study, reduced-Gram implementations and raw timing repetitions. The original April/May project also remains on the cluster, including the May heat repair branch and session records. Selected sources were copied into the current heat worktree with verified hashes; originals were preserved.

[Preserved source inventory](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/SOURCE-MANIFEST.json) · [July cost-study chronology](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/neurips/HEAT_ROUNDS.md) · [May heat diagnosis](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/may/2026-05-21-heat/DIAGNOSIS.md) · [May training and quadrature follow-up](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/may/claude-lab/sessions/2026-05-26.tex)

The May notes contain successive corrections, changing diagnoses and accuracy-focused training experiments; their prose is historical evidence, not an independently verified explanation of the current model. The July study is particularly useful because it retains paired implementation controls and timing arrays. The original large paper speedups were subsequently corrected; this review does not reinstate them.

## Recomputed historical records

| PDE | N | Variant | Median ROM ms | Median FOM ms | Median per-case speedup | Mean final error (%) | Worst final error (%) | Faster cases | ROM / FOM timing outliers |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heat2d | 128 | `recompiled_acc_red_r8A_redgraph` | 35.788390 | 38.081696 | 1.079566 | 0.863504 | 2.036324 | 5 / 10 | 0 / 0 |
| heat2d | 128 | `recompiled_acc_v4w_r8A_redgraph` | 213.057889 | 38.081696 | 0.258782 | 1.091972 | 2.024527 | 0 / 10 | 0 / 0 |
| heat2d | 128 | `recompiled_fast_red_r8A_redgraph` | 31.352537 | 38.274756 | 1.214041 | 1.274771 | 3.036263 | 5 / 10 | 0 / 0 |
| heat2d | 128 | `recompiled_fast_v4w_r8A_redgraph` | 95.068297 | 38.274756 | 0.577599 | 1.443244 | 3.025890 | 1 / 10 | 0 / 0 |
| heat3d | 128 | `recompiled_acc_v5w_r8B_graph` | 29.340512 | 121.742347 | 4.116510 | 18.320561 | 37.893239 | 10 / 10 | 0 / 0 |

Provisional historical evidence: every row reports a shared node rather than an exclusive node. Errors are final-state errors averaged across cases in the original headline, not the current worst-over-all-requested-times metric. The FOM is the compiled iterative CG time integrator used in that study, not the current direct spectral/coarse FOM. Reduced variants also use a different precision arrangement from the present all-float64 protocol. This prevents transplanting a historical speed ratio into the current benchmark.

Per-case medians are recomputed from all stored timed repetitions: the measurement source performs warmups separately before storing these arrays. The speedup is the median of per-case FOM/ROM ratios, not the ratio of the two cohort median times. Timing outliers above one-and-a-half times their own case/method median are counted here and retained. Error scalars are checked against their stored per-case arrays; full fields and per-repetition error arrays were not independently validated in this review.

## What actually transfers

| Mechanism | Historical implementation | Current heat status | Next controlled comparison |
| --- | --- | --- | --- |
| Precompute spatial algebra | CP basis or quadrature stencils are fixed offline; reduced variants contract the normal equations into the coefficient space | Exact weak spatial operator is already precomputed; online residuals already avoid the full grid | Compare fully contracted normal equations against the existing small weak-residual assembly |
| Small positive-definite solve | Cholesky replaces the general LU solve | Current LM uses a general linear solve | Replace only the damped-system factorization and check fields, gradients and stopping reasons |
| Compile the full query | Including the formerly eager encoder produced a large gain in some archived controls | Current input fit, latent rollout and readout are already compiled | Preserve this; there is no missing Python-loop fix to claim |
| Graph execution and counted loops | Compiler flag and fixed-count loops helped selected configurations and hurt others that converged early | No matched graph/counted-loop study in this current branch | Test only after the algebra comparison, applying compiler settings to both FOM and ROM |
| Learned initialization and explicit amplitude | Encoder supplies a latent guess; solver evolves latent plus scale | Current model fits initial coordinates and has no evolving scale | Separate architecture/training follow-up; not required to test Cholesky or contracted algebra |

The previous source review was incomplete: it identified the encoder and amplitude differences but omitted the later reduced-Gram, Cholesky and graph studies. These provide a more direct implementation comparison before changing the time integrator or retraining the decoder.

## The current weak objective can be retained

For the existing weak solve, write $r(z)=B h(z)-b$, with fixed bank-to-test matrix $B$. Precompute $S=B^\top B$ and, once per target, $c=B^\top b$. If $D=\partial h/\partial z$, then

$$J^\top J=D^\top S D,\qquad J^\top r=D^\top(S h-c),$$

$$\|r\|_2^2=h^\top S h-2h^\top c+b^\top b.$$

The existing target normalization must be carried through every expression. This is algebraic contraction of the same weak objective, not the archived strong-form residual or a free-coefficient linear ROM. It preserves the existing nonlinear decoder, time steps and stopping tolerance mathematically. Numerical parity must still be measured: cancellation near a small residual and changes in rounding can alter an acceptance decision.

Our current weak residual is already small and independent of the full mesh size. Thus the old reduction from a grid-sized Jacobian is already largely achieved here; the remaining contraction could reduce the constant cost, but its gain is not established. Cholesky is the simplest missing change to isolate first. No current accuracy or speed improvement is claimed for either transfer yet.

## Provenance

- [`recompiled_acc_red_r8A_redgraph`](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/neurips/share-bundle/results/raw/heat2d_n128_nm_rom_recompiled_acc_red_r8A_redgraph_20260730T065419Z.json): job `1994862`, node `pax010`, originating summary `r8A_redgraph_h2d_n128_acc_20260730T062534Z_1994862.json`; source and array aggregates checked, historical only.
- [`recompiled_acc_v4w_r8A_redgraph`](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/neurips/share-bundle/results/raw/heat2d_n128_nm_rom_recompiled_acc_v4w_r8A_redgraph_20260730T065419Z.json): job `1994862`, node `pax010`, originating summary `r8A_redgraph_h2d_n128_acc_20260730T062534Z_1994862.json`; source and array aggregates checked, historical only.
- [`recompiled_fast_red_r8A_redgraph`](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/neurips/share-bundle/results/raw/heat2d_n128_nm_rom_recompiled_fast_red_r8A_redgraph_20260730T065419Z.json): job `1994862`, node `pax010`, originating summary `r8A_redgraph_h2d_n128_fast_20260730T062732Z_1994862.json`; source and array aggregates checked, historical only.
- [`recompiled_fast_v4w_r8A_redgraph`](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/neurips/share-bundle/results/raw/heat2d_n128_nm_rom_recompiled_fast_v4w_r8A_redgraph_20260730T065419Z.json): job `1994862`, node `pax010`, originating summary `r8A_redgraph_h2d_n128_fast_20260730T062732Z_1994862.json`; source and array aggregates checked, historical only.
- [`recompiled_acc_v5w_r8B_graph`](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910/neurips/share-bundle/results/raw/heat3d_n128_nm_rom_recompiled_acc_v5w_r8B_graph_20260730T065419Z.json): job `1994862`, node `pax010`, originating summary `r8B_graph_h3d_n128_acc_20260730T063607Z_1994862.json`; source and array aggregates checked, historical only.

## Glossary

- **CP / NMROM / FOM:** tensor-product decoder / nonlinear-manifold reduced model / full-grid numerical solver.
- **N / PDE:** archived grid size per axis / partial differential equation.
- **Variant:** archived implementation label; `acc` and `fast` identify the original operating settings, not a new success classification.
- **red / v4w / v5w:** contracted coefficient-space normal equations / compiled dense comparison with Cholesky / minimal Cholesky-only solve change alongside the compiled encoder.
- **Median ROM/FOM time:** median of case-level median query times, in milliseconds.
- **Median per-case speedup / faster cases:** median of each case's FOM time divided by its ROM time / number of cases whose ratio exceeds unity.
- **Mean or worst final error:** average or maximum over stored final-state relative errors across cases; neither measures the entire time history.
- **Timing outlier:** stored repetition above the declared within-case threshold; not removed.
- **Gram / normal equations / Jacobian:** precomputed inner-product matrix / small linear system for a nonlinear least-squares update / derivative matrix.
- **Cholesky / LU / positive-definite:** specialized matrix factorization / general matrix factorization / property ensured here by the damped least-squares system in exact arithmetic.
- **Weak objective / coefficient space:** residual tested against smooth functions / coordinates multiplying the fixed spatial bank.
- **LM / CG:** Levenberg–Marquardt nonlinear least squares / conjugate-gradient linear solver.
- **Graph / counted loop:** compiled GPU execution mechanism / loop with a fixed iteration count, potentially doing unnecessary work after convergence.
- **Encoder / amplitude:** learned initial-coordinate map / separate scalar scaling the decoded field.
- **Shared node / parity:** machine also hosting other jobs / agreement of two implementation paths.
