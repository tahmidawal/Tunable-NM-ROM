# Reconciling the earlier tensor/Poisson speedups with the current campaign

This audit traces the user's [Burgers–Poisson tensor tables](2026-09-03-burgers-poisson-tensor-tables.md) to archived run JSONs and scientific source, then compares their protocol with the current multiresolution campaign. Historical measurements remain scoped to their original checkpoints, cohorts, hardware and timers; current measurements remain provisional development evidence. No new numerical experiment was run.

**The user's recollection is supported.** Cached reduced computation was approximately flat in the earlier studies, and Burgers had a same-job tensor speedup against its then-selected FOM. The current Burgers table does not test that tensor configuration. Poisson retains quadrature-free reduced algebra, but now competes primarily with a direct solver and uses a broader query/accuracy protocol. The prior explanation that a nonlinear ROM solves a harder coordinate problem was incomplete as an explanation of the change: nonlinear solving was present in the successful historical configurations too.

## What the earlier tables actually measured

### Burgers: the tensor speedup exists in the retained records

| Old N (nodes/axis) | GPU | Tensor solve ms | Paired tensor / FOM ms | Historical FOM/ROM | Mean / worst time-case error (%) |
| --- | --- | --- | --- | --- | --- |
| 64 | NVIDIA A100-PCIE-40GB | 27.54 | 29.57 / 12.37 | 0.42 | 2.309 / 17.911 |
| 256 | NVIDIA A100 80GB PCIe | 27.70 | 29.53 / 17.24 | 0.58 | 2.451 / 16.398 |
| 512 | NVIDIA H200 | 19.97 | 21.50 / 21.01 | 0.98 | 2.456 / 15.578 |
| 1024 | NVIDIA A100 80GB PCIe | 27.13 | 34.11 / 117.69 | 3.45 | 6.381 / 33.725 |

The solve column is the pooled median of retained latent-evolution timings. The paired columns are medians of trajectory repetition medians from alternating ROM/FOM measurements. The historical ratio divides those two cohort medians; it is not the current campaign's median of per-case ratios. All paired repetition arrays were checked, without discarding outliers. Worst time/case error is newly summarized from the archived per-time errors; the original headline reports the mean over trajectories and times. These errors use each current-time same-grid reference norm, not the current campaign's initial-normalized refined-reference metric.

Each resolution used a different GPU and, in this ladder, a mesh-specific checkpoint. These rows do not establish a same-GPU scaling exponent or frozen-weight mesh transfer. The separately audited [Burgers 1D tensor study](2026-08-29-b1d-tensor-sample-free-burgers.md) did include a one-process resolution ladder supporting flat cached solve cost; it did not establish a win against its efficient tridiagonal FOM.

The old tensor already shared its nonlinear head, initialization and solver with the sampled control. At the largest grid the sampled control also recorded a 3.31× FOM/ROM ratio; switching from sampling to the tensor reduced its unpaired fused device-query median by 5.70%. Thus the old FOM advantage was not created solely by replacing quadrature. The tensor removed fitting/setup and preserved approximately the sampled rule's online cost.

### Poisson: constant reduced cost and a CG crossover, with a direct-solver caveat already present

| Old N (nodes/axis) | Cached sampled ROM ms | Selected CG ms | Historical CG/ROM | Mean error (%) |
| --- | --- | --- | --- | --- |
| 64 | 2.06 | 1.31 | 0.63 | 3.751 |
| 128 | 3.09 | 3.75 | 1.21 | 3.062 |
| 256 | 2.54 | 8.16 | 3.21 | 2.893 |
| 512 | 3.06 | 26.36 | 8.62 | 3.154 |
| 1024 | 3.50 | 98.31 | 28.07 | 3.476 |

This is the earlier cached sampled-ROM/CG ladder from the report, regenerated using its original source selection. The CG entry is the cheapest recorded tolerance whose mean error is no larger than the ROM's. The separate quadrature-free experiment, on frozen checkpoints, recorded:

| Old N | Full-grid residual ms | EQ residual ms | Quadrature-free residual ms |
| --- | --- | --- | --- |
| 128 | 3.98 | 3.14 | 2.93 |
| 256 | 4.96 | 2.75 | 2.68 |
| 512 | 11.11 | 3.09 | 2.87 |

All three columns here include source projection, the nonlinear solve and full interior-field decoding on the device. They are not residual-kernel-only timings despite the old table's short “solve ms” label. Their source and output transfers are outside the timer. The quadrature-free path is approximately flat; the full-grid alternative grows. Checkpoints/jobs differ across the rows, so no scaling exponent is inferred.

At old N=1024, the **same archived Poisson study already recorded a direct spectral solve of 0.639 ms**, faster than its ROM. That was a discrete same-grid direct solution and the quoted cost was a device calculation, not today's host-to-host query. The older large speedup was against unpreconditioned CG. It did not establish that the same performance carries to different PDE coefficients/geometries where a direct solver is unavailable; those problems were not tested by that table.

The old quadrature-free rows also retain their reported censored stopping status: reaching an early stopping condition is not proof of a stationary/global best fit. The earlier largest-grid CG-comparison driver measures error in a call before the timing loop and checks agreement with another call afterward; it does not capture the error from a timed invocation under today's stricter protocol. The later quadrature-free driver does capture timed outputs. This audit reproduces timing aggregates and archived error summaries; it does not rerun the PDEs or independently reconstruct their full fields.

## What changed in the current campaign

| Current PDE | Output intervals/axis | Cases | Worst physical error (%) | Host-to-host ROM / FOM ms |
| --- | --- | --- | --- | --- |
| Burgers | 1024 | 4 | 4.635 | 49.46 / 24.90 |
| Poisson | 512 | 30 | 6.802 | 4.02 / 2.27 |

These current entries have a different meaning from the historical tables above. Burgers reports the maximum initial-normalized full-grid physical error; Poisson reports the maximum steady relative physical error on its common observation grid. The selected FOM satisfies the declared empirical accuracy target, and coarse-grid solving plus charged interpolation is eligible. Current times include supplied host input and requested host output. Do not subtract old and current times and interpret the difference as a measured regression: checkpoints, jobs, cases, metrics and timing boundaries changed together.

| Item | Earlier Burgers tensor study | Current Burgers multiresolution study |
|---|---|---|
| Representation | K=16, R=64; checkpoint changes with mesh | K=16, R=512; frozen checkpoint transferred from its training mesh |
| Nonlinear advection | Precomputed quadratic tensor | Nonnegative empirical quadrature with the sign-dependent upwind stencil |
| Initial fit | Trained sampled-field encoder plus correction using full-field projection | Sampled-field nearest-code fitting; later corrected Gauss initialization |
| FOM preconditioner | Dense sine-matrix products | FFT-based sine transforms |
| FOM choices | Same-grid Newton/linear tolerance ladder, fixed timestep | Same/coarse-grid, timestep and tolerance choices at a physical target |
| Time/output protocol | Step 0.005, 51 states retained on GPU | Selected time steps and requested observation times; all fields returned to host |
| Accuracy summary | Average relative error against same-grid truth | Worst time/case physical error with an empirical reference allowance |

The omission of the tensor was deliberate in the kickoff source/protocol: it chose the sign-dependent sampled operator as the general path. But the implications were not made clear in the later results discussion. The old trained sampled-field initializer also was not carried into the current Burgers query. A learned starting map is therefore a mechanism to revisit from project history, not an entirely new architectural idea. These are coverage gaps in the comparison with the user's earlier work, not evidence that its cached tensor stopped scaling.

For Poisson, the exact precomputed weak matrix is still present. The current repeated residual is $B h(z)-f_m$; it has not reverted to evaluating the full-grid Laplacian during every iteration. The tested weak dimension increased from 64 to 257, checkpoints and cases changed, and the latest primary residual-reduction setting is 0.01. Its initialization and source-projection studies also changed implementation choices. Both input processing and dense output remain resolution-dependent, even when the repeated reduced algebra is not.

## Why the Burgers operator requires a controlled restoration

With a fixed learned bank and fixed backward differences, advection admits

$$a_i(h)=\sum_{j,k}T_{ijk}h_jh_k.$$

The tensor is precomputed; the online contraction contains no grid-size loop. However, the FOM switches its upwind stencil where the decoded field changes sign. The old fixed-backward tensor is exact for nonnegative decoded states. Archived decoders had negative undershoots, and the tensor/full-upwind discrepancy was measured as small and nonzero. The historical report is not a blanket exactness certificate for sign-changing states or a different bank.

Changing to a larger bank also changes tensor work/storage through its bank dimensions. The old larger-bank compression trial stopped at its accuracy check; it did not establish a usable tensor result for the current checkpoint. Retaining the old small-bank tensor and the current sampled checkpoint as separate controls is necessary before deciding which model is preferable.

The full query cannot be strictly independent of resolution while reading and returning arbitrary dense fields: those operations must touch the requested values. The old mathematical claim applies to the cached reduced solve. Its full-field ends happened to remain small over the tested range, and host transfers were not part of its device “end-to-end” timer.

## Corrected conclusion and next comparison

The earlier observations are not retracted by the current negative table. The current table does not establish that the tensor method lost its old performance, and the old table does not establish a win under today's efficient-FOM and host-output protocol. The argument “the ROM is nonlinear” describes overhead present in both generations and cannot, by itself, explain their different reported ordering.

Before a broad new architecture sweep, the next bounded comparison should reconnect these experiments:

1. Restore the archived small-bank Burgers tensor checkpoint and sampled/full-upwind controls under a common physical-input and output contract. Reproduce the original controls first, then measure frozen transfer separately from mesh-specific weights.
2. On each mesh and one GPU job, persist both device-resident and host-to-host query timings. Retain component timings separately; do not substitute them for complete-query measurements.
3. Use the current FFT-preconditioned FOM at the same mesh, and the efficient coarse-mesh envelope at matched physical accuracy. Keep the old dense-preconditioner FOM only as a historical control if needed to isolate the baseline change.
4. Cross residual choice at a fixed checkpoint before comparing different bank sizes. Audit sign changes, tensor/full-upwind residual and derivative agreement, actual trajectory error, convergence failures and every timing repetition. Compare on a shared refined reference with mean, median and worst errors.
5. For Poisson, verify that the old quadrature-free path and current exact weak path agree when supplied the same checkpoint, mesh, tests, source and stopping rules. Compare both with direct and properly identified iterative baselines under the same timer.

These are proposed experiments, not results. No new job, branch or worktree was created for this audit. Existing approved experiment worktrees remain separate, and old wave experiments were not used.

## Sources and reproduction

The [user-specified historical table](2026-09-03-burgers-poisson-tensor-tables.md), [Burgers tensor report](2026-08-30-b2d-tensor-ladder.md), [Poisson quadrature-free report](2026-08-27-b1d-node-screening-and-poisson-qf.md), and [current campaign report](2026-09-07-multiresolution-pilots.md) retain their original scope.

Inspected source: [old tensor timer](../worktrees/2026-08-29-b2d-tensor/experiments/separable-decoder/sep_b2d_tensor.py), [old dense-preconditioned FOM](../worktrees/2026-08-29-b2d-tensor/experiments/separable-decoder/sep_burgers_r4.py), [old Poisson QF timer](../worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder/sep_poisson_qf.py), [current Burgers operators](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/engines.py), [current Burgers host query](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/rollout.py), [current Poisson assembly](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/core.py), and the [approved campaign protocol](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/MULTIRESOLUTION-PAPER-DESIGN.md).

Run `reports/generate_historical_cost_audit.py` with the repository's absolute Python environment to regenerate this report and its [evidence JSON](2026-09-10-historical-poisson-and-burgers-cost-audit.json). The JSON contains source content fingerprints, exact extracted values and timing/error aggregation checks. Newly recorded hashes identify the bytes reviewed; they are not independent proof of original GPU execution. Earliest rows without raw repetitions in their schema remain identified rather than assigned invented validation. Edit the generator template and regenerate.

## Plain-language glossary

- **FOM / ROM / NM-ROM:** the classical full solver / a reduced solver / a reduced solver constrained to a learned nonlinear family of fields.
- **N, nodes, intervals:** the old N counts grid nodes per axis including boundaries; the current N-like label counts intervals. Their interior unknown counts differ, even at equal printed labels.
- **K, R, M:** latent coordinates being solved / spatial features produced by the bank / weak equations used to fit the PDE. A small K does not imply a small R.
- **Bank, head, checkpoint, frozen transfer:** learned spatial functions / the network producing their coefficients / saved trained weights / evaluating the same weights on different grids.
- **Tensor, EQ, QF, full-grid residual:** precomputed coefficients of a quadratic advection operator / empirical quadrature using fitted points and weights / quadrature-free reduced evaluation / evaluating and projecting every grid point during each residual call. A full-grid residual ROM is not the FOM.
- **Oracle / sampled control / positivity / undershoot:** the full-upwind ROM diagnostic / the same decoder with quadrature / a nonnegative field / decoded values below zero. The tensor's sign qualification concerns decoded states, not only truth data.
- **CG, Newton, preconditioner, FFT, spectral solve:** iterative linear solving / nonlinear correction / a helper linear operator accelerating iterations / a fast transform algorithm / solving diagonal equations in a transformed basis.
- **Solve ms, paired, ratio, AB/BA:** milliseconds for reduced evolution / alternating ROM and FOM timings in one job / historical cohort FOM median divided by ROM median / alternated execution order. Above one favors ROM under that row's original protocol.
- **Device query / host-to-host query / decode:** inputs and outputs already on the GPU / starting and ending in CPU memory with transfers charged / reconstructing fields from reduced coefficients. Old “end-to-end” includes initialization and decoding but not host transfers.
- **Mean / median / worst time-case error:** average over recorded cases and times / the middle statistic / the largest recorded case/time error. Current and initial normalization divide by the current or initial truth norm respectively; physical and same-grid references are different comparisons.
- **Censored, stationary, stopping rule:** a target reduction was not achieved before another stopping condition / a sufficiently small optimization gradient / the condition terminating the solve. None alone is a physical-error guarantee.
- **Envelope, refined reference, empirical allowance:** the cheapest tested qualifying FOM / a more resolved comparison solution / an observed estimate of reference uncertainty, not a rigorous bound.
- **Outlier, seed, provisional, scope:** an unusually long retained repetition under the stated rule / recorded randomness / pending independent confirmation / the precise data, hardware, metrics and timer to which a result applies.
