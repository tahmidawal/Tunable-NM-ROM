# Transferring the CP Gram and Cholesky optimizations to heat NMROM

Completed and independently audited development experiment. Numbers are provisional for paper claims because this uses one frozen training checkpoint and existing development cases; final confirmation cases remain unopened.

At 1024 intervals, the declared Gram + Cholesky method takes 12.915947 ms versus 13.465232 ms for the original NMROM: a 1.043× ratio of same-job median GPU times. Worst current-relative errors are 4.555479% and 4.555479%. This algebra change reduces the measured cohort median GPU cost.

Cholesky alone takes 12.396389 ms, a 7.938% reduction in median GPU time, with worst error 4.555479%. This is a separately declared ablation, not a replacement for the joint method's primary result.

The same-grid direct FOM takes 0.885449 ms with 0.000350% worst error. The coarse direct FOM takes 0.287753 ms with 2.577910% worst error. The primary nonlinear method is not faster than the coarse GPU control at the declared 5% development target.

## All declared comparisons

All timings below come from one allocation. GPU time runs from supplied GPU initial field to all requested GPU fields, blocked. Host time adds measured input and output transfers for that same invocation. Data generation, mesh assembly and compilation are excluded from every online time and recorded separately.

| Intervals | Method | Median GPU ms | Median host ms | Worst relative error (%) | Original / method GPU ratio | Nonstationary fits / steps | GPU / host outliers |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Original NMROM | 13.282339 | 14.238903 | 4.559260 | 1.000 | 0 / 0 | 0 / 0 |
| 64 | Cholesky only | 12.282824 | 12.853772 | 4.559260 | 1.081 | 0 / 0 | 0 / 0 |
| 64 | Precomputed Gram only | 12.775359 | 13.268740 | 4.559260 | 1.040 | 0 / 0 | 0 / 0 |
| 64 | Gram + Cholesky (declared primary) | 12.091856 | 12.777536 | 4.559260 | 1.098 | 0 / 0 | 0 / 0 |
| 64 | Free-coefficient linear bank control | 0.110811 | 0.450824 | 1.675754 | 119.865 | 0 / 0 | 0 / 0 |
| 64 | Same-grid direct FOM | 0.167490 | 0.507660 | 0.089749 | 79.303 | 0 / 0 | 0 / 0 |
| 64 | Coarse direct FOM with interpolation | 0.163405 | 0.525770 | 2.585355 | 81.285 | 0 / 0 | 12 / 12 |
| 1024 | Original NMROM | 13.465232 | 28.791183 | 4.555479 | 1.000 | 0 / 0 | 0 / 0 |
| 1024 | Cholesky only | 12.396389 | 27.967277 | 4.555479 | 1.086 | 0 / 0 | 0 / 0 |
| 1024 | Precomputed Gram only | 13.471694 | 29.283634 | 4.555479 | 1.000 | 0 / 0 | 0 / 0 |
| 1024 | Gram + Cholesky (declared primary) | 12.915947 | 28.389882 | 4.555479 | 1.043 | 0 / 0 | 0 / 0 |
| 1024 | Free-coefficient linear bank control | 0.514046 | 16.184805 | 1.675830 | 26.195 | 0 / 0 | 0 / 0 |
| 1024 | Same-grid direct FOM | 0.885449 | 16.662808 | 0.000350 | 15.207 | 0 / 0 | 0 / 0 |
| 1024 | Coarse direct FOM with interpolation | 0.287753 | 16.085625 | 2.577910 | 46.794 | 0 / 0 | 9 / 0 |

Provisional development results: 12 cases, 3 repetitions per case and method. Error is the worst over every case, requested time and retained repetition. Outlier rule: Above 1.5 times the median of the same case/method/mesh repetition group; none excluded.

The nonlinear variants all retain the same 8 latent coordinates, 32 spatial-bank functions, 64 smooth weak tests, initial fitting policy, 20 Crank–Nicolson steps of size 0.025, and normalized gradient tolerance 1e-05. Every method returns all 6 requested fields. The linear bank control has free coefficients and is a different model class.

## Per-case primary comparison

These are medians within each case at 1024 intervals. A ratio below unity means the primary method is slower.

| Case | Original GPU ms | Gram + Cholesky GPU ms | Original / primary | Primary worst error (%) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 14.768559 | 13.978758 | 1.057 | 4.030350 |
| 1 | 12.029291 | 11.588098 | 1.038 | 1.717485 |
| 2 | 13.275342 | 12.508335 | 1.061 | 3.621957 |
| 3 | 12.120616 | 11.553896 | 1.049 | 0.978114 |
| 4 | 13.747684 | 12.961620 | 1.061 | 1.730044 |
| 5 | 12.789757 | 12.523481 | 1.021 | 3.533069 |
| 6 | 13.584564 | 13.181208 | 1.031 | 3.411909 |
| 7 | 12.960722 | 12.562281 | 1.032 | 4.555479 |
| 8 | 13.515551 | 13.096328 | 1.032 | 1.233786 |
| 9 | 14.697582 | 13.716171 | 1.072 | 4.545182 |
| 10 | 16.700032 | 16.030637 | 1.042 | 3.593967 |
| 11 | 13.400407 | 12.528199 | 1.070 | 1.329292 |

Median of per-case speed ratios: 1.045×; primary faster in 12 of 12 cases. This is distinct from the ratio of cohort median times reported above.

## Numerical equivalence and convergence

Predeclared field and internal-latent parity gates passed: tolerances 1e-08 and 1e-06. All initial-fit and time-step iteration/acceptance/termination counters agree with the same-job original. Rounding differences are measured, not assumed absent.

| Variant | Maximum field relative difference | Maximum internal-latent relative difference | Comparisons failing a parity gate | Comparisons with different counters |
| --- | ---: | ---: | ---: | ---: |
| Cholesky only | 2.1826289799e-15 | 2.66140819037e-14 | 0 | 0 |
| Precomputed Gram only | 2.87807658318e-15 | 5.35443510255e-14 | 0 | 0 |
| Gram + Cholesky (declared primary) | 3.00211788972e-15 | 3.99495781693e-14 | 0 | 0 |

Independent NumPy head derivatives reproduce 5760 saved time-step residual/gradient pairs, maximum discrepancy 9.96807007054e-16. They reproduce 576 initial-fit diagnostic pairs within 1.68615121865e-15. Initial projection targets are recovered through independent QR from sampled linear-control initial fields; this is not a full-grid independent reconstruction of the input projection.

Full-field errors and hashes were audited for 504 timed invocations and 228 distinct arrays; maximum metric discrepancy 5.26245713672e-14. Sampled neural reconstruction over all requested output times has maximum relative mismatch 4.53593797667e-16. It complements full-field metric checks without claiming a separate full-grid neural evaluation.

Independent QR operator and Gram checks agree within 2.16387077449e-14. The unchanged baseline reproduces earlier archived fields within 2.30920215881e-15. The maximum continuum-spectral reference-refinement delta is 1.62810884586e-12; this remains empirical evidence, not a rigorous continuum bound.

## What was transferred, and what remains

For $r=(Bh(z)-b)/s$ and $D=\partial h/\partial z$, precompute $S=B^\top B$ offline and evaluate the normal matrix as $H=D^\top SD/s^2$. Here $B$ maps bank coefficients to weak moments, $b$ is their target, and $s=\max(\|b\|,\epsilon)$ normalizes the residual with a small positive floor. The gradient uses the same contraction, anchored at the actual initial residual to limit cancellation. The explicit small weak residual remains the loss used to accept or reject each trial.

Cholesky factors the positive-definite damped system instead of using general LU. Neither transformation changes the mathematical objective, time integrator or decoder. The test includes each transformation separately and both together; no post-selection arm is promoted to a predeclared result.

The historical CP optimization removed full-grid Jacobian work. This heat solver already projects to a small weak test space, so the transferable contraction acts on a much smaller matrix. Its payoff must be measured; historical speed ratios cannot be inherited. The full rollout was already compiled, so compiling it again does not reproduce the older eager-to-compiled gain.

The weak solve has no full-grid Jacobian, but still performs iterative nonlinear fits and time-step corrections. A grid-independent iteration can remain more expensive than this direct heat FOM. Reading the supplied full field and producing all requested full fields also continue to scale with resolution.

The original worst initial error is 4.545182% at 1024 intervals. Algebraic speed changes cannot repair that initial fitting/representation error. The next distinct architecture experiment is a learned initializer from supplied-field features, trained jointly with the coefficient head, with an explicit scale coordinate if useful. That proposal still requires accuracy and whole-query timing tests; it has not been trained or validated here.

This experiment uses the existing single-bump family and fixed diffusivity. It does not establish multi-bump, variable-coefficient, Burgers, Poisson or wave performance. The old heat CP evidence also used a different iterative FOM, precision and error summary; its speedups are not comparable to the current direct/coarse controls. See the [historical source review](2026-07-30-cp-heat-optimization-transfer.md).

## Reproducible evidence

Job `3528798` on `pax007`, `NVIDIA A100 80GB PCIe`, scientific source `8af2f6bbc1083701b83fbe2852a598a52e38bef1`. GPU preflight, float64, highest matrix precision, seed regeneration, private job directory and complete logs passed. Timing repetitions are retained; no timing outliers are discarded. Source and output checksums passed before the exact remote attempt directory was removed.

The local smoke completed in 24.831804 seconds and passed an independent exact least-squares fixture with a nonlinear head. Local smoke timings are not benchmark evidence.

[Raw results](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/cp_algebra08/archive/outputs/results.json) · [Independent audit](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/cp_algebra08/analysis/audit.json) · [Configuration](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/cp_algebra08/archive/experiments/mr-heat2d/config-cp-algebra.json) · [Archive manifest](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/cp_algebra08/ARCHIVE.json)

## Glossary

- **Intervals:** spatial cells per axis; saved arrays contain interior nodes.
- **CP / NMROM / FOM:** tensor-product decoder / nonlinear-manifold reduced model / full-grid numerical solver.
- **Bank / head / latent coordinates:** learned spatial functions / nonlinear coefficient map / compressed variables being solved.
- **Gram / normal equations / Jacobian:** matrix of inner products / small least-squares update system / derivatives of outputs with respect to solved variables.
- **LU / Cholesky:** general matrix factorization / factorization specialized to symmetric positive-definite systems.
- **Weak tests / residual / stationarity:** smooth averages of the equation / their discrepancy / meeting the specified gradient stopping rule.
- **Crank–Nicolson / damped solve:** original time discretization / least-squares correction stabilized by a positive diagonal term.
- **Median GPU / host ms:** middle retained blocked device-query time / same invocation including CPU–GPU transfers, in milliseconds.
- **Worst relative error:** maximum full-field error divided by the reference norm at the same time, over all cases, times and repetitions.
- **Original / method ratio:** original median time divided by the listed method median; greater than unity is faster than the original.
- **Nonstationary fits / steps:** queries containing any initial fit missing the gradient rule / individual time steps missing it; counts include repetitions.
- **Outliers:** timings above the stated threshold within the same case, mesh and method; all remain included.
- **Case / development target:** seeded initial condition / provisional physical-error threshold; neither is a final-cohort validation claim.
- **Parity / counters:** agreement in trajectory or latent coordinates / attempted and accepted corrections and termination reasons.
- **Free-coefficient control / coarse direct FOM:** linear bank model with unconstrained coefficients / smaller-grid direct propagation interpolated to output nodes.
- **QR / reference refinement / sampled reconstruction:** independent orthogonal-triangular factorization / difference between reference resolutions / checking decoder values at common physical nodes.
- **Invocation / checkpoint / source checksum:** one timed solve producing the graded fields / frozen trained parameters / content hash proving exact provenance.
