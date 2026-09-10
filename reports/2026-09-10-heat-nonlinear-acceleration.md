# Accelerating heat while retaining the nonlinear decoder

Completed and independently audited development experiments with the frozen current nonlinear decoder. The numbers are provisional for paper claims: the convergence follow-up was chosen after the initial screen, and independent final cases remain unopened.

In the convergence follow-up at 1024 intervals, the original NMROM takes 12.939106 ms and linear prediction plus nonlinear projection takes 7.240736 ms, a 1.787× same-job speed ratio. Worst current-relative errors are 4.555479% and 4.545182%. The projection method has 3 nonstationary timed steps on this mesh.

The same-job direct FOM takes 1.085098 ms with 0.000350% worst error; its interpolated coarse counterpart takes 0.362294 ms with 2.577910%. The nonlinear acceleration does not beat that matched-target coarse GPU baseline. All headline ratios use one allocation.

A smaller improvement passes the stationarity checks in the initial screen: adaptive initialization with original stepping takes 11.955556 ms versus its paired original NMROM at 13.385909 ms, reducing median GPU time by 10.686%. Its worst error is 4.555479%, with 0 nonstationary time steps and 0 queries containing nonstationary attempted initial fits. This result is also development-only.

All nonlinear methods retain 8 latent variables, 32 learned bank functions and 64 smooth weak tests, with no retraining. The linear control evolves 32 free coefficients and is a different model class. Every method receives a full supplied initial field and returns all 6 requested full fields.

## Convergence follow-up

The initial screen allowed 30 correction attempts per projection. Some solves exhausted that budget despite acceptable physical error, so calling that method converged was premature. The follow-up raises only the full-projection limit to 120, retains the original initialization, and reruns the original NMROM and all direct controls in the same job. This is a development follow-up, not independent confirmation.

| Intervals | Method | Median GPU ms | Worst relative error (%) | Meets physical development target | Nonstationary steps / all steps | Affected cases | GPU timing outliers |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 64 | Original NMROM | 12.790777 | 4.559260 | yes | 0 / 720 | 0 | 0 |
| 64 | Linear prediction, full-budget nonlinear projection | 6.871912 | 4.545174 | yes | 3 / 180 | 1 | 0 |
| 64 | Free-coefficient linear bank control | 0.112250 | 1.675754 | yes | 0 / 0 | 0 | 0 |
| 64 | Same-grid direct FOM | 0.182579 | 0.089749 | yes | 0 / 0 | 0 | 0 |
| 64 | Coarse direct FOM with interpolation | 0.176062 | 2.585355 | yes | 0 / 0 | 0 | 12 |
| 1024 | Original NMROM | 12.939106 | 4.555479 | yes | 0 / 720 | 0 | 0 |
| 1024 | Linear prediction, full-budget nonlinear projection | 7.240736 | 4.545182 | yes | 3 / 180 | 1 | 0 |
| 1024 | Free-coefficient linear bank control | 0.599154 | 1.675830 | yes | 0 / 0 | 0 | 0 |
| 1024 | Same-grid direct FOM | 1.085098 | 0.000350 | yes | 0 / 0 | 0 | 0 |
| 1024 | Coarse direct FOM with interpolation | 0.362294 | 2.577910 | yes | 0 / 0 | 0 | 6 |

Provisional development results: all 12 cases and 3 repetitions per case are retained. Physical qualification uses worst full-field error plus the empirical reference-refinement allowance at the declared 5% target. It does not imply solver stationarity. Step counts include repeated invocations; affected cases count unique first-repetition case trajectories. FOM and free-bank controls have no nonlinear steps.

### Per-case comparison at 1024 intervals

Cohort medians can hide a slow difficult query. Each row below retains the median over that case's repetitions; a speed ratio below unity means the projected method is slower.

| Case | Original GPU ms | Projected GPU ms | Original / projected speed ratio | Projected worst error (%) | Nonstationary steps in first repetition |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 14.227108 | 25.558143 | 0.557 | 4.030350 | 1 |
| 1 | 12.024641 | 6.745754 | 1.783 | 1.717485 | 0 |
| 2 | 12.850553 | 9.340657 | 1.376 | 3.621957 | 0 |
| 3 | 11.580378 | 6.014972 | 1.925 | 0.978114 | 0 |
| 4 | 13.046219 | 6.811775 | 1.915 | 1.730044 | 0 |
| 5 | 12.310496 | 6.747395 | 1.824 | 3.319693 | 0 |
| 6 | 13.372372 | 9.706382 | 1.378 | 2.827104 | 0 |
| 7 | 12.177000 | 6.667211 | 1.826 | 4.499150 | 0 |
| 8 | 13.436337 | 10.369932 | 1.296 | 1.233786 | 0 |
| 9 | 14.191441 | 7.184598 | 1.975 | 4.545182 | 0 |
| 10 | 16.080183 | 10.918007 | 1.473 | 3.593967 | 0 |
| 11 | 12.483632 | 7.261066 | 1.719 | 1.329292 | 0 |

### Remaining convergence failures

The following distinct first-repetition corrections miss the declared normalized gradient threshold 1e-05. Extra attempts do not establish convergence for these cases; the measured physical errors and speed gains do not remove this limitation.

| Intervals | Case | Output time | Attempts | Accepted attempts | Relative weak residual | Normalized gradient |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0 | 0.2 | 120 | 83 | 0.018446981879 | 1.79200252885e-05 |
| 1024 | 0 | 0.2 | 120 | 83 | 0.0185548832279 | 2.03935833873e-05 |

A useful next controlled test is a shorter prediction interval, charging the extra projections and checking both physical error and stationarity. Increasing the correction cap again without diagnosing the difficult step is not an established solution.

### Initialization, transfers and energy

| Intervals | Method | Median host query ms | Host outliers | Skipped second fits / queries | Queries with nonstationary attempted initial fits | Total initial attempts | Total time-step attempts | Worst initial error (%) | Largest energy increase |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Linear prediction, full-budget nonlinear projection | 7.329501 | 0 | 0 / 36 | 0 | 810 | 1434 | 4.545174 | -0.000586234119 |
| 64 | Coarse direct FOM with interpolation | 0.518864 | 12 | 0 / 36 | 0 | 0 | 0 | 0.000000 | -0.000570736651 |
| 64 | Same-grid direct FOM | 0.552729 | 0 | 0 / 36 | 0 | 0 | 0 | 0.000000 | -0.000585852711 |
| 64 | Free-coefficient linear bank control | 0.475393 | 0 | 0 / 36 | 0 | 0 | 0 | 1.675754 | -0.000585864834 |
| 64 | Original NMROM | 13.547893 | 0 | 0 / 36 | 0 | 810 | 1812 | 4.545174 | -0.000585108432 |
| 1024 | Linear prediction, full-budget nonlinear projection | 31.794531 | 0 | 0 / 36 | 0 | 810 | 1338 | 4.545182 | -0.000585530789 |
| 1024 | Coarse direct FOM with interpolation | 24.937627 | 0 | 0 / 36 | 0 | 0 | 0 | 0.000000 | -0.000569148004 |
| 1024 | Same-grid direct FOM | 25.578003 | 0 | 0 / 36 | 0 | 0 | 0 | 0.000000 | -0.000585265861 |
| 1024 | Free-coefficient linear bank control | 25.129488 | 0 | 0 / 36 | 0 | 0 | 0 | 1.675830 | -0.000585279067 |
| 1024 | Original NMROM | 36.776523 | 0 | 0 / 36 | 0 | 810 | 1821 | 4.545182 | -0.00058437199 |

Counts include every retained timed repetition. A negative largest energy increase means every consecutive requested output loses energy. A skipped initialization has its own recorded reason and is excluded from the nonstationary-fit count. Host query time adds measured input/output transfers to the same invocation. Neither solver is charged for data generation, offline assembly or compilation; setup and warmup costs are retained in the native results.

### Follow-up evidence

Job `3519092`, `NVIDIA A100-PCIE-40GB` on `pax003`, scientific source `4c4d27d201440c998ab94cc3055f258c7737b787`. GPU preflight, float64 and highest matrix precision were checked. The private remote directory was removed after checksum collection.

Independent NumPy/SciPy audit: 180 unique full fields; 360 timing/error invocations; 26352 metric entries, maximum discrepancy 5.26245713672e-14. Reference-refinement delta 1.62810884586e-12; this is empirical agreement, not a rigorous continuum error bound.

For 144 nonlinear trajectories, every requested time was independently reconstructed at the common physical observation grid from saved latent coordinates and frozen neural weights; maximum relative mismatch 4.12405834879e-16. This checks sampled decoder identity alongside full-field error auditing; it is not an independent full-grid neural reconstruction.

The audit reconstructs 360 projected residual/gradient pairs with maximum difference 1.00691803935e-15, and checks every applicable adaptive-initialization gate. Reduced operators agree with independent QR algebra within 2.5483808158e-14. Original NMROM fields match the earlier frozen-head archive within 4.07217186713e-15. Timing comparisons use this allocation only.

[Raw results](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/nonlinear07/archive/outputs/results.json) · [Independent audit](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/nonlinear07/analysis/audit.json) · [Configuration](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/nonlinear07/archive/experiments/mr-heat2d/config-nonlinear-convergence.json)

## Initial screen: all declared variants

The most aggressive combined variant uses at most 2 correction attempts per projection and conditionally skips the second initial fit. At 1024 intervals it takes 4.152898 ms, but its worst error is 6.193099%, above the declared physical target. There is no hidden fallback. It remains a reported control, not a selected success.

| Intervals | Method | Median GPU ms | Worst relative error (%) | Meets physical development target | Nonstationary steps / all steps | Affected cases | GPU timing outliers |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 64 | Original NMROM | 12.990940 | 4.559260 | yes | 0 / 720 | 0 | 0 |
| 64 | Original stepping, adaptive initialization | 11.539594 | 4.559260 | yes | 0 / 720 | 0 | 0 |
| 64 | Linear prediction, full-budget nonlinear projection | 6.730020 | 4.545174 | yes | 6 / 180 | 2 | 0 |
| 64 | Linear prediction, full-budget projection, adaptive initialization | 6.923037 | 4.545174 | yes | 3 / 180 | 1 | 0 |
| 64 | Linear prediction, bounded nonlinear projection | 4.411399 | 6.129091 | no | 180 / 180 | 12 | 0 |
| 64 | Linear prediction, bounded projection, adaptive initialization | 3.382973 | 6.129091 | no | 180 / 180 | 12 | 0 |
| 64 | Free-coefficient linear bank control | 0.130679 | 1.675754 | yes | 0 / 0 | 0 | 0 |
| 64 | Same-grid direct FOM | 0.190665 | 0.089749 | yes | 0 / 0 | 0 | 0 |
| 64 | Coarse direct FOM with interpolation | 0.183567 | 2.585355 | yes | 0 / 0 | 0 | 12 |
| 1024 | Original NMROM | 13.385909 | 4.555479 | yes | 0 / 720 | 0 | 0 |
| 1024 | Original stepping, adaptive initialization | 11.955556 | 4.555479 | yes | 0 / 720 | 0 | 0 |
| 1024 | Linear prediction, full-budget nonlinear projection | 7.195643 | 4.545182 | yes | 6 / 180 | 2 | 0 |
| 1024 | Linear prediction, full-budget projection, adaptive initialization | 7.636279 | 4.545182 | yes | 3 / 180 | 1 | 0 |
| 1024 | Linear prediction, bounded nonlinear projection | 4.883697 | 6.193099 | no | 180 / 180 | 12 | 0 |
| 1024 | Linear prediction, bounded projection, adaptive initialization | 4.152898 | 6.193099 | no | 180 / 180 | 12 | 0 |
| 1024 | Free-coefficient linear bank control | 0.537293 | 1.675830 | yes | 0 / 0 | 0 | 0 |
| 1024 | Same-grid direct FOM | 0.947071 | 0.000350 | yes | 0 / 0 | 0 | 0 |
| 1024 | Coarse direct FOM with interpolation | 0.492373 | 2.577910 | yes | 0 / 0 | 0 | 9 |

Provisional development results: all 12 cases and 3 repetitions per case are retained. Physical qualification uses worst full-field error plus the empirical reference-refinement allowance at the declared 5% target. It does not imply solver stationarity. Step counts include repeated invocations; affected cases count unique first-repetition case trajectories. FOM and free-bank controls have no nonlinear steps.

### Initialization, transfers and energy

| Intervals | Method | Median host query ms | Host outliers | Skipped second fits / queries | Queries with nonstationary attempted initial fits | Total initial attempts | Total time-step attempts | Worst initial error (%) | Largest energy increase |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Linear prediction, bounded nonlinear projection | 4.950410 | 0 | 0 / 36 | 0 | 810 | 360 | 4.545174 | -0.000584434527 |
| 64 | Linear prediction, bounded projection, adaptive initialization | 3.840023 | 0 | 27 / 36 | 0 | 318 | 360 | 4.545174 | -0.000584434527 |
| 64 | Linear prediction, full-budget nonlinear projection | 7.320168 | 0 | 0 / 36 | 0 | 810 | 1080 | 4.545174 | -0.000586234119 |
| 64 | Linear prediction, full-budget projection, adaptive initialization | 7.464673 | 0 | 27 / 36 | 0 | 318 | 1062 | 4.545174 | -0.000586234119 |
| 64 | Coarse direct FOM with interpolation | 0.538623 | 12 | 0 / 36 | 0 | 0 | 0 | 0.000000 | -0.000570736651 |
| 64 | Same-grid direct FOM | 0.558475 | 0 | 0 / 36 | 0 | 0 | 0 | 0.000000 | -0.000585852711 |
| 64 | Free-coefficient linear bank control | 0.550012 | 0 | 0 / 36 | 0 | 0 | 0 | 1.675754 | -0.000585864834 |
| 64 | Original NMROM | 14.161595 | 0 | 0 / 36 | 0 | 810 | 1812 | 4.545174 | -0.000585108432 |
| 64 | Original stepping, adaptive initialization | 12.515711 | 0 | 27 / 36 | 0 | 318 | 1935 | 4.545174 | -0.000585108432 |
| 1024 | Linear prediction, bounded nonlinear projection | 23.005209 | 0 | 0 / 36 | 0 | 810 | 360 | 4.545182 | -0.000585477056 |
| 1024 | Linear prediction, bounded projection, adaptive initialization | 22.593383 | 0 | 27 / 36 | 0 | 318 | 360 | 4.545182 | -0.000585477056 |
| 1024 | Linear prediction, full-budget nonlinear projection | 26.019443 | 0 | 0 / 36 | 0 | 810 | 1065 | 4.545182 | -0.000585530789 |
| 1024 | Linear prediction, full-budget projection, adaptive initialization | 25.288837 | 0 | 27 / 36 | 0 | 318 | 1065 | 4.545182 | -0.000585530789 |
| 1024 | Coarse direct FOM with interpolation | 18.760439 | 0 | 0 / 36 | 0 | 0 | 0 | 0.000000 | -0.000569148004 |
| 1024 | Same-grid direct FOM | 19.117499 | 0 | 0 / 36 | 0 | 0 | 0 | 0.000000 | -0.000585265861 |
| 1024 | Free-coefficient linear bank control | 18.875582 | 0 | 0 / 36 | 0 | 0 | 0 | 1.675830 | -0.000585279067 |
| 1024 | Original NMROM | 31.074143 | 0 | 0 / 36 | 0 | 810 | 1821 | 4.545182 | -0.00058437199 |
| 1024 | Original stepping, adaptive initialization | 30.949321 | 0 | 27 / 36 | 0 | 318 | 1923 | 4.545182 | -0.00058437199 |

Counts include every retained timed repetition. A negative largest energy increase means every consecutive requested output loses energy. A skipped initialization has its own recorded reason and is excluded from the nonstationary-fit count. Host query time adds measured input/output transfers to the same invocation. Neither solver is charged for data generation, offline assembly or compilation; setup and warmup costs are retained in the native results.

### Initial-screen evidence

Job `3518484`, `NVIDIA A100 80GB PCIe` on `pax106`, scientific source `da53af9eae791da3f234d98ff9818d98d493de7c`. GPU preflight, float64 and highest matrix precision were checked. The private remote directory was removed after checksum collection.

Independent NumPy/SciPy audit: 276 unique full fields; 648 timing/error invocations; 47088 metric entries, maximum discrepancy 5.26245713672e-14. Reference-refinement delta 1.62810884586e-12; this is empirical agreement, not a rigorous continuum error bound.

For 432 nonlinear trajectories, every requested time was independently reconstructed at the common physical observation grid from saved latent coordinates and frozen neural weights; maximum relative mismatch 4.67229512442e-16. This checks sampled decoder identity alongside full-field error auditing; it is not an independent full-grid neural reconstruction.

The audit reconstructs 1440 projected residual/gradient pairs with maximum difference 1.36782946081e-15, and checks every applicable adaptive-initialization gate. Reduced operators agree with independent QR algebra within 3.0010322191e-14. Original NMROM fields match the earlier frozen-head archive within 3.55889896134e-15. Timing comparisons use this allocation only.

[Raw results](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/nonlinear06/archive/outputs/results.json) · [Independent audit](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/nonlinear06/analysis/audit.json) · [Configuration](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/nonlinear06/archive/experiments/mr-heat2d/config-nonlinear-fast.json)

## Method and limits

The original integrator uses 20 internal Crank–Nicolson steps. The new method uses 5 linear predictions and nonlinear projections, one per requested observation interval. It changes the time integrator, not just the algebra used to produce the old trajectory.

Write the bank as $G=QR$ and the verified reduced linear generator in orthogonal coordinates as $L$. Precompute $P_a=R^{-1}\exp(\Delta tL)R$. Each nonlinear update solves

$$z_{j+1}\approx\arg\min_z\|B h(z)-B P_a h(z_j)\|_2^2,\qquad u_{j+1}=G h(z_{j+1}).$$

Here $B$ maps bank coefficients to smooth weak moments, and $h$ is the frozen nonlinear head. The free linear predictor is intermediate; every returned nonlinear-method field is decoded from the latent state. A matrix exponential does not make the constrained nonlinear rollout exact.

The screen's adaptive initializer runs the nearest-code fit first, and tries the mean-code fit when stationarity, finiteness or the full-field initial-error gate of 4% fails. The error gate includes the component outside the learned bank. This is a changed initialization policy and can change trajectories. The convergence follow-up isolates projection cost by retaining the original two-start policy.

The follow-up projected method has worst initial error 4.545182% at 1024 intervals. Faster time stepping does not repair the initial nonlinear fit or the decoder's representation limit. Accuracy work should next diagnose and improve that initial representation/fit; speed work should reduce its online cost without hiding the supplied-field input or requested field output.

On that same worst-initial-error case (9), the free-bank initial projection error is only 0.932465%. This identifies a gap beyond the fixed spatial bank's representation error. The present experiment does not separate nonlinear-head representation limits from local-fit optimization limits; a converged local fit is not proof of a global optimum.

The current test uses fixed diffusivity and the existing single-bump family. It does not establish multi-bump, variable-diffusivity or other-PDE performance. The full resolution ladder, per-resolution training and independent final-cohort confirmation remain open. Endpoint results and post-screen convergence tuning are insufficient for a paper-wide scaling claim.

Timing outlier rule: Above 1.5 times the median of the same case/method/mesh repetition group; none excluded.

## Glossary

- **Intervals:** cells per spatial axis; homogeneous boundary values are implicit in saved interior arrays.
- **NMROM / FOM:** nonlinear-manifold reduced model / full-grid numerical solver.
- **Bank / head / latent:** frozen spatial functions / nonlinear coefficient map / evolving compressed coordinates.
- **Linear prediction / nonlinear projection:** evolve free bank coefficients, then fit them back to the nonlinear decoder using weak moments.
- **Full-budget / bounded projection:** use the stated larger correction limit / allow only the stated small number of attempts. Neither label assumes convergence.
- **Adaptive initialization / two-start policy:** conditionally try a second starting guess / always fit both declared guesses and select the better fit.
- **LM / attempt / stationarity:** damped nonlinear least squares / an accepted or rejected correction trial / satisfying the gradient stopping rule.
- **Nonstationary steps / affected cases:** timed corrections that miss the stopping criterion / distinct first-repetition case trajectories containing such a correction.
- **Queries with nonstationary attempted initial fits:** invocations where at least one attempted starting-guess fit misses stationarity; not necessarily the selected fit.
- **Weak moments / residual:** averages against smooth sine test functions / discrepancy in those averages.
- **Relative weak residual / normalized gradient:** weak discrepancy divided by the target moment norm / least-squares gradient divided by the Jacobian norm, used for the stopping rule.
- **Crank–Nicolson / matrix exponential:** the original implicit time-step formula / direct propagation for a fixed linear differential equation.
- **Median GPU / host query:** median blocked GPU-input-to-GPU-output time / same invocation including measured CPU–GPU field transfers.
- **Worst relative / initial error:** largest field error divided by the current reference norm over all times / initial time only, expressed as percentages.
- **Physical development target / cohort:** declared field-error threshold / fixed group of sampled cases used for development, distinct from final confirmation.
- **Outlier:** timing above the stated within-case threshold; no such samples are removed.
- **Energy increase:** consecutive change in discrete squared-field energy; positive values indicate growth.
- **Free-coefficient control / coarse FOM:** linear learned-bank model / smaller-grid direct solve interpolated to requested output points.
- **Refinement allowance / sampled identity check:** empirical difference between reference resolutions / independent decoder reconstruction at shared physical observation nodes.
- **QR / orthogonal coordinates:** factorization into orthonormal and triangular matrices / coordinates defined by that orthonormal bank.
- **Scientific source / checksum collection:** exact code version that produced a run / validation of collected files against their recorded hashes.
