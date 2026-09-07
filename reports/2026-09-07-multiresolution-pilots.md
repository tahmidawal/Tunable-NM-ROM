# Multiresolution development pilots: accuracy and complete-query cost

This report covers the first frozen-network mesh-transfer pilots and bounded numerical improvements of the current separable NM-ROM against efficient FOM solvers. The numbers are provisional development evidence; independent confirmation and the full resolution study remain open.

The tested frozen decoders produce solutions on new meshes, but these primary configurations have not established a complete-query advantage over efficient FOMs. Increasing resolution does not reliably reduce ROM error in these pilots. Further work must address representation, initialization or reduced-solver cost, according to the PDE.

The older ViT + CP architecture is excluded. Waves use only the fresh verified lineage. The final cohorts remain unopened. Different rows use different physical error definitions, stated below; they must not be ranked as a common cross-PDE accuracy score.

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

The table uses a budget of `180` iterations and `4` training-code starting guesses, reporting the worst physical case. The same edge samples give essentially the same result under Gram and QR fitting; fixed physical midpoint or Gauss sampling greatly reduces the finer-grid fitting error. The remaining bank floor also grows with resolution. Thus both the changing sampled objective and a real frozen-bank representation loss matter. The earlier explanation based only on optimizer starting guesses is withdrawn. The clipped initial-condition family has not changed. These are best-recorded nonlinear fits, not stationary or globally optimal certificates. Fixed physical field sampling with charged interpolation is an initialization method, not strong-form PDE collocation. Corrected-initializer rollouts and their complete-query costs remain unmeasured.

[Complete Burgers follow-up findings](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/reports/2026-09-07-burgers2d-multiresolution.md).

All cost ratios in this report use ratios of per-case timing medians before the cohort median. Some native exploratory reports also retain medians of per-repetition ratios or select configurations by the pooled median across all repetitions under an explicit different label; those statistics are not interchangeable.

## Next experiments justified by the diagnostics

- **Burgers:** carry fixed physical sampling into a complete rollout with charged input interpolation; preserve the original initial-condition family and compare against the unchanged efficient FOM envelope. If the frozen-bank boundary gap still blocks accuracy, test boundary-aware training coverage separately. The improved reference supports only the stated development targets.
- **Heat:** after the measured tolerance improvement, address the nonlinear-head reconstruction gap with controlled original-cohort versus expanded-coverage head refinement. The spatial bank and head architecture can remain fixed for that diagnostic.
- **Poisson:** the test-mode and representation diagnostics point to bank/head capacity or training coverage for accuracy. A specialized small-matrix solver is a separate remaining runtime test.
- **Waves:** use matched-dimensional linear and nonlinear controls with a frozen spatial bank to separate compression from autonomous dynamics. Finer rendering alone cannot address the current error.

These are development decisions. The complete study still needs the full mesh ladder, separately labeled per-resolution training, independent data/training repeats, validation-selected settings and sealed final evaluation.

## Visual artifacts

[Heat accuracy and complete-query cost](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign/heat-pilot01-accuracy-cost.png). [Reflective wave evolving](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign/dirichlet-wave-evolution-case1.png). [Absorbing wave evolving](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/multiresolution_campaign/absorbing-wave-evolution-case1.png).

The wave still sequences show reference displacement, predicted displacement and absolute difference with fixed scales. They use saved fields from an actual finer-mesh ROM solve; the display resolution is labeled. Displacement pictures do not replace the velocity, energy or vanishing-field diagnostics. PDF exports and figure source/provenance are beside the images.

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
