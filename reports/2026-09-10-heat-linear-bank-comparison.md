# Direct linear heat evolution in the learned spatial bank

Completed and audited development comparison of a linear ROM using the frozen learned bank, the current nonlinear ROM, and direct heat FOMs. These are provisional scientific findings because the cohort is used for development; final paper cases remain unopened.

At the finest requested mesh, freeing the bank coefficients changes worst current-relative error from 4.555479% to 1.675830% and GPU query time from 12.318828 ms to 0.561659 ms. The measured speed ratio is 21.933× versus the current NMROM and 1.842× versus the same-grid direct FOM. These ratios compare medians within this allocation; they do not imply matched FOM error.

The coarse direct FOM takes 0.313116 ms on the GPU with worst error 2.577910%. The linear ROM and this coarse FOM both meet the declared 5% development target. This run therefore does not establish a linear-ROM GPU advantage over the coarse-FOM control, although the linear ROM is more accurate.

The linear ROM evolves 32 free bank coefficients; the NMROM evolves 8 nonlinear latent variables. Both use the same frozen learned spatial weights and 64 smooth sine test moments. This is a linear learned-basis ROM, not the original nonlinear-manifold method and not a matched-dimension comparison.

## Finest mesh: 1024 intervals per axis

| Method | GPU query median (ms) | Host query median (ms) | Worst physical relative error (%) | Worst initial error (%) | GPU / host timing outliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear bank, exact reduced time evolution | 0.561659 | 17.681166 | 1.675830 | 1.675830 | 0 / 0 |
| Linear bank, original discrete weak step | 0.580536 | 24.580228 | 1.675830 | 1.675830 | 0 / 0 |
| Current nonlinear ROM | 12.318828 | 36.325427 | 4.555479 | 4.545182 | 0 / 0 |
| Same-grid direct FOM | 1.034563 | 25.068664 | 0.000350 | 0.000000 | 0 / 0 |
| Coarse direct FOM, interpolated | 0.313116 | 24.364979 | 2.577910 | 0.000000 | 10 / 0 |

Each row includes all 12 fixed development cases and 3 timed repetitions per case. Errors cover every requested output time, including the actual ROM initial fit/projection. The physical reference uses a continuum spectral operator with an empirical refinement check; the FOM solves its discrete spatial problem exactly in time.

## Complete mesh ladder

| Intervals | Method | GPU query median (ms) | Host query median (ms) | Worst physical relative error (%) | Worst same-grid relative error (%) | GPU / host outliers |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | Linear bank, exact reduced time evolution | 0.116167 | 1.146627 | 1.675754 | 1.675754 | 4 / 5 |
| 64 | Linear bank, original discrete weak step | 0.103405 | 0.545752 | 1.675754 | 1.675754 | 0 / 1 |
| 64 | Current nonlinear ROM | 12.091245 | 12.540239 | 4.559260 | 4.554246 | 0 / 0 |
| 64 | Same-grid direct FOM | 0.184825 | 0.557447 | 0.089749 | 0.000000 | 0 / 0 |
| 64 | Coarse direct FOM, interpolated | 0.176126 | 0.518843 | 2.585355 | 2.623291 | 11 / 12 |
| 128 | Linear bank, exact reduced time evolution | 0.117986 | 1.271889 | 1.675826 | 1.675826 | 2 / 3 |
| 128 | Linear bank, original discrete weak step | 0.102106 | 0.688633 | 1.675826 | 1.675826 | 0 / 1 |
| 128 | Current nonlinear ROM | 11.795272 | 12.407910 | 4.556344 | 4.555162 | 0 / 0 |
| 128 | Same-grid direct FOM | 0.189691 | 0.728313 | 0.022414 | 0.000000 | 1 / 0 |
| 128 | Coarse direct FOM, interpolated | 0.178217 | 0.701571 | 2.580599 | 2.590796 | 11 / 12 |
| 256 | Linear bank, exact reduced time evolution | 0.137731 | 2.326213 | 1.675830 | 1.675830 | 2 / 0 |
| 256 | Linear bank, original discrete weak step | 0.117647 | 1.398111 | 1.675830 | 1.675830 | 0 / 0 |
| 256 | Current nonlinear ROM | 11.888083 | 13.147866 | 4.555681 | 4.555390 | 0 / 0 |
| 256 | Same-grid direct FOM | 0.209627 | 1.438594 | 0.005602 | 0.000000 | 0 / 2 |
| 256 | Coarse direct FOM, interpolated | 0.190585 | 1.400465 | 2.578601 | 2.581196 | 12 / 9 |
| 512 | Linear bank, exact reduced time evolution | 0.235394 | 5.162003 | 1.675830 | 1.675830 | 0 / 0 |
| 512 | Linear bank, original discrete weak step | 0.236426 | 4.615708 | 1.675830 | 1.675830 | 0 / 0 |
| 512 | Current nonlinear ROM | 11.902372 | 16.352426 | 4.555520 | 4.555447 | 0 / 0 |
| 512 | Same-grid direct FOM | 0.336687 | 4.594101 | 0.001400 | 0.000000 | 0 / 1 |
| 512 | Coarse direct FOM, interpolated | 0.211906 | 4.482676 | 2.578051 | 2.578703 | 10 / 0 |
| 1024 | Linear bank, exact reduced time evolution | 0.561659 | 17.681166 | 1.675830 | 1.675830 | 0 / 0 |
| 1024 | Linear bank, original discrete weak step | 0.580536 | 24.580228 | 1.675830 | 1.675830 | 0 / 0 |
| 1024 | Current nonlinear ROM | 12.318828 | 36.325427 | 4.555479 | 4.555461 | 0 / 0 |
| 1024 | Same-grid direct FOM | 1.034563 | 25.068664 | 0.000350 | 0.000000 | 0 / 0 |
| 1024 | Coarse direct FOM, interpolated | 0.313116 | 24.364979 | 2.577910 | 2.578073 | 10 / 0 |

## What changed and what was charged

Factor the learned spatial bank as $G=QR$. If $B$ maps bank coefficients to sine moments, define $C=BR^{-1}$. Free bank coordinates obey the continuous weak least-squares system

$$\dot y=-\nu C^+\Lambda C y,\qquad u=Qy.$$

Small reduced propagation maps are precomputed for the requested times. Online evaluation projects the supplied full field, applies those maps, and reconstructs every requested full field. There is no iterative nonlinear fit or timestep solve. This removes the nonlinear head restriction and uses more independent state coefficients.

The discrete linear control precomputes powers of the linear least-squares step obtained by replacing the original head with free coefficients. It retains the old time formula. A smaller-step control is recorded for accuracy only.

GPU timings start with the supplied full initial field on the GPU and finish with all full outputs ready on the GPU. Projection, initialization, evolution and full reconstruction are charged. The host column adds actual full input/output transfers from the same invocation. Offline learned-bank/operator assembly, checkpoint training and compilation are excluded and retained in the raw JSON. No data-generation descriptors enter any solver.

The coarse FOM uses 16 intervals and physically aligned interpolation to every requested output node. Its full input is restricted on the GPU in this run, so its host protocol differs from the earlier heat transfer report. Ratios to earlier allocations must not be computed.

These linear methods retain a full-grid initial projection and full-grid reconstruction. Their small evolution maps are independent of mesh size; total full-query cost is not constant with resolution.

## Validation and provenance

GPU job `3511417` on `NVIDIA A100-PCIE-40GB`, node `pax003`, scientific source `73fdaa88eb754470539f6ff9dfaa9422cb00255a`. Frozen checkpoint `expanded_seed790715`; family `single_bc_poly_gaussian_v1`; diffusivity `0.02`; cohort seeds `[790711, 790716]`. Float64 and highest matrix precision were verified, with GPU preflight and no rejected runtime warnings.

The independent NumPy/SciPy audit checked 492 unique preserved fields, 900 paired timing/error invocations, and 71712 metric entries. Maximum recomputation discrepancy: 5.26245713672e-14. Independent QR-based reduced-operator checks have maximum relative discrepancy 2.66367551041e-14.

Reference meshes: `[1024, 2048]`. Maximum current-relative restricted reference difference: 1.62810884586e-12, below the declared empirical budget 0.0001. This is not a certified continuum error bound. Restricted reference pairs are preserved; finer source arrays retain hashes and regeneration seeds.

Maximum linear discrete-step half-step discrepancy: 0.00030740759169. Nonstationary nonlinear initial-start groups / steps: 0 / 0. Counts include repeated invocations and both attempted initial starts; stationarity is distinct from field accuracy.

All 60 NMROM case/mesh fields match the previously archived frozen-head rollout to a maximum relative difference of 2.90530056652e-15. This verifies the comparator fields; its timings come entirely from the new allocation.

Above 1.5 times the median of the same case/method/mesh repetition group; none excluded.

Transport and archived members passed checksums, and the exact remote job directory was removed. The heat branch remains separate as previously requested. No earlier measurement is retracted; this is a new reduced method and a new paired comparison.

[Raw results](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/linear05/archive/outputs/results.json) · [Independent audit](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/linear05/analysis/audit.json) · [Configuration](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/linear05/archive/experiments/mr-heat2d/config-linear.json)

## Glossary

- **Intervals:** cells per spatial axis; the boundary values are identically zero and interior arrays are returned.
- **ROM / NMROM / FOM:** reduced model / nonlinear-manifold reduced model / full-grid solver.
- **Learned bank:** frozen spatial functions shared by the linear and nonlinear reduced models.
- **Coefficient / latent variable:** independent linear weight / coordinate passed through the nonlinear head.
- **Weak test moment:** a smooth sine-weighted average used to measure the PDE residual.
- **GPU query / host query:** full accelerator input-to-output time / the same invocation including input and output transfers.
- **Median:** middle timing value over all retained cases and repetitions at that mesh and method.
- **Worst physical relative error:** largest full-field Euclidean error divided by the current reference norm, across all cases, repetitions and output times; reported as a percentage.
- **Worst same-grid relative error:** the same statistic against the exact-in-time discrete solver on the requested mesh.
- **Worst initial error:** largest relative error of the fitted/projected initial field.
- **Outlier:** a timing exceeding the stated within-case threshold; retained in the reported medians.
- **Exact reduced time evolution:** matrix exponential of the fixed reduced linear differential equation; it still has spatial approximation and floating-point error.
- **Discrete weak step / half-step:** the original Crank–Nicolson weak least-squares formula / its smaller-timestep control.
- **Coarse FOM:** a direct solver on a smaller grid followed by interpolation to the requested output grid.
- **Reference refinement:** agreement between finer spectral grids; empirical evidence, not a rigorous bound.
- **Stationarity:** satisfying the nonlinear objective's gradient stopping rule, not proof of global optimality or physical accuracy.
- **Development cohort / checkpoint:** examples used for method development / saved frozen learned weights and codes.
