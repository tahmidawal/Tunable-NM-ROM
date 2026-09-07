# Fixed physical cold fitting in the full Burgers query

This is a predeclared development follow-up to the audited cold-only result. It keeps the checkpoint, clipped Gaussian physical family, validation seed/case count and final-cohort exclusion unchanged. Both spatial and latent neural tracks are frozen.

The query accepts a dense host initial field and viscosity, and returns all six dense host fields through the same physical horizon. Gaussian descriptors only regenerate test inputs. Fixed Gauss–Legendre nodes sample the supplied field with charged bilinear interpolation; this is initial-state fitting and does not collocate a PDE residual. Its online fitting objective is

$$\min_z \sum_i w_i\left(g(x_i)^T h(z)-I_L[u_0](x_i)\right)^2.$$

Here $I_L$ is bilinear interpolation on the supplied grid. The physical nodes and positive weights are independent of $L$. QR transforms this sampled objective without changing its field-fit minimizer. The weak evolution, sign-dependent upwind advection and per-grid nonnegative quadrature remain unchanged.

The meshes have 256, 512 and 1024 intervals. Use the original four physical cases at seed 7090702 and three timing repetitions per declared configuration. The cold-only controls support a single starting code; a larger budget checks the previously retained budget exits. Predeclare these ROM settings:

| Cold rule | Fit budget | Starts | Timestep | Evolution improvement threshold |
|---|---:|---:|---:|---:|
| Original edge | 60 | 1 | 0.005 | 0.01 |
| Fixed Gauss | 60 | 1 | 0.005 | 0.01 |
| Fixed Gauss | 180 | 1 | 0.005 | 0.01 |
| Fixed Gauss | 180 | 1 | 0.005 | 0.001 |
| Fixed Gauss | 180 | 1 | 0.0025 | 0.01 |

The first pair isolates the initial fitting objective at matched settings. The other Gauss arms diagnose fitting budget, reduced timestep and evolution stopping. Initial/evolution statuses and all iteration arrays remain visible; budget and improvement exits do not prove stationarity or global optimality.

Retain every earlier FOM candidate: nested solver grids through the requested grid; timestep/Newton-tolerance pairs `(0.01,0.01)`, `(0.01,0.003)`, `(0.005,0.01)`, `(0.005,0.003)`, `(0.0025,0.003)`. The FOM is tolerance-adaptive and preconditioned; coarse solves charge interpolation to the requested output. Eligibility uses casewise measured error plus that case's empirical reference margin, and the margin must also meet the predeclared reference-budget fraction. Select configurations by the median of per-case repetition medians; speed ratios are medians of per-case FOM/ROM median ratios. All ratios are within this job.

Regenerate the six reference settings used in the prior follow-up, including the finest 4096-interval solve at timestep 0.0003125. Store their fields restricted to the finest requested mesh, along with residual/iteration arrays. Record reference differences and observed orders separately under each requested-grid norm. The reference remains empirical; no rigorous bound or final-cohort certificate is claimed.

After compilation/warmup, randomize configuration order with a recorded fixed timing-order seed and burn the GPU before each timed invocation. Timing includes the host-to-device input, initial fit, autonomous evolution, requested dense decoding and host output. Hashing, error scoring and archival happen after the timer stops. Reusable setup and compilation are recorded separately.

Retain every invocation's common-grid fields, errors, statuses and dense-output hash. Save each configuration/case's first actual dense output; subsequent exact hash matches reference that artifact, and any differing output is separately saved. This permits independent complete-grid norm and repetition-parity audits without storing identical dense arrays repeatedly. Same-grid tight-FOM fields are also saved. This corrects the archive limitation of the cold-only diagnostic.

One A100 job has a two-hour hard limit, with an expected duration comparable to the previous reference campaign plus per-invocation warmup. Keep the archive below approximately ten gigabytes. No extra weight training, family change or final-cohort use is part of this run.

## Plain-language glossary

- **Cold rule / fit budget / starts:** initial-field sampling method / maximum optimizer trials / number of training codes used as initial guesses.
- **Fixed Gauss / QR:** fixed physical Gaussian quadrature / an orthogonal triangular factorization preserving the sampled least-squares objective.
- **Weak evolution / NNLS / sign-upwind:** reduced PDE equations averaged against smooth tests / nonnegative least squares for advection quadrature / spatial differences selected by the local field sign.
- **Common-grid / complete-grid norm:** field error at shared observation nodes / at every requested output node.
- **Empirical margin / observed order:** estimated reference error from refinements / convergence rate inferred from three levels; neither proves a bound.
- **Configuration / paired ratio / median:** a fixed solver setting / full-model time divided by reduced-model time for the same case / the middle observation.
- **Dense output / artifact / hash parity:** all requested grid values / saved numerical file / equality of content fingerprints across repetitions.
- **FOM / ROM / A100:** full-order model / reduced-order model / the requested GPU hardware type.
