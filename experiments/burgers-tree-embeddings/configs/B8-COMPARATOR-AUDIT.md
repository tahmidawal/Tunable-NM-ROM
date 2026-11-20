# B8 native comparator audit and implementation requirements

Read-only inspection of retained pinned author sources and existing B6/B7 evidence. This is an implementation audit, not a claim of trained comparator accuracy or PDE speed. The new CROM driver below remains an offline training/projection diagnostic until separately qualified.

## Provenance

All paths in the source table are relative to `results/b6native/output/sources/`; these are retained author files, not a newly downloaded revision. Their manifest is `results/b6native/output/result.json`, also retained in the owned cluster namespace's `archive/b6native/output/`.

| Source | Pinned commit |
|---|---|
| `dc3042/CROM_offline_training` | `39333f76ccf878dacc9ae425c185b76ed4e72c5c` |
| `peterchencyc/CROM_online_deployment` | `ccb8a6603318492442e32f9d254d420b63c86419` |
| `CMU-CBML/NeuralROMs.jl` | `35c55e44cdcc7eb902c6c0b6311fb6c524271f91` |

| Relative source | SHA256 |
|---|---|
| `crom_offline/run.py` | `5948975955b4342d51c659a1d3adf86c75d42379995815a3a17dadc1310918ba` |
| `crom_offline/cromnet.py` | `0e3cf05e9de775178cfb6f934cff411782f2c621f4ac19cd3f5eb987a0d3250b` |
| `crom_offline/util.py` | `384ffad79613bcb85a530477dcc9167aa2f2b8dd7eddad9d379b0e93a84ecd46` |
| `crom_online/online/NonlinearSolver.py` | `48a9f6f385ecb7fd0d2ff66ade315e8b7d10196d2bf898a0889448dc498cffc1` |
| `crom_online/online/Experiments/Burger.py` | `801fd326a97ae0255f5a4c1e5c8363d9603b4f16f58e29b5d00d255013a0bc73` |
| `snf/experiments_SNFROM/smoothNF.jl` | `f3b44c2122886ac17f27fe409382cc5cc07c7eb049f3e659988826b0b7ab6b2a` |
| `snf/experiments_SNFROM/compare.jl` | `be5d3610612ef11f0bc1c6839c2f1ff66602c741096a04bc81bc4236abbb659e` |
| `snf/experiments_SNFROM/cases.jl` | `6950cdf4a54f27ea15d7a14d98595ffa55aa2ccf3b217126fa5fc3798aa131cd` |
| `snf/src/train.jl` | `59b7fc70dbe347293fd4a052229be9e4fb6459b781d7ac65317d8b38aba50c8a` |
| `snf/src/metrics.jl` | `d8f9e0510656b65e10c5128fcea0e72631d97ad117821b2ca09bf0ac9239e9f2` |
| `snf/src/pdeproblems.jl` | `5dcae9bd3479a4a6efece7d8f588201aa7b070b879d1af8fb504c1f29df1e212` |

## CROM

Existing `train_crom_native.py` and `diagnose_crom_projection.py` already exercise author networks, objective/gradient parity, analytic decoder Jacobians, and the author's unmodified Gauss–Newton routine. The existing run's history is in `results/b6cromdeps/output/result.json`; it selected an early checkpoint under the lower learning-rate schedule and does not establish a strong comparator.

Author `crom_offline/run.py:123` sets base learning rate `1e-4`; line125 sets multipliers `[10,5]`. `cromnet.py:227` uses Adam and its LambdaLR schedule. `util.py:183` requires equal-length multiplier/epoch lists, so specify `[100,100]` explicitly rather than copying the CLI's mismatched default epoch list. `cromnet.py:242` defines five SIREN hidden layers; `scale_mlp=64` and input dimension2 give width128. Preserve CNN kernel6/stride4 and native reconstruction loss instead of substituting our child-training objective.

New entrypoint: `train_crom_expanded.py --out OUTPUT --train-jobs b8trainrefs --dev-jobs b8devrefs --epochs 200 [--resume-job JOB] [--latent 32]`.

The driver uses the same original and expanded trajectories, reconstructs the native uniform128² inputs from verified reference coefficients, and estimates normalization from training only. Development is never training. It verifies archived dependency/source hashes and copies the float64 preprocessing port into its own output directory. Native objective/gradient and analytic-Jacobian parity are repeated before training. All model arithmetic is float64 with TF32 disabled.

The first configuration is latent32, width128, batch4, base learning rate `1e-4`, author multipliers `[10,5]` for `[100,100]` epochs, and no encoded-error early stop. Every20epochs it retains full encoded development metrics plus independently scored native-GN projection on the declared primary/expanded subset. Selection uses the projected worst error, not encoded reconstruction alone. Projection failures remain recorded. Native solver thresholds are `1e-15`; the driver reports stationarity rather than claiming the solver's return automatically proves convergence. Full model, optimizer, scheduler and random-generator states are retained, allowing continuation in a new owned job with unchanged data/configuration. Earlier source outputs are never mutated. All-development projection and actual4096 output remain separate mandatory work.

The source `crom_online/online/Experiments/Burger.py` implements forced inviscid one-dimensional Burgers, not this scalar two-dimensional viscous problem. A tested problem adapter must supply the common initial field, Dirichlet handling, two-dimensional advection/diffusion and requested times while retaining native CROM projection/evolution. Preserve distinctions between native framework ports and architecture-inspired weak-solver controls. Do not treat failed reproduction as a defeated competitor.

### CROM scalar 2D online adapter qualification

`baselines/crom_scalar2d.py` is a new scalar viscous/Dirichlet problem adapter, not the author's original 1D example. It preserves the explicit physical Euler target and native GN projection in pinned `crom_online/online/online.py:216-233`. Fine-grid sign-upwind advection and five-point diffusion use spacing `1/16384` regardless of spatial sample count. Decoder values and code Jacobians must be in physical units. Boundary/outside values and Jacobians are exactly zero; this does not establish that an unconstrained trained decoder has a sufficiently smooth near-boundary trace.

Pinned `Experiments/Burger.py:121-123` updates `q` but not `q_sample`. The nonlinear solver's final `computeLossAndR` does refresh accepted samples, so this is **not evidence of a stale-state defect in the native nonlinear branch**. The linear branch lacks that solver side effect. The adapter explicitly refreshes samples on each RHS and accepted step, and its multistep regression checks against an independent dense predictor/projection recurrence.

The tiny CPU fixture imports unmodified author `Decoder.py` (SHA256 `5cbc11c2ea15a206a3de5efc676e78b2f6267979e91b5728133c4435b9a7c385`) and `NonlinearSolver.py` (SHA256 `48a9f6f385ecb7fd0d2ff66ade315e8b7d10196d2bf898a0889448dc498cffc1`). It passed dense RHS equality, first-order continuum convergence, boundary/Jacobian checks and three native-GN steps. These are implementation fixtures, not trained-model PDE accuracy or runtime results. No PDE comparator is admitted by them.

Native `1e-15` stopping thresholds remain the reference configuration. Report projection residual, scaled physical stationarity and whole-step field change; the last quantity is not the native final-iteration stopping increment. Trained-model timestep and sample-count convergence remain mandatory. The full-grid forward-Euler diffusion bound is approximately `1.86e-7`, but manifold-projected stability must be assessed rather than automatically imposing roughly 1.34 million steps or replacing native evolution with the tree weak stepper. Total accounting must include actual stencil decoding, all GN trials, initialization and requested full output.

Retained local fixture output (2026-11-19; CPU-only, float64, one Torch thread, absolute project Python through `jaxrun`, 55-second process cap; exit 0):

```json
{"status":"passed","native_decoder_used":true,"native_gauss_newton_used":true,"dense_rhs_max_error":0.0,"continuum_rhs_errors":[0.19357365792465026,0.09657052405842809,0.048223324827118524],"code_jacobian_fd_max_error":8.874295048810765e-11,"rhs_code_jacobian_fd_max_error":2.615922700677231e-09,"three_step_code_errors":[1.1102230246251565e-16,3.885780586188048e-16,4.996003610813204e-16],"final_projection_diagnostics":{"relative_projection_residual":0.009547691933101088,"scaled_stationarity":2.024914674060865e-17,"relative_pde_field_update":0.003184834401469053},"stale_sample_regression_passed":true,"physical_intervals":16384,"sample_points":1024,"stencil_union_points":5120,"pde_comparator_admitted":false}
```

## SNF-ROM

Operational evidence: `results/b7snff64/output/result.json` passes the native float64 GPU-gradient and scalar-equivalent RHS fixture. It does not establish that the complete native training stack is float64-ready. Julia1.10.10, the resolved Manifest and private Lux rule patch are retained; do not reinstall or re-resolve dependencies.

Known hashes: resolved Manifest `df1738f2fd04e499511562a8fb41c5962533d0dde8e51c622265ee78aa1ab866`; patched Lux rule `a9c05d770f09a29e9ec4b16dbe663b7e10667b480f67f6d541561997b24f2f1d`. Reuse through a new writable depot overlay and read-only archived lookups. `LuxCUDA` must be imported. Convert parameters and batches to float64 after GPU adaptation because the retained adapter can cast them to float32.

Native entrypoints are `experiments_SNFROM/smoothNF.jl:119` (`train_SNF`) and `compare.jl:106` (`train_SNF_compare`). The author 2D example is `experiments_SNFROM/burgers_fourier2D/snf.jl`; it uses smooth-weight regularization. These pieces still need a separate verified port:

1. JLD2 adapter with spatial coordinates `x[2,N]`, fields `u[outdim,N,Ncase,11]`, times, offline trajectory descriptors and metadata. Preserve independent trajectory splits. Descriptors can support the native offline hypernetwork but cannot become privileged inputs to the online solve.
2. `cases.jl:231` assertsFloat32; `smoothNF.jl` allocatesFloat32 spatial/parameter arrays; `src/train.jl:86` transfers parameters and line88 assertsFloat32. Convert arithmetic/data deliberately and preserve the exact diff. `src/metrics.jl:386` requires regularization scalar type to equal parameter type. Float64 weights with Float32 regularization arguments do not suffice.
3. `smoothNF.jl:39` normalizes all supplied fields before the split at line51. Estimate normalization on training trajectories only, then apply it to held-out data. This is a documented data-contract correction.
4. `smoothNF.jl:213` passes only training data to `train_model`, even though development arrays have been constructed. Native early stopping therefore does not provide held-out checkpoint selection. Add separate held-out evaluation without training on it.
5. Native `evolve_SNF` at line423 obtains the initial embedding from true descriptors via the hypernetwork. Under this study's input contract, fit the supplied initial field instead; count that initialization. Keep the native field derivatives, Galerkin projection and integration method.
6. `hyperreduction_idx` at line610 uses known solution fields through `compute_residual` at line676. Treat this as offline training-data selection, freeze its points before testing, and do not supply test solutions to that routine. The cached-index branch also contains mismatched variable names (`hyppath`/`hypfile`); any correction needs an explicit retained patch.
7. `src/pdeproblems.jl:110` implements two-component viscous Burgers. Equal components reproduce the scalar equation, as the fixture checks. A scalar/tied-component adapter and boundary-condition treatment still need full numerical validation before deployment claims.

A reasonable author-framework starting configuration is latent32 and width128 with the original sine decoder/smooth-weight regularization. The author wrapper defaults to hyperwidth8 (`compare.jl:118`) and decoder `hd=5` (`compare.jl:122`, meaning six hidden layers in `smoothNF.jl`). The broader training family warrants a prospectively documented larger hypernetwork, rather than silently retaining an eight-wide bottleneck. Preserve the author's smoothing terms and staged optimizer (`cases.jl:519`) and verify the Float64 loss/gradient before launching real training. These are recommendations, not completed SNF training or proven optimal settings.

## Comparison contract

Use matched verified trajectories and no final-test solutions until method freeze. Allow strong comparator training and retain unsuccessful recipes/ports. Compare complete solves on the same allocation/hardware with physical accuracy and cost from the same invocation. Include initialization, projection, integration, refinement/rejections where applicable, assembly and requested4096² reconstruction. Representation-only timing is not a PDE speed result.

Glossary: native means the pinned authors' implementation; port means a documented adaptation of that code; projection fits a field using a decoder's latent variables; encoded reconstruction uses an encoder without that subsequent fit; stationarity measures the remaining fitting gradient; hypernetwork maps offline parameters/time into latent codes; hyperreduction chooses fewer spatial evaluation points; matched data means the same underlying trajectories and times, with method-appropriate sampling recorded explicitly.
