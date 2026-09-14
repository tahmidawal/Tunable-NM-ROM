# Same-grid iterative Burgers multiresolution comparison

This is a predeclared development comparison for the user's September request. The numbers in the configuration specify the experiment; measured results are not yet accepted.

Continue the current frozen separable decoder, fixed physical Gauss initial fitting, backward-Euler time stepping and sign-dependent upwind weak advection. The bank has $r=512$ features, the latent has $k=16$ coordinates, and the weak projection has $M=64$ modes with $m=256$ weighted samples. The weak mass and diffusion terms are preassembled. Advection uses the existing sampled full sign-upwind operator, **not** the historical smaller-bank polynomial tensor. No network is retrained. The existing quadrature fit evaluates advection on decoded training states, never time-step PDE residual snapshots; it is rebuilt at each mesh exactly as in the accepted current solver.

The complete configuration is `config-iterative.json`. It fixes the four existing development cases from seed 7090702, requested intervals 64/256/1024, three repetitions, and outputs at $t=0,0.05,\ldots,0.25$. All methods accept the same full supplied initial field. Physical descriptors only regenerate test inputs. Initial fitting uses one start, budget 180, and the original small-improvement threshold; evolution uses budget 30 and threshold 0.01. These exits are not stationarity certificates. Every original termination category remains visible.

The main FOM is compiled Newton–BiCGStab on the requested grid with the same backward-Euler step size $\Delta t=0.005$. Newton tolerances 0.01, 0.003, and 0.000001 form the predeclared sensitivity. The first two use inexact linear tolerance 0.5; the tight arm uses 0.00000001. The older dense sine-transform Helmholtz preconditioner is retained at Newton tolerance 0.003 as a named control. The FFT-based Helmholtz implementation is the primary current iterative baseline. Newton stops on its measured relative residual, not a fixed iteration count. A charged true linear-residual check records every Newton correction's BiCGStab result. A full-step previous field is retained at each output for independent backward-Euler residual verification. These additional diagnostic outputs are included in GPU blocking, so comparisons concern the instrumented solvers.

Inputs and all public output fields are device resident for the primary timer. A second timer surrounds the same invocation with actual host-to-device input transfer and device-to-host public output transfer. Compilation, setup, output scoring and archival are excluded. Before every timed invocation the GPU is burned in; method order is shuffled with the recorded seed. Every repetition retains its actual timing, complete-field hash, errors and convergence evidence. Exact duplicate fields may reference the first preserved artifact. Medians and within-case timing outlier counts are reported with no timing exclusions.

Fine references are regenerated on the cluster from the same physical seed. Three spatial and temporal refinement levels, including 4096 intervals and $\Delta t=0.0003125$, preserve the previous empirical continuum comparison. Every requested mesh uses its own full-grid norm and refinement margin. A five-percent target requires all cases' fixed-initial errors plus their empirical margins to pass, the margin to be below one tenth of target, and refinement differences to decrease. This is an empirical development test, not a rigorous continuum error bound, matched-error equality, globally optimized classical envelope or final-cohort result. Current-relative errors are also retained separately.

One A100 allocation uses float64 and highest matrix precision, mandatory GPU preflight, its own private paralab directory, and a two-hour ceiling. Expected runtime is about half an hour including references. New kernels are checked locally in under a minute, source is frozen before staging, and all timed fields/metrics/provenance are independently audited after checksum collection. The exact completed remote directory is removed after successful collection. Existing worktrees remain separate.

## Glossary

- **Intervals / nodes:** grid spaces per axis / points including both boundary walls; a grid with $L$ intervals has $(L+1)^2$ nodes.
- **FOM / NMROM:** full-grid PDE solver / nonlinear manifold reduced model.
- **Bank / latent / weak modes:** frozen spatial features / low-dimensional unknown coordinates / smooth functions averaging the PDE residual.
- **Gauss initial fitting:** approximate physical least squares from interpolated supplied-field samples at fixed Gaussian quadrature nodes.
- **Upwind / preassembled:** a spatial difference selected by the local field sign / a reusable operator computed before a query.
- **Newton–BiCGStab / preconditioner:** nonlinear iteration with iterative linear corrections / an approximate inverse that speeds those corrections.
- **FFT / dense sine transform:** fast-transform / matrix-product implementations of the same Helmholtz inverse.
- **GPU / host timing:** elapsed complete query with device-resident public fields / the same query including public field transfers.
- **Fixed-initial / current-relative error:** trajectory field-error norm divided by initial truth norm / by truth norm at that time.
- **Empirical margin / refinement:** observed reference difference allowance / repeating a numerical solve on finer space or time grids.
- **Budget / improvement exit:** optimizer iteration cap / termination after a sufficiently small step or decrease, without a stationarity claim.
- **Development cohort / final cohort:** cases used for method development / an unopened independent confirmation set.
