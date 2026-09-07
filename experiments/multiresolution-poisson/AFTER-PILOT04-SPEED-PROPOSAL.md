# Separate Poisson source-projection and initialization proposal

This is a proposed later speed experiment. Neither change enters the current
coverage-by-loss training factorial or its fixed evaluation controls.

First compare the current two skinny sine matrix products with a forward
orthonormal sine transform followed by gathering the same retained smooth modes
and applying the same inverse-eigenvalue weights. The retained shells, weak
objective and source input are unchanged. Check projected coefficients, final
fields, normalized gradients, stopping reasons and solver counters with frozen
numerical tolerances. Time complete host-source-to-host-field queries, keeping
the existing projection and efficient full DST controls in the same GPU job.

Separately compare mean-training-code initialization with a nearest-code lookup
in cached weak decoder predictions. For every training code $z_i$, cache
$B h(z_i)$ when the mesh operators are built. Given the projected source $f_m$,
choose the code minimizing $\|B h(z_i)-f_m\|_2$. This uses only decoder codes and
the full source's weak coefficients, with no Gaussian descriptors or reference
answers. Charge the distance calculation and lookup in the complete online
query; record cache bytes, setup time, selected code index and resulting solver
status. The weak objective, damping, trust-radius rule and stopping criteria
remain fixed, and failures or different local minima remain visible.

A controlled factorial would cross these two changes while preserving the
original projection/mean-code baseline. The proposed budget is frozen in
`speed_factorial_proposal.json`; actual endpoint selection awaits the completed
training study and coordinator review. Retain the original checkpoint and select
one scheduled continuation using every development case on both meshes. Among
complete endpoints whose tighter generic weak solves are all solver-valid,
minimize the worst empirically reference-adjusted physical error across those
cases and meshes, breaking ties by their median error and then model name. The
chosen endpoint and content hash will be generated from the full run JSON, with
no selection based only on an inspected hard case.

The main factorial uses the enlarged smooth test space and one shared guarded
fused solver. Primary timing covers the projection-by-initialization variants
alongside same-grid and coarse-grid DST. Direct coefficient and final-field
agreement checks accompany the panel. Separate fully recorded single-call
stationary controls cover every model, implementation variant, case and mesh;
those calls do not enter speed selection. Every source in both development
cohorts is retained, with the checkpoints frozen throughout.

Projection variants must satisfy a frozen coefficient-agreement tolerance and
scale-aware final-field agreement for the same initialization. Initialization
variants are allowed to take different trajectories or reach different local
minima; their fields, objective values, gradients and stop statuses are evaluated
directly. The existing residual-reduction target is relative to each query's own
initial residual, so changing initialization also changes its absolute stopping
threshold. Preserve that original rule, report the threshold explicitly, and use
the tighter stationary control to isolate initialization effects at convergence.

The coordinator will review the concrete selected checkpoint and complete
budget before this proposal is implemented or submitted.

## Plain-language glossary

- **Projection / retained mode:** extracting smooth source coefficients / one
  kept sine test function.
- **DST / shell:** discrete sine transform / complete group of tied sine modes.
- **Weak prediction / cache / lookup:** decoder field represented in the PDE's
  smooth tests / stored reusable array / selection from that array.
- **Initialization / code / local minimum:** solver starting point / compact
  neural coordinates / a locally best objective value that may depend on start.
- **Factorial / control:** independently crossed changes / unchanged comparison.
