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
original projection/mean-code baseline. Freeze checkpoint selection using the
completed training study before choosing the next experiment's configuration.
Retain the original checkpoint as a control so an improved implementation cannot
be confused with an improved model. The coordinator will review a bounded source,
case, mesh and timing budget before this proposal is implemented or submitted.

## Plain-language glossary

- **Projection / retained mode:** extracting smooth source coefficients / one
  kept sine test function.
- **DST / shell:** discrete sine transform / complete group of tied sine modes.
- **Weak prediction / cache / lookup:** decoder field represented in the PDE's
  smooth tests / stored reusable array / selection from that array.
- **Initialization / code / local minimum:** solver starting point / compact
  neural coordinates / a locally best objective value that may depend on start.
- **Factorial / control:** independently crossed changes / unchanged comparison.
