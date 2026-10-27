# Independent ROM FE-adapter review and response

The physics reviewer audited source `4cee0ac24`. Dimensionless scaling, Stokes pressure
lift, H1 supremizers, inf-sup whitening, coefficient transport, deterministic eigenmodes,
pressure convention and mapped-point lifetimes were consistent. Two blockers remained:

- Source arrays were silently promoted to float64 rather than checking stored precision.
- The pressure Gram used ideal SVD samples while coupling used reconstructed FE coefficients.
  Cancellation near the retained singular cutoff could make these different bases.

The author fixed both at `97847a479` (mirror `1416756dd`). Original NPZ headers and original
byte bounds are checked before payload loading. Actual reconstructed FE modes are evaluated;
weighted per-mode discrepancy and actual Gram spectral deviation must meet the fixed
1e-10 checks. Inf-sup now uses the actual reconstructed pressure Gram. Failing consistency
checks do not change ranks/cutoffs. Synthetic near-cutoff examples deliberately reject
cancellation. Strict finite-real singleton extraction covers scalar FE values/measures.

Independent narrow review confirmed both fixes and all source-manifest hashes. READY as
corrected source infrastructure for future gated controls; no NGSolve execution or training
occurred. Source-bound G2 data manifests, mapped Jacobians, FE layout digests, curved mesh
reload evidence and actual FE runtime controls remain required before admission.
