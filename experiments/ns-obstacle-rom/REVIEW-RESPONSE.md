# Independent infrastructure review response

Status: repair prepared for independent diff confirmation. The coordinator forwarded the independent Codex review of commit `b8e50edbf`; the principal algebra was consistent, with one blocking timing-evidence defect. This record does not certify FE integration, training or physical experiments.

## Blocking finding: mutable invocation evidence

The recorder hashed output arrays but retained references to the caller's arrays and accounting dictionaries. A solver reusing buffers could overwrite earlier scored fields and work counts while leaving their recorded hashes unchanged.

**Repair:** deep-copy the complete returned record immediately after each timed call, before hashing/storage and before any subsequent invocation. Convert the retained output values to arrays and mark them read-only. Work counts, nested accounting and component-time dictionaries come from the same independent snapshot. Record evidence-copy/hash time separately; caller-required output materialization stays inside the timed callable.

**Regression:** the synthetic smoke deliberately returns one shared NumPy array, one mutable work dictionary with a nested list, and one mutable component-time dictionary from both methods on every repetition. After collection it mutates every original object again. Each retained output must still match its original hash and expected call index, each accounting history must remain unchanged, and retained arrays must be read-only. The generated result lives in `CPU-SMOKE.json` under `timing_reused_buffer_regression`.

## Interpretation clarification: sampled rule versus online cost

The current compact rule changes convection tensor assembly, then uses the same online tensor dimensions as the full-rule control. Point-count reduction is not an online-solve speedup result. Assembly/setup must be charged, and a direct sampled evaluator would require separate implementation, review and measurement. This limitation is now stated in `API.md`.

No equations, thresholds, training data or physical admission status changed. Independent confirmation of this repair remains pending when this response is written.

## Subsequent FE adapter infrastructure — pending independent review

`fe_adapter.py`, `fe_algebra.py` and `FE-ADAPTER-CONTRACT.md` add the training-only FE bridge. The coordinator authorized source construction while G1 controls run; this does not release FE/POD/data/training execution. `FE-ADAPTER-SYNTHETIC.json` records only array/matrix checks, including rejection before NGSolve import. No FE integration pass is asserted.

Review should inspect pinned-library mesh/space/coefficient round-trip, named transformation lifetime and copied arrays, explicit-rule metric assembly, constant pressure reconstruction, coefficient-preserving POD orientation, H1 supremizer/full-velocity reference norms, eigenvalue-cluster truncation, fixed 64/16 enumeration, resource guards and provenance. The reference author has received the exact missing G2 exports. The actual producer/reloader test and all scientific gates remain open.

## FE adapter review corrections — source review of `4cee0ac24`

The coordinator forwarded two required corrections. The archive reader formerly converted arrays to float64 before validating their original dtype; it now inspects original NPY headers before any payload load or conversion and rejects every non-float64 member. Declared array allocation bounds use the accepted original eight-byte dtype, with arbitrary-precision shape products. The regression replaces `np.load` with a failing sentinel for invalid dtype archives, demonstrating rejection before payload loading.

The coefficient-to-POD back-transform can lose precision near the unchanged singular cutoff. Before FE matrix assembly the adapter now evaluates the actual velocity and pressure coefficient modes and checks per-mode weighted relative discrepancy and actual-Gram spectral deviation from identity, both at fixed `1e-10`. Failure reports diagnostics and stops; it neither changes ranks/cutoffs nor re-orthogonalizes the failed modes. Reference inf-sup now uses the actual reconstructed FE pressure Gram, matching the pressure coefficients in the coupling/supremizer solves. The synthetic near-cutoff velocity and pressure examples retain their original numerical rank but fail this reconstruction gate as intended; `FE-ADAPTER-SYNTHETIC.json` retains the measured discrepancies.

The separately observed pinned-library scalar tuple issue is addressed by `strict_scalar`: finite real scalars/singleton arrays or tuples are accepted, while larger/empty arrays, nonfinite values, complex values and strings are rejected. Scalar FE point evaluations use this helper. Multi-component values and gradients retain their explicit shapes.

The coordinator reported that the original scaling, supremizer, beta-whitening, eigenproblem and transformation-lifetime logic passed source review; those equations were not changed. Synthetic regression and syntax/diff checks pass. No NGSolve import or FE/training execution occurred. Narrow confirmation of this correction commit is pending; actual FE integration and G2 remain unpassed.

## A6 candidate frozen after reset recovery, 2026-10-28

The completed independent Claude consultation and Codex adjudication superseded adapter execution readiness. The accepted risks were coefficient reconstruction through small singular-value division and missing component copies in a single-vector eigenbank. The central A6 policy is now bound by exact design/campaign hashes in `MODEL-CONTRACT.json`; `FE-ADAPTER-SOURCES.json` preserves earlier bindings and records the inspected producer proposal.

`fe_pod_a6.py` constructs coefficient modes with pivoted mass-metric CGS2 and a small-factor SVD, retaining the original sampled rank decisions. Its pressure construction protects the normalized constant during both orthogonalization passes. `fe_controls_a6.py` supplies independent analytic orientation/closure and nonsymmetric-gradient checks. The adapter emits scalar eigenmodes in explicit component order through checked maps, labels incomplete spectral-count evidence honestly, and exports declared dimensionless fields with separately named physical geometry sidecars. One validated common Stokes lift is shared across snapshots; the corresponding FOM reuse is allowed, with setup charged offline.

The current `FE-A6-SYNTHETIC.json` verifies healthy and near-cutoff coefficient construction, actual Grams, singular spectra and registered-prefix residuals. It also exercises amplitude invariance, cluster-cut rejection, zero branches, permutation accounting, scrambled component DOF maps, cross-block rejection, normal-sign/closure rejection and gradient-transpose rejection. `FE-A6-PREVIOUS-REGRESSIONS.json` reruns original dtype/size/scalar/admission/inf-sup checks. These are float64 NumPy/SciPy checks with no NGSolve import, native FE execution, training or physical admission. The old quotient failure and the initial unprotected pressure-constant rejection remain in `legacy/` with hashes. No rank cutoff or physical threshold changed after those outcomes.

Exact independent A6 source review remains required. Actual pinned-library serialization, geometry normals, component maps, matrix assembly and FE eigen/supremizer controls remain unpassed. Genuine G2 training exports and a reviewed bounded launcher are still required before this bridge can run. The current two-dimensional reference normal failure is not resolved by this source freeze.
