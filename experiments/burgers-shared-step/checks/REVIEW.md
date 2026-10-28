# Affine initialization source review

The independent Codex replay-lane agent `/root/latent_pair_evidence` reviewed `affine.py`, the retained-core diff, complete runner, analyzer and source stager on 2026-10-28. Reviewed implementation commit: `2b74658c8`; code-only mirror: `462fa76c9`.

The review found the active 16-column Jacobian, retained predictor tail, fine history and damping, original fused baseline, guards, and paired cost gate correct. It requested explicit affine/direct comparison of the initial code, initial field and initial-fit decisions on every development case. The runner and offline analyzer now check those comparisons.

Reviewer closure: “Your initial-state parity change closes my affine review: READY at 2b74658c8 / mirror 462fa76c9 for exact-bundle staging review, no remaining source blocker.”

Static Python compilation, retained-source AST invariants, and shell syntax checks passed. No numerical model execution or GPU result was produced by this lane agent. Exact staged-bundle and numerical checks remain required; no speed or accuracy conclusion is implied.

## Plain-language glossary

Active columns: derivatives with respect to the 16 coordinates allowed to change in the preliminary solve. Predictor tail: the other 48 coordinates kept from the fine temporal prediction. Initial-fit decisions: the original initialization's iteration count, exit reason and chosen starting-code index. AST: Python's parsed source structure, used here to compare preserved functions without executing JAX. Mirror: the backed-up code-only GitHub branch.
