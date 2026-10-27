**BLOCK** — the algebra is consistent, but the diagnostic has concrete controls and accounting gaps.

- **Large JAX captured constants:** [pilot.py:120–124](experiments/burgers-multilevel/pilot.py:120) builds a zero-argument JIT closing over `data`, production states and anchor states, anew for every invocation. `Psi` alone is 128 MiB; `Gq` and `Gs` add 64 MiB. Pass these arrays as runtime arguments to one reusable compiled function.
- **Zero-budget control never executes a zero-budget solve:** [pilot.py:71](experiments/burgers-multilevel/pilot.py:71) only measures stationarity at initialization. It cannot detect ignored budgets or an incorrectly executing solver. Invoke the actual solver with budget zero and check unchanged state, zero updates and failed convergence.
- **Finite-difference control is missing:** [pilot.py:57–72](experiments/burgers-multilevel/pilot.py:57) checks analytic derivatives against autodiff, but never executes the finite-difference check promised in `DESIGN.md`. The synthetic smoke also lacks it.
- **Provenance is incomplete:** [pilot.py:36–45](experiments/burgers-multilevel/pilot.py:36) hashes selected constructed arrays, but does not pin source files, checkpoint, rotation or native bank blocks. `SOURCE_COMMIT` may be absent, and the experiment directory is currently untracked. This does not implement the required pre-solve source/input manifest.
- **Acceptance-check cost is excluded:** [pilot.py:114–124](experiments/burgers-multilevel/pilot.py:114) stops the timer before independent certification and agreement checks, without recording their duration separately. The reported seconds measure the query, but cannot support a complete accepted-query cost claim.

Static checks that passed:

- Fine and restricted Jacobians match the fixed weighted residual; the coarse objective is its exact pullback.
- Fine histories stay fixed during each intervention. Coarse damping does not overwrite fine damping; rejected coarse updates preserve the post-pre-step state.
- Jacobian counters match the explicit evaluation paths: each LM contributes `iterations + 1`, including rejected trials; intervention checks are added separately.
- Timed outputs are synchronized, and saved certificates and agreement distances use that same invocation’s states.
- Predictor selection and the two BE startup steps remain inherited.

No edits, jobs, imports or numerical tests were run. These findings concern the development algebraic diagnostic, not PDE accuracy.