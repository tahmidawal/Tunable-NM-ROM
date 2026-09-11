# Direct CP algebra transfer to the frozen heat NMROM

Prospective development ablation of the historical CP implementation mechanisms. No performance gain is established by this design.

The user requested applying the recovered CP techniques to the current method. Continue the approved heat worktree from `96133d1`. Freeze the checkpoint, latent dimension, original two-start initialization policy, weak objective, Crank–Nicolson steps, tolerance and output times. Use the existing development cohorts and mesh endpoints in `config-cp-algebra.json`. All full-input and full-output timings and errors come from the same invocation. Final paper cases remain closed.

## Factorial comparison

- Original weak-Jacobian assembly with a general LU solve.
- The same assembly with a Cholesky solve only.
- Precomputed bank-space Gram normal equations with LU.
- The contracted normal equations with Cholesky.

Rerun the free-bank linear control and same-grid/coarse direct FOM controls in the same allocation. The linear control is a distinct model class. The nonlinear controls all share the same extra diagnostic outputs so timing contracts match.

## Algebra

For the normalized weak objective $r=(B h(z)-b)/s$, where $s=\max(\|b\|,\epsilon)$, assemble $S=B^\top B$ offline. With $D=\partial h/\partial z$,

$$H=D^\top S D/s^2,\qquad g=D^\top B^\top(Bh-b)/s^2.$$

Anchor the contracted gradient at the initial head value and actual initial weak residual to reduce cancellation. Evaluate the objective with the explicit small weak residual throughout: subtracting quadratic forms to compute a tiny squared residual is unnecessary here. Thus this contracts the normal equations, not the stable loss evaluation. The fixed Gram is supplied as a runtime argument and is never rebuilt inside an LM iteration.

The damped system is positive definite in exact arithmetic. Cholesky changes its factorization, not its damping or stopping criteria. The generic and contracted variants must retain actual termination reasons and all attempted iterations. Rounding may change an acceptance decision; field and latent parity are measured, and iteration counters are compared rather than assumed equal.

## Verification and reporting

Before cluster work, run an exact least-squares fixture with a nonlinear head and a sub-minute full-query smoke against the frozen decoder. On the cluster save every internal latent state, both fitted initial states, every termination diagnostic, paired timing repetition arrays and all requested fields. Independently reconstruct residuals and gradients from the original weak objective at every saved internal state, recompute field errors, check the precomputed Grams, and compare all variants with the same-job original NMROM. Also verify that the original control reproduces earlier archived fields.

Report every declared arm, GPU and host medians, outlier counts, full-cohort worst error, parity deltas and stationarity failures. Setup and compilation remain separate. Source hashes, GPU preflight, float64/highest precision, archive checksums and private-directory cleanup follow the repository protocol. No graph flags, counted loops, encoder/head retraining, added amplitude variable, approximate quadrature or new time integrator are part of this ablation.

## Glossary

- **CP / NMROM / FOM:** tensor-product decoder / nonlinear-manifold reduced model / full-grid solver.
- **Weak objective:** residual averaged against smooth test modes.
- **Bank / head / latent:** fixed spatial functions / nonlinear coefficient map / compressed solved coordinates.
- **Gram / normal equations:** precomputed inner-product matrix / small system defining the least-squares update.
- **LU / Cholesky:** general / positive-definite matrix factorization.
- **LM / stationarity / parity:** damped nonlinear least squares / meeting the gradient stopping rule / agreement between implementations.
- **Factorial comparison:** test each change separately and jointly.
- **Invocation / outlier:** one timed solve producing the graded fields / repetition above the declared within-case threshold, retained in the results.
