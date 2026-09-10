# Direct linear evolution in the learned heat bank

This is a prospective development comparison; no new result is established by this design. Frozen checkpoint and declared cohorts/settings are read from `config-linear.json` and the earlier transfer configuration.

The primary linear ROM frees the learned spatial-bank coefficients. With $G=QR$, set $u=Qy$ and $C=BR^{-1}$, where $B$ maps bank coefficients into the original smooth sine test moments. The exact continuous weak least-squares dynamics are

$$\dot y=-\nu C^+\Lambda C y.$$

Precompute the small matrix exponential for each requested output time. The query projects the supplied field into $Q$, applies those reduced maps, and reconstructs every requested full field. The entire evolution has no nonlinear optimization. This is a linear ROM with learned spatial functions, not the nonlinear manifold ROM and not a matched-state-dimension comparison.

A second control replaces the nonlinear head in the original discrete objective with free coefficients:

$$y_{j+1}=C^+\operatorname{diag}(f)C y_j.$$

Here $f$ is the original Crank–Nicolson sine-mode factor. Precompute powers of this small step operator, preserving that discrete weak least-squares problem. Retain a half-step accuracy control. Check weak-operator rank and generator stability; no pointwise PDE residual or empirical quadrature is introduced. Sine test dimension exceeds the free coefficient dimension as declared by the checked operator assembly.

Rerun the current expanded-head NMROM and the exact-in-time same-grid discrete sine transform FOM in the same GPU allocation. Also retain the previously selected coarse FOM with full-grid interpolation. Primary timings run from full supplied GPU input to blocked full GPU output. Record input/output transfer and host totals from the same invocation separately; the coarse FOM now restricts its full supplied input on the GPU, so this is a new device contract, not a replay of the earlier host restriction contract. Source descriptors only generate inputs and references.

All fixed development cases and meshes run without retraining or tuning. Use the same frozen checkpoint across meshes, record setup/compilation separately, burn in before each paired block, retain repetition arrays, and report median times and outliers. Preserve the actual timed predictions and recompute errors against same-grid and refined continuum-spectral reference fields. Reference refinement is empirical. Preserve restricted reference pairs and hashes of discarded finer arrays. Final paper cohorts stay closed.

## Glossary

- **Bank:** learned spatial functions shared by both ROMs.
- **ROM / NMROM / FOM:** reduced model / model restricted to a nonlinear decoder manifold / full grid solver.
- **Weak moments:** smooth sine-weighted averages of fields and PDE residuals.
- **Coefficient / latent dimension:** number of freely evolved bank weights / compressed nonlinear coordinates.
- **Matrix exponential:** exact propagation of the chosen constant linear reduced differential equation, with floating-point numerical error.
- **Crank–Nicolson:** the existing centered time-discretization formula.
- **Checkpoint:** frozen trained weights and training codes.
- **Development cohort:** previously declared examples used for method development, not final confirmation.
- **GPU / host:** accelerator / CPU memory. Full-output transfers are separately visible.
- **Refinement:** comparing finer reference grids or smaller timesteps; not a certified continuum bound.
