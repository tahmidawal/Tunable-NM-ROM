# Current frozen Poisson decoder against the older iterative FOM

This development comparison ports the original unpreconditioned CG algorithm into the current full-source/full-output timing contract. The config freezes the existing relative-loss checkpoint, source cohort, mesh ladder, weak solver, tolerances, and repetitions before timing. This is a current-model extension, not the historical decoder replay.

The source field is supplied on the requested grid. The NMROM projects that field onto smooth sine tests, chooses a starting latent code using only cached training-code predictions, solves the existing weak nonlinear objective, and decodes the complete requested field. Its weights stay fixed across grids. Both projection and full output remain grid dependent; only the contracted weak solve is independent of grid size. No Gaussian descriptors, answer-dependent starting guesses, or empirical quadrature enter this query.

The primary classical baseline is same-grid, zero-start, unpreconditioned CG for the positive five-point Dirichlet Laplacian. It retains a relative residual stop, an iteration cap and a charged final true-residual check. The complete iteration loop is compiled. The historical tight tolerance and the predeclared looser controls are all retained. A direct sine-transform solver is a labeled diagnostic and remains relevant to the limits of the iterative comparison.

Timing begins with a supplied GPU source and ends with the complete GPU field and solver diagnostics. The same invocation also records host input/output transfers and host total. Every timed call follows GPU burn-in; alternating order, all repetitions, output hashes, actual fields, numerical stops and setup/warm-up costs are retained. No repetition is dropped. Aggregates use medians of case medians; paired speed ratios use the median of per-case ratios, with the ratio of cohort medians also named separately.

Primary accuracy is $\|u-u_{\mathrm{ref}}\|_2/\|u_{\mathrm{ref}}\|_2$ on the full requested mesh. Independent nested finite-difference sine-transform references are restricted to that mesh. Their observed difference $\delta$ gives an explicitly empirical adjustment $(e+\delta)/(1-\delta)$, not a rigorous continuum bound. Same-grid discrete and common-observation errors remain separate. All previous development sources are included, including difficult cases; physical accuracy failures cannot be hidden by a stationary weak solve. The earlier selected model missed the nominal development target, so a runtime advantage may still fail that accuracy requirement.

## Plain-language glossary

- **NMROM / CG / FOM:** nonlinear reduced model / conjugate gradients / full-order solver.
- **Frozen checkpoint / latent code / bank:** fixed neural weights / compact solved coordinates / learned spatial features.
- **Weak test / projection:** a smooth function used to measure the equation / conversion of a supplied field into those smooth coefficients.
- **Intervals / nodes:** grid cells along one axis / grid points including both boundaries; the latter count is one larger.
- **Tau / stationarity / stopping:** initial-residual reduction target / small normalized objective gradient / the recorded numerical reason for returning a solution.
- **Development / target / empirical reference:** already-open experimental cases / requested worst relative accuracy / independently refined numerical truth without a proven error bound.
- **GPU / host / median / outlier:** accelerator / CPU memory / middle sorted value / a timing repetition above the declared multiple of its own case median, retained in every aggregate.
