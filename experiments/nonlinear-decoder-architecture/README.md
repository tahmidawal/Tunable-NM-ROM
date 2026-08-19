# Pure nonlinear decoder architecture study

This experiment compares the frozen fully modulated FiLM coordinate decoder
against compact residual and grouped-FiLM coordinate decoders. It covers
Poisson-2D and Burgers-2D using the established data, weak-form objectives,
empirical quadrature, and held-out tests.

Every candidate is fully nonlinear in the latent state and coordinates. None
has a POD component or fixed output basis. The residual family uses a
coordinate-only nonlinear stem followed by latent-modulated residual blocks;
the grouped-FiLM family preserves all-layer modulation while sharing modulation
coefficients within small channel groups. The optional transport arm learns a
low-dimensional translation and anisotropic scale before coordinate features
are evaluated.

Architecture and objective selection require every one of training seeds 0, 1,
and 2 to pass, not merely the three-seed mean. Poisson requires decoder, full
weak-ROM, and EQ weak-ROM mean errors at or below `6e-3`, with `EQ/full <= 1.05`.
Burgers requires decoder error at or below `8e-3`, full and EQ trajectory errors
at or below `1e-2`, and the same degradation ratio. Among passing rows the
selector minimizes quadrature points and then parameter count. End-to-end
deployment rows are evaluated separately because a fully converged validation
objective can still be budget-censored at a practical stopping tolerance.

`nda_summarize.py` and `nda_figures.py` regenerate the tables and figures from
pulled JSON artifacts; `nda_audit.py` independently checks checksums, staged
launch provenance, GPU/f64 guards, completion markers, and every persisted
timing repetition. Numerical conclusions are marked final only after that audit
passes.
