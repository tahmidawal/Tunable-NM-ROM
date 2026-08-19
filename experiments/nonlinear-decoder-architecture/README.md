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

`nda_summarize.py` and `nda_figures.py` regenerate the tables and figures from
pulled JSON artifacts. Numerical conclusions remain provisional until the
cluster logs, timing arrays, and multi-seed runs have passed the repository
audit.
