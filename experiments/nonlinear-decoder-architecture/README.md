# Pure nonlinear decoder architecture study

This experiment compares the frozen fully modulated FiLM coordinate decoder
against compact residual coordinate decoders with grouped late FiLM.  It covers
Poisson-2D and Burgers-2D using the established data, weak-form objectives,
empirical quadrature, and held-out tests.

The new decoder is fully nonlinear in the latent state and coordinates.  It has
no POD component or fixed output basis.  A coordinate-only nonlinear stem is
followed by latent-modulated residual blocks; the optional transport arm learns
a low-dimensional translation and anisotropic scale before the coordinate
features are evaluated.

All numerical results remain provisional until the cluster logs, timing arrays,
and multi-seed runs have passed the repository audit.

