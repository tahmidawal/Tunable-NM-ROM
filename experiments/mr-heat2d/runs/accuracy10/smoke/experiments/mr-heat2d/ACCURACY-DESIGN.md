# Heat accuracy through fixed-bank training continuation

This is a prospective development experiment; no improved accuracy or speed is
claimed. It continues the corrected heat branch at `1f576c9`, preserving the
existing nonlinear decoder, weak Crank–Nicolson method and Cholesky solver.

The current model has eight latent coordinates and 32 spatial functions. Earlier
unrestricted-bank dynamics motivates checking the bank first on every current
development case and output time. At the training mesh, compare its exact
orthogonal field projection and four-start stationary nonlinear field fits.
These are best-found fits, never certified global nonlinear optima. If any bank
projection exceeds the declared 3% current-relative threshold, halt this
head-only experiment and report the case before changing the bank.

The original 160 training trajectories are regenerated from seeds 790710 and
790713. The current expanded head initializes every arm. No spatial weights,
dimensions, data, or lookup-library size change. Each trained arm performs 24000
updates of 192 snapshots, with the same optimizer schedule and sampling seed.
The frozen head is the unchanged control. Uniform continuation samples all times
uniformly. Initial-only continuation places half the batch at the initial time
and half across later times. Initial-plus-tail continuation uses those same
sampled indices and additionally changes the objective. Writing $e_s$ for a
relative field reconstruction error, that objective is

$$L=\tfrac12\operatorname{mean}(e_s^2)+
\tfrac12\sqrt{\operatorname{mean}(e_s^4)+10^{-30}}.$$

The exact QR-compressed field metric includes the constant orthogonal error.
All endpoint checkpoints are saved before head-fit or rollout evaluation. No
development score chooses a checkpoint, update count, training case, or code.
The four original and eight fresh opened cases remain in evaluation; four new
development cases from predeclared seed 790920 provide additional confirmation.
This is still development, not the final paper cohort, which remains unopened.

Every endpoint is evaluated at 64, 256 and 1024 intervals with frozen weights.
All initial fits, 20 weak steps, and all six requested full fields are charged.
All attempted fit/step stationarity checks are retained separately from physical
accuracy. Same-job controls are tight CG, the existing loose CG setting, exact
sine-transform FOM, and unrestricted linear-bank evolution. All repetitions and
their full fields are saved; no timing is combined with accuracy from another
invocation. Full input/output transfers are reported separately from the blocked
GPU query. The reference is the existing nested continuum-spectral refinement.

Same-architecture accuracy improvement with unchanged small matrix dimensions is
the intended test. It does not guarantee fewer online iterations or a speed gain.
The independent audit reconstructs fields and weak stationarity with NumPy and
SciPy, verifies the frozen bank, source hashes, and complete retained coverage.

## Plain-language glossary

- **Bank / head / latent coordinate:** spatial functions / nonlinear coefficient
  map / compressed number solved by the ROM.
- **Current-relative error:** field error divided by the reference magnitude at
  the same output time.
- **QR / orthogonal projection:** a stable matrix factorization / best field fit
  available from freely varying all bank coefficients.
- **Uniform / initial-only / initial-plus-tail:** training with equal snapshot
  probabilities / more initial snapshots / that sampling plus extra hard-case
  emphasis in the loss.
- **Stationarity / best-found:** sufficiently small objective gradient / the
  best of the attempted numerical fits, not proof of a global optimum.
- **CG / sine transform / FOM:** iterative conjugate gradients / a direct modal
  solver / the full-grid numerical model.
- **Development / final cohort:** cases used to assess and improve the method /
  independent cases reserved for the eventual paper confirmation.
