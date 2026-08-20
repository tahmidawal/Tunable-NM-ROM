# Burgers 1e-3 / 10x checkpoint log

## 2026-08-19 — intake and preregistration

Branch `exp/2026-08-19-burgers-1e3-10x` began at audited Burgers hybrid/FOM
commit `1752d9ee718e7bd1295d1d7cf9dc14e12257c611`.  The complete canonical lab log
and repository operating rules were read before action.  No new numerical job
preceded `PRE-REGISTRATION.md`.

The inherited H160 group-FiLM floor decomposes to an initial-condition/boundary
fit of `1.824e-2`, an oracle inferred-latent trajectory error of `7.688e-3`, a
full weak error of `1.039e-2`, and EQ errors of `1.555e-2` at m=256 and
`1.164e-2` at m=512.  This motivates the preregistered transported analytic
Hermite-Gaussian decoder rather than another closed FiLM width/group sweep.
