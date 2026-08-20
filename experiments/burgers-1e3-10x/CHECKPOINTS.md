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

## 2026-08-19 — D0 queue correction

Job `2667361` (`ctol_b10_d0`) remained pending at zero elapsed because its H100
request targeted the cluster's single mixed H100 node.  It produced no log or
scientific output.  It was cancelled by explicit numeric ID through the
repository-safe `experiments/cost-to-tolerance/cluster/cancel.sh`; the empty
remote staging directory was then removed after the queue check.  The identical
D0 science was resubmitted once in fresh directory `d0_r2` as job `2667377`,
requesting H200.  Job `2667361` is infrastructure-only and consumes no scientific
choice, but remains part of the operational record.

The independent preregistered FOM-calibration cell was submitted as job
`2667374` in directory `fom_cal`.  It uses only seed-20260822 calibration cases,
never model-validation or confirmation data.
