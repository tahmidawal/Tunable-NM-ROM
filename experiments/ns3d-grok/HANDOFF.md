# Handoff — ns3d-grok

Updated 2026-09-21 after the local smoke of `diag01` passed.

## State

Worktree `2026-09-21-ns3d-grok`, branch `exp/2026-09-21-ns3d-grok`, forked from
`exp/2026-09-20-paper-ns3d` at `a104a637`. Namespace
`/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/`. Final seed 202609203 is
closed. Nothing is running.

Inherited final07 (job 4027788, 32 held-out cases, same-grid relative $L^2$):

- NM-ROM $q=0$: initial worst 12.21%, evolved worst 20.34%, 31/32 over 5%.
- NM-ROM $q=256$: evolved worst 18.92%. The correction ladder moves the worst
  case by about one point.
- Free-bank Galerkin $R=1536$: initial worst 2.03% (0/32 over 5%), evolved
  worst 10.17% (16/32 over 5%). Every case grows.
- POD weak and POD Galerkin agree at ranks 64 and 320. Rank 320 already misses
  the initial field by 21.5% worst.
- CNAB2 $\Delta t=0.01$ passes at 2.47% worst evolved, about 3.2 ms.

So the head sits far above the bank floor at $t=0$, and CNAB2 inside the bank
still drifts from 2% to 10%. Capacity05 POD-3072 development snapshot floor was
3.84% worst and was never integrated.

## What is running

Job **4139559** `ns3dgrok_diag01`, partition `gpu`, pending (Priority) at
submit. Directory `/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag01/`.
Source commit `010cf066`. One job for this lane. Do not submit another until
this one finishes and its remote directory is deleted.

Local smoke `experiments/ns3d-grok/runs/smoke01` finished in 5.6 s on the GB10.
`verify_diag.py` passed. That smoke is four trajectories at $N=8$ and is not
a result.

## Next step

When 4139559 exits 0, confirm `jax_backend=gpu` in the log, pull
`summary.json`, `verify.json`, and `OUTPUTS.sha256` with checksums, recompute
is already inside the job, delete the remote directory, then read the
development floor against the Galerkin rollout before choosing a fix. Do not
open seed 202609203.
