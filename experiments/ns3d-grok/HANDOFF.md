# Handoff — ns3d-grok

Updated 2026-09-21 after diag01 (job 4139559) was pulled, verified, and deleted
from the cluster. About to submit diag02.

## State

Worktree `2026-09-21-ns3d-grok`, branch `exp/2026-09-21-ns3d-grok`, forked from
`exp/2026-09-20-paper-ns3d` at `a104a637`. Namespace
`/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/`. Final seed 202609203 is
closed.

diag01 completed on an A100-40GB (`jax_backend=gpu`, commit `010cf066`, 11 min).
Local NumPy `verify_diag.py` passed. Remote directory deleted. The generated
table is `experiments/ns3d-grok/results/diag01.md`. Plain POD rank 3072 is the
first rank whose CNAB2 rollout stays under 5% on all 16 development cases. An
oracle per-time shift does it at rank 64. The centered spectrum has 237 modes.
That center uses future truth, so it is not a model.

Inherited final07, for orientation only: NM-ROM evolved worst 20.34% at $q=0$
and 18.92% at $q=256$; free-bank Galerkin evolved worst 10.17%; CNAB2
$\Delta t=0.01$ passes at 2.47%.

## What is running

Nothing yet. Next submit is one job, `ns3dgrok_diag02`.

## Next step

Submit diag02: centered POD, frozen centroid of $u_0$, CNAB2 Galerkin at
$\Delta t\in\{0.001,0.004,0.01\}$, development only. Do not open the final
cohort. After it exits: NumPy verify, checksum pull, delete the remote
directory.
