# Handoff — ns3d-grok

Updated 2026-09-21 after diag01 was pulled and diag02 was submitted as job 4140791.

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

Nothing until the scheduler starts it. Job **4140791** `ns3dgrok_diag02` is pending
(Priority). Directory
`/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag02/`. Source commit
`e7b9dba5`. One job for this lane.

## Next step

When 4140791 exits 0, confirm `jax_backend=gpu`, pull `summary.json`,
`verify.json`, and `OUTPUTS.sha256`, rerun `verify_diag02.py` locally, delete
the remote directory, and compare the frozen-center rollout with the 5% bar.
Do not open seed 202609203.
