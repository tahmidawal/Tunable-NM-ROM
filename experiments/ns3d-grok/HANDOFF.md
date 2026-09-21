# Handoff — ns3d-grok

Updated 2026-09-21 after diag02 showed a frozen initial center misses, before diag03.

## State

Worktree `2026-09-21-ns3d-grok`, branch `exp/2026-09-21-ns3d-grok`. Namespace
`/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/`. Final seed 202609203 is
closed.

diag01 (job 4139559, commit `010cf066`) and diag02 (job 4140791, commit
`e7b9dba5`) are pulled, NumPy-verified, and deleted from the cluster. Tables:
`results/diag01.md`, `results/diag02.md`.

A plain POD needs rank 3072 before CNAB2 stays under 5% on development. An
oracle per-time shift does it at rank 64. Freezing the centroid of $u_0$ is
already past 5% at rank 64 on the first saved time. The time step does not
move that number. Both remote job directories have been deleted.

## What is running

Nothing until `ns3dgrok_diag03` is submitted.

## Next step

Submit diag03: every-step re-centering from the ROM centroid, with a
true-centroid control and an output-time-only arm. Development only. Do not
open the final cohort.
