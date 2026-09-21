# Handoff — ns3d-grok

Updated 2026-09-21 after diag03 passed on development. diag04 is the fused timing.

## State

diag03 (job 4141126, commit `70f1573a`, A100-40GB, `jax_backend=gpu`) is
pulled, NumPy-verified, and deleted. Table: `results/diag03.md`. Every-step
ROM-centroid tracking is under 5% on all 16 development cases at ranks 64 and
128, for $\Delta t=0.004$ and $0.01$. Output-time re-centering is not. Final
seed 202609203 is closed.

## What is running

Job **4141875** `ns3dgrok_diag04` is pending (Priority) in
`/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag04/`, commit `9b8adc74`.
One job for this lane.
