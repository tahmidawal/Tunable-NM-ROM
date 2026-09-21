# Handoff — ns3d-grok

Updated 2026-09-21. The sealed truncation job is in the queue.

## State

diag06 (job 4147975, commit `b447d836`) selected truncation tail `1e-6`,
9222 frequencies, on development. Local NumPy verify passed. The Slurm exit
was 1 because `verify_diag.py` was not staged; the computation had already
written `summary.json`. Remote directory deleted. Table:
`results/diag06.md`.

## What is running

Job **4148215** `ns3dgrok_diag07` is pending (Priority) in
`/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag07/`, commit
`f12b8fc2`. It is the only reader of seed 202609211. The setting is frozen:
rank 64, dt=0.01, tail 1e-6. Seed 202609203 stays closed.

## Next step

On exit: confirm `jax_backend=gpu` and `DIAG_EXIT=0`, checksum pull,
NumPy-verify, delete the remote directory. Do not change the tail if the
sealed error exceeds 5%.
