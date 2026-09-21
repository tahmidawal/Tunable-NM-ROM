# Handoff — ns3d-grok

Updated 2026-09-21. diag06 is the development coefficient-space test.

## State

Seed 202609203 stays closed. Seed 202609211 is reserved and has not been
generated. diag06 compares the grid center tracker with its coefficient-space
form on development seed 202609202. The sealed draw happens only if that job's
`summary.json` selects a method.

## What is running

Job **4147975** `ns3dgrok_diag06` is pending (Priority) in
`/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag06/`, commit
`b447d836`. One job for this lane.

## Next step

On exit: confirm `jax_backend=gpu`, NumPy-verify, checksum pull, delete the
remote directory. Seal 202609211 only for the selected method. If the
selection is null, do not open that seed.
