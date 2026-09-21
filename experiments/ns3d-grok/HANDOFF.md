# Handoff — ns3d-grok

Updated 2026-09-21. diag04 selected the setting. diag05 opens the sealed cohort once.

## State

Development diagnosis is finished. Final seed 202609203 has not been read yet.
The frozen setting, chosen as the fastest diag04 row with zero development
cases over 5%, is centered POD rank 64, ROM-centroid re-centering every
startup step, $\Delta t=0.01$. Table: `results/diag04.md`. Commit of that
table is the parent of the sealed job.

## What is running

Job **4142080** `ns3dgrok_diag05` is pending (Priority) in
`/cluster/tufts/paralab/tawal01/ns3d_grok_20260921/diag05/`, commit `9c40c70c`.
It is the only reader of seed 202609203. One job for this lane.

## Next step

Submit diag05. On exit: confirm `jax_backend=gpu`, NumPy-verify, checksum
pull, delete the remote directory, then pick the FOM comparator from the saved
FOM table by the pre-registered rule.
