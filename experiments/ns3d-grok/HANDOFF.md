# Handoff — ns3d-grok

Updated 2026-09-21. diag04 selected the setting. diag05 opens the sealed cohort once.

## State

Development diagnosis is finished. Final seed 202609203 has not been read yet.
The frozen setting, chosen as the fastest diag04 row with zero development
cases over 5%, is centered POD rank 64, ROM-centroid re-centering every
startup step, $\Delta t=0.01$. Table: `results/diag04.md`. Commit of that
table is the parent of the sealed job.

## What is running

Nothing until `ns3dgrok_diag05` is submitted. That job is the only reader of
seed 202609203. Do not submit a second sealed job and do not change the rank
or the time step after seeing its errors.

## Next step

Submit diag05. On exit: confirm `jax_backend=gpu`, NumPy-verify, checksum
pull, delete the remote directory, then pick the FOM comparator from the saved
FOM table by the pre-registered rule.
