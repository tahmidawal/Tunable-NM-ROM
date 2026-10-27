# Burgers 3D identical-sample reconstruction screen

User-approved reconstruction-only screen, 2026-10-27. Implementation is in preparation; no numerical results exist. Dynamics and sealed/OOD cohorts are excluded.

This lane uses the existing frozen W1024 spatial bank and fits coefficients from the identical supplied nodal samples consumed by interpolation and continuous POD. The input sources are restricted E0 fields and independently evolved coarse FOM fields, both at the same physical times. Initial setup and all per-query fitting/decoding costs are accounted separately.

- Branch: `exp/2026-10-27-burgers-recon3d`, fork `93f399faa`.
- Cluster namespace: `/cluster/tufts/paralab/tawal01/b3d-recon-20261027/`.
- Allocation ceiling: four GPU-hours, including failed attempts. Numerical release requires parent review of the frozen design, code and exact job.
- Proposed coarse intervals 16/32, fine scoring intervals 64; bank prefixes 256/512/1024; exposed dev4/val64 only.
- No new neural training. The original multimesh bank has extra fine-training information.

The central design in the coordinator lane governs this document. Exact input and source hashes will accompany every result. No files outside this worktree are modified except required canonical lab-log entries.

## Glossary
- Interval count: one fewer than total nodes along each axis.
- Bank prefix: a selected number of the existing ordered spatial functions.
- E0: the archived, conditionally qualified second-order numerical reference.
- POD: classical basis obtained from the training snapshots.
- Coarse FOM: numerical PDE solution used only to supply samples to reconstruction.
