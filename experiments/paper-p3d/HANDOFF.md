# Poisson3D first tranche handoff

The development training/comparison job is submitted. Its exact source, configuration checksum, submission time and remote directory are recorded in `runs/pilot01/SUBMISSION.json`; staged file hashes are in `runs/pilot01/PROVENANCE.json`. The complete tiny local pipeline passed; its independent SciPy audit is `runs/smoke1/audit.json`. Neither smoke results nor an in-flight job are paper evidence.

The cluster preflight printed `jax_backend=gpu` and the run printed f64/highest precision. The initial training job is the only P3D GPU allocation. All final-case data remain unopened. The submitted code automatically saves partial and final bank/head checkpoints, learning curves, reference refinement, representation diagnostics, timed fields and repeated timings before building `summary.json`.

From this worktree, monitor the exact attempt with:

```bash
ssh tufts-login 'squeue -j 3989715; tail -25 /cluster/tufts/paralab/tawal01/paper_p3d_20260920/pilot01/job.out; tail -15 /cluster/tufts/paralab/tawal01/paper_p3d_20260920/pilot01/job.err'
```

When the job has ended, inspect the exit status and check for failed precision, GPU initialization, numerical or resource gates. For a complete run, checksum-collect, independently audit and remove only the exact verified attempt with:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-p3d/cluster/collect.py pilot01 --remove-verified
```

If training succeeds but evaluation fails, preserve the completed checkpoints and failure records before constructing a new unique evaluation attempt. Do not delete an incomplete attempt through the successful-result collection workflow.

The next tranche is diagnostic tuning based on bank-versus-head training and validation errors, followed by FNO3D/U-Net3D on identical forcing/solution pairs and a shared-allocation panel. Add a second mesh, independent training seed and finally a frozen untouched cohort as time allows. DeepONet and Transolver are outstanding. No speed or neural-manifold win is assumed for this linear PDE; free-bank QR, Galerkin, POD and exact DST remain mandatory controls.
