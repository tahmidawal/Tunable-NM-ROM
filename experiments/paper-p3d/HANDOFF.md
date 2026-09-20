# Poisson3D tuned comparison handoff

The first pilot is complete, checksum collected, independently audited and retained in commit `68fd0cca`; its exact cluster directory was removed. It is an accepted diagnostic, with unsuccessful model-accuracy and quadrature outcomes retained. The generated cross-PDE paper table is maintained by the root coordinator from its archived JSONs. The local audit checked 416 fields, 16 SciPy DST references and 26 summary rows; see `runs/pilot01/audit-local.json` and the unchanged remote checksums under `runs/pilot01/archive/`.

The next active job is **3990494**, attempt `tuned02`, scientific source `68fd0cca50c44b8c866986d1fcbf44d9ee8b4c1b`, submitted `2026-09-20T04:49:34.330974+00:00`. Exact source/configuration checksums and cluster namespace are in `runs/tuned02/SUBMISSION.json` and `PROVENANCE.json`. It requests one A100 for two hours, with its own directory and GPU/f64/highest gates. There is no second P3D GPU job. Final data remain unopened.

This job trains a fresh learned rank-128 bank with materially increased optimization budget and validation checkpoint selection, then latent-8/16 heads. It trains the frozen H-lane FNO3D and U-Net3D implementation on the same 512 training and 16 development forcing/solution members. It evaluates the full correction ladder and classical controls together with both operators at 32/64 intervals, preserving full output costs and repetitions. At 64, direct operator transfer and native-32 prediction followed by boundary-aware interpolation are separate methods. Protocol amendment A1 and `config.json` give the exact prospective budgets, criteria and quadrature repair; no numerical success is presumed.

Meaningful local checks passed: the actual validation-checkpoint NM-ROM pipeline and independent output/reference audit in `runs/smoke2`, and two-step actual training, validation selection, checkpoint replay, task-adapter parity and independently checked boundary-aware nodal interpolation for both operator families in `runs/operator_smoke1/audit.json`. Every smoke was a bounded local jaxrun under one minute. Imported primitive code and SHA256 provenance live in `operators/IMPORTS.json` (H source `84d302a7`).

Monitor with:

```bash
ssh tufts-login 'squeue -u tawal01; tail -30 /cluster/tufts/paralab/tawal01/paper_p3d_20260920/tuned02/job.out; tail -15 /cluster/tufts/paralab/tawal01/paper_p3d_20260920/tuned02/job.err'
```

The run saves partial optimizer state, selected checkpoints, bank/head validation curves and operator curves before evaluation. Selected bank/head checkpoints can differ from the latest iteration. Inspect validation accuracy, training-versus-bank floors and continuation slopes before deciding further capacity/optimization changes; all comparisons remain development evidence. No convergence is asserted from a finite budget. The failed pilot quadrature rows stay archived. Tuned sampled rows require the unchanged held-out forcing-moment certificate before supporting a deployment claim.

When this job ends successfully, from this worktree run:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-p3d/cluster/collect.py tuned02 --remove-verified
```

The collector refuses live account-queue membership, verifies source/output checksums and basic job logs, writes an independent audit to `audit-local.json` outside the checksummed output tree, then rechecks the original manifest before deleting only the literal verified remote attempt. An incomplete job must be preserved through a separate recovery workflow. Do not overwrite remotely checksummed `audit.json` with a local recomputation.

After this development comparison, remaining campaign work is validation-based model tuning if still weak, a second initialization seed when feasible, and a frozen selection followed by one untouched final cohort. DeepONet/Transolver are explicit backlog. Neural superiority over the free linear bank, POD or DST is not assumed. No merge or push is authorized. Root owns the canonical LAB-LOG and main reports; this worker writes only the P3D tree/namespace.
