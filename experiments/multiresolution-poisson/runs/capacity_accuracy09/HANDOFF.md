# Poisson capacity job handoff

Status: RUNNING and UNAUDITED. Job 3564896 passed GPU preflight, all focused controls and the complete cluster integration smoke. The real r128 training has started cleanly; its bank phase and initial rank/function/Jacobian checks passed, and its head phase is complete at this handoff snapshot. Do not read the first `CAPACITY ACCURACY COMPLETE` marker as study completion: it belongs to `out/smoke`. Accept only `out/pilot/result.json` complete, terminal successful accounting and `EXIT_CODE=0`, then independent audit/retention.

- Scientific source: `f3e3c21a440a31eb97b06fdb9ab9951662e17935`.
- Slurm job: `3564896` on `pax049`, GPU partition, one-hour allocation.
- Exact remote: `/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/capacity_accuracy09`.
- Stdout/stderr: `/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/capacity_accuracy09/logs/3564896.out` and `.err`.
- Local record: `experiments/multiresolution-poisson/runs/capacity_accuracy09` in the existing approved Poisson worktree.
- Config and protocol: `config-capacity-accuracy.json`, `CAPACITY-ACCURACY-DESIGN.md`.
- Prior accepted comparison: `runs/staged_accuracy08/panel.json`, retention commit `3d4b607`.

The source uses a function-preserving learned-bank widening, r128/k16 vs per-phase optimizer-time-matched r64. It retains original Fourier/trunk layers and appends deterministic affine-complement spatial outputs, with zero appended head/skip coefficients. Training stages are [('bank', 40000), ('head', 30000), ('joint', 20000)]. No difficult-case weighting or latent increase. Original, both r64 head/joint endpoints and both r128 head/joint endpoints are frozen across [64, 256, 1024]. All 42 development cases are already opened; the later 12 informed this capacity choice. No final cohort is touched. Bank target 3% remains a proposed diagnostic target, separately reported from 5% online physical eligibility. Do not silently relax either target.

Every phase, checkpoint, free/optimal training coefficients, parity field/Jacobian, rank/condition and actual optimizer block is retained. The real run expects 3402 paired timing/field invocations with full supplied-source/init/solve/decode cost, repeated arrays, same-job CG/DST and all full-field bank/head diagnostics. The nearest-training-cache input is generic supplied source, with exact smooth weak contraction and no EQ. Failed numerical fits, caps or physical targets must remain visible.

## Collection and acceptance

After this exact job leaves the queue, use:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/multiresolution-poisson/capacity_cluster.py collect capacity_accuracy09
```

The inherited collector verifies source/result/pull checksums, retains a split archive and removes only the exact completed remote attempt. Never overwrite or resubmit this attempt. Preserve a failed attempt the same way before changing code.

Then run the independent CPU checks with bounded BLAS threads:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/tahmid/Dev/.venv/bin/python experiments/multiresolution-poisson/reports/audit_capacity_accuracy.py experiments/multiresolution-poisson/runs/capacity_accuracy09
/home/tahmid/Dev/.venv/bin/python experiments/multiresolution-poisson/reports/verify_split_archive.py experiments/multiresolution-poisson/runs/capacity_accuracy09
/home/tahmid/Dev/.venv/bin/python experiments/multiresolution-poisson/reports/summarize_capacity_accuracy.py experiments/multiresolution-poisson/runs/capacity_accuracy09
```

The audit is prepared, not yet validated against a completed capacity archive. Its training helper passed all six local-smoke phases independently, including widened zero-head proof, complement/scales, exact QR physical metric, frozen subtrees and per-phase elapsed-time matching. The all-field portion adapts the accepted08 auditor to checkpoint-specific ranks and compares the frozen network to every prior08 development field. Review any failure with evidence; do not relax scientific gates to force acceptance. Archive and source hashes remain strict; independently regenerated exp/QR values are compared numerically across platforms. Root can run its second-implementation worst-endpoint audit once fields arrive.

Panel output includes all/existing/later-development mesh groups, every model and FOM, pooled GPU/host repetitions, physical eligibility, separate 3% bank-target status, k/r, head-fit numerical failures and paired cost regression flags. It is gated on an accepted owner audit. Commit the raw/restorable archive, every checkpoint, audits/panel and handoff; append closure to canonical LAB-LOG by absolute path. Root owns main reports and status block. No merge.
