# Poisson3D complete-operator comparison handoff

The longer NM-ROM/FNO/U-Net development panel is complete, checksum collected and independently audited. Its immutable original output is `runs/tuned02/archive/out`, local independent audit is `runs/tuned02/audit-local.json`, and source-generated diagnostics are `runs/tuned02/diagnostics.json`. The exact completed cluster directory has been removed. The root coordinator owns the generated main report and canonical lab log.

The active additional-operator job is **3995104**, attempt `extra03`, source `a50ce0977373977688d73f82700108a6138b433b`, config SHA256 `fdbe781595dddce13e4e6c36317c486b0d51a68ae79c58ee3227dee370a48c87`, submitted `2026-09-20T05:29:52.432424+00:00`. It runs on an A100 in `/cluster/tufts/paralab/tawal01/paper_p3d_20260920/extra03` with GPU/f64/highest preflight verified. Its source, reused checkpoint and configuration manifest is `runs/extra03/PROVENANCE.json`. No other P3D GPU job exists; final data remain unopened.

The audited tuned model now meets the prescribed same-grid development target on its latent-16 dense head and improves further through correction enrichment. Its bank and unrestricted linear endpoint are substantially stronger than the first pilot. Dimension-matched small POD controls are worse than the head, but the strongest POD and direct DST controls are more accurate and cheaper than NM-ROM. Both quadrature certificates still fail their unchanged threshold. Those sampled rows remain diagnostic and cannot support deployment claims.

The native-mesh FNO and U-Net are accurate. Direct frozen evaluation at the finer mesh deteriorates sharply, while native-mesh prediction followed by boundary-aware interpolation remains accurate. These are different methods, explicitly preserved as separate positive and negative rows. The new job reuses both operator checkpoints and all NM-ROM checkpoints, regenerates the identical data from seed, checks the training-array and membership hashes, adds DeepONet and actual physics-attention Transolver, and remeasures every model/control in that one allocation. All frozen parameters are placed on the GPU once before timing. The earlier NumPy-selected head parameters could incur avoidable repeated transfers; archived timings remain valid for that implementation and are never pooled with this corrected panel.

Meaningful bounded local checks passed for both added operator families: actual training and validation selection, saved checkpoint replay, forcing-adapter parity, and independent SciPy verification of zero-boundary interpolation. The whole frozen NM-ROM replay and independent audit also pass; retained evidence lives in `runs/{deeponet_smoke1,transolver_smoke1,reuse_smoke1}`. Imported source and license hashes are in `operators/EXTRA_IMPORTS.json`; common upstream source is H commit `5933b3706c119e8ffbcfdf98db3940464af7b4fc`.

Monitor with:

```bash
ssh tufts-login 'squeue -u tawal01; tail -25 /cluster/tufts/paralab/tawal01/paper_p3d_20260920/extra03/job.out; tail -15 /cluster/tufts/paralab/tawal01/paper_p3d_20260920/extra03/job.err'
```

When the active job has finished successfully, collect from this worktree:

```bash
OPENBLAS_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python experiments/paper-p3d/cluster/collect.py extra03 --remove-verified
```

The collector refuses active account-queue membership, verifies original source/output manifests, recomputes all saved-field errors with independent SciPy DST references and checks every summary/count/counter, then rechecks the original manifest before deleting only the literal completed attempt. The local audit never overwrites the original remotely checksummed evidence. Failed or interrupted jobs require separate retained recovery.

For a fresh checkout, the sole large tuned checkpoint is stored in verified chunks. Restore it with:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-p3d/runs/tuned02/large-artifacts/restore.py
```

The existing full file remains on disk, byte-identical to the original output manifest; only that redundant large file is ignored, while all chunks and their manifest are tracked.

Remaining campaign priorities after the additional operators finish: inspect the actual four-family errors and validation curves; improve any materially weak learned baselines under documented development-only changes; consider a short bank/head continuation because their last selected checkpoints were still improving; run independent initialization seeds for frozen finalists; coordinate selection with the root before opening the untouched final cohort. The finer scaling panel is deferred until shared device matrices and a small selected rung set avoid unnecessary memory duplication. A failed quadrature certificate, negative operator transfer, or stronger classical control is retained, never relabelled as a success. No merge or push is authorized. Root owns main reports and the canonical log; this worker writes only this approved P3D tree and namespace.
