# Sparse-solver collection and verification

The AmgX Burgers trajectories and the two ROM timing allocations have been collected. The generated [trajectory tables](../../RESULTS-burgers-stepper.md) and [ROM tables](../../RESULTS-ours-same-gpu.md) include the numerical results and the failed Poisson timing check.

The AmgX driver measured every trajectory repetition. It evaluated the field error on the first timed repetition of each case. The threshold table therefore includes both the registered all-repeat median and a median from the exact calls that produced the errors. The AmgX cohort is the first subset of the full Burgers test cohort. For the Burgers ROM rerun, every timed invocation passed its output witness against the scored reference pass. The driver checked sampled-field hashes throughout and full-field hashes on a subset. Poisson saved an error and field witness in each ROM timing record. The sparse solvers and ROMs ran in separate allocations.

## Reproduction

From the sparse-solvers experiment directory, use the project Python environment.

```bash
/home/tahmid/Dev/.venv/bin/python summarize_ours.py > RESULTS-ours-same-gpu.md
/home/tahmid/Dev/.venv/bin/python summarize_step.py results/step/h200 > RESULTS-burgers-stepper.md
/home/tahmid/Dev/.venv/bin/python ledger/closeout-2026-11-15/verify.py
```

The verifier compares the compact JSONs with the SHA-256 digests recorded before collection, recalculates AmgX timing and error statistics from cases, checks ROM errors and time medians, and verifies staged source entries against the original Git blobs. The captured scheduler accounting, manifests and run logs preserve job, source, host and precision records. Job scripts regenerated the inputs from the recorded cohorts. The AmgX launcher removed transient reference arrays when each mesh finished, as specified in its script. Reference metadata and regeneration code remain in the archive.

## Bulk retention

The complete surviving job directories were archived under `/cluster/tufts/paralab/tawal01/sparse-solvers-20261113/retained/2026-11-15-three-job-closeout/`. [Archive digests](archives/archives.sha256) and per-file SHA-256 lists accompany this record. Each archive was compared byte-for-byte against the corresponding cluster job directory and then transferred to RAM-backed staging on the local machine. The independent local verifier read every archive member and reproduced its SHA-256. The full archives are retained on the paralab project share, and the compact results, logs, verification record and hashes are retained in Git. To repeat the full recovery check, copy the archive files to a volume with sufficient free space and supply its path to the verifier with `--archives`. The remote share is the durable home of the bulk arrays.

## Glossary

A trajectory is the sequence of fields returned by one PDE solve. A timed repetition measures another execution at the same setting and case. A field witness is a checksum of a result or a sampled part of it. The neighbour check measures the change in timing after different preceding operations. A SHA-256 digest identifies the exact bytes of a file.
