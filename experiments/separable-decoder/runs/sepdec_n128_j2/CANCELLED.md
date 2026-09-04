# j2 (Slurm 2827226) — CANCELLED by user decision at 2h34m55s

The user redirected the study to a focused N=256 push mid-run. Only cell 1 of 5
completed: Burgers K=16 R=128 STEPS=200k MAX_SNAPS=16384 (JSON + checkpoint
here, `complete: true` inside the JSON). Cell 2 (Burgers K=32 R=256 200k) was
killed mid-training; cells 3-5 (the three Poisson 200k cells, incl. the m=32K
EQ probe) never started. `RESULTS.sha256` was generated at PULL time (the
in-job one is only written at job end, which never ran); remote-side file
hashes were captured by rsync from the still-intact job dir before deletion.
