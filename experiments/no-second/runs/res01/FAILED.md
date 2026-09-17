# res01 — job 3783920 FAILED in the preamble (2026-09-17)

`sacct`: FAILED after 00:00:59 on pax050, exit 1:0. The preflight passed in full
(`jax_backend=gpu`, 395 data files verified, 3/3 checkpoints verified, torch CUDA), the
resolution smoke passed on every rung for all three families, the prolongation check against
`engines.output_field` returned a maximum difference of 0.0, and the matched cohort was rebuilt
(8 cases). The third task then died on its first line of real work:

```
FileNotFoundError: [Errno 2] No such file or directory: 'code/checkpoints.json'
```

Cause: `cluster/stage.py`'s explicit `CODE` list omitted `checkpoints.json`, the frozen-checkpoint
manifest that `worker_resolution.py` passes as `--checkpoints code/checkpoints.json`. The staged
`MANIFEST.sha256` has 30 entries and none is `checkpoints.json`. This is the same class as `pois01`
(missing configs). No operator was evaluated, no number exists to retract, and zero science GPU
time was used, so per LANE-PROTOCOL rule 13 the attempt does not count toward the job cap.

Fix: `checkpoints.json` is now staged, and the stager runs `check_references()`, which resolves
every `code/<path>` literal in every staged `.py`/`.json` against the staged tree and parses every
staged JSON — the omission now fails at staging on this machine instead of at 59 s on the cluster.
Resubmitted unchanged as attempt `res02` (`specs/res02.json` differs from `res01.json` only in
`job_name`). Full machine record: `FAILURE.json`; logs in `logs-failed/`.
