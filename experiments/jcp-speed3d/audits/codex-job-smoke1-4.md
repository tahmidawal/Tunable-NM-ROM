**HOLD** at HEAD `8e22f7db554d`. Separate-job isolation is resolved; resource classification still has blocking honest-path errors.

1. **False FAIL: wrong log stream.** [run.sbatch:11](experiments/jcp-speed3d/runs/smkmesh384/run.sbatch#L11) directs stderr to `.err`, while your acceptance command reads `.out`. An uncaught `MemoryError` or GPU resource exception can produce Slurm `FAILED` with the resource traceback only in `.err`. CPU execution of the acceptance branch confirmed **FAIL**. Combine the streams or inspect the terminating stderr evidence.

2. **False PASS: tail membership still does not establish the terminating cause.** [smoke_accept.py:369](experiments/jcp-speed3d/smoke_accept.py#L369) accepts any resource string within 40 lines, even when followed by an unrelated fatal exception. CPU reproduction confirmed **PASS**. Require the terminal error to establish resource exhaustion.

Other checks passed:

- Both manifests verify: **44 / 38 entries**; **40 / 34 provenance files** match HEAD; commit markers match.
- Both jobs’ shell syntax, GPU preflight, precision settings and separate paralab paths check out.
- Pilot requests H200, 240G, 24 hours; writes `output/smkmesh384/result.json`.
- Main smoke excludes the pilot.
- Pair consumes **two** of three concurrent slots. Live occupancy remains unverified: SSH authentication failed; submit both only with at most one other lane job active.

No repository/git changes, GPU execution or submissions.