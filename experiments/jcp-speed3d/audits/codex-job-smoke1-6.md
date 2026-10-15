**HOLD** at HEAD `86c359017b91`. Isolation is resolved; one blocking false-PASS remains.

[smoke_accept.py:371](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke_accept.py:371) selects the last line containing `error` or `RESOURCE_EXHAUSTED`, rather than establishing the terminal failure. CPU execution of the actual acceptance branch confirms: an allocation warning followed by **`Segmentation fault (core dumped)`**, with sacct `FAILED`, produces **PASS**. An honest native crash can therefore be misclassified as resource infeasibility. Require terminal resource evidence; subsequent non-resource termination must fail.

Other checks passed:

- Manifests: **44/38 entries** verified; **40/34 provenance files** match HEAD; commit markers match.
- Both jobs pass shell syntax checks; GPU preflight, f64/highest precision and separate paralab paths are configured.
- Pilot: separate H200 job, **240G**, 24 hours; output targets `output/smkmesh384/result.json`.
- Main smoke excludes the pilot.
- Incomplete-result timeout/cancellation checks now correctly FAIL.
- Pair occupies **2 of 3** slots. Live occupancy remains unverified because SSH closed the connection; submission requires at most one other lane job active.

No repository/git changes, GPU execution, or submissions.