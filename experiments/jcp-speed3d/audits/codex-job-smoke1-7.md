**SUBMIT** — round-6 HOLD resolved at HEAD `9f8093d64ff6`, under DESIGN A5.

- L=384 has a separate H200 Slurm job, 240G memory allocation and directory. Main smoke excludes it.
- Pilot output resolves to `output/smkmesh384/result.json`.
- Both manifests verify: **44/38 entries**; **40/34 provenance files** match HEAD; commit markers match.
- Both jobs pass shell syntax checks; commands, GPU preflight, f64/highest precision and paralab paths check out.
- CPU checks confirm resource termination is accepted; allocation warnings followed by native crashes, unrelated exceptions, timeout or cancellation fail.

The pair uses **2 of 3 concurrent slots**. SSH closed the connection, so live occupancy remains unverified: recheck before submission and launch both only with at most one other lane job active.

No repository/git changes, GPU execution or submissions.