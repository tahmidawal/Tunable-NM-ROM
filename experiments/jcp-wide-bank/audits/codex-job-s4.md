- **PASS — Sbatch:** Requests one H200, 240G host RAM and four hours on `gpu`, with f64, highest precision, 0.90 device-memory fraction, manifest verification and mandatory GPU preflight.
- **PASS — Staged provenance:** All 12 payload files match HEAD `243ef4806`, including `w3d.py`, both banks and training records, and `rules_extra.npz`; the complete manifest verifies.
- **PASS — Pins:** Bank, both rule archives and offmesh SHA256 values match the staged bytes.
- **PASS — Bank metadata:** Rotation is 1024×1024, with finite spread entries `'1024'=0.11765954298738913` and `'32'=0.11667775120116541`.
- **PASS — Configuration:** Uses 129 nodes, R′=1024, κ=4 giving **M=4097**, every requested arm, shared gl48/check storage, no deployment tensor, validation cases 0–3, certification seed 923811 ×4 and two timing cases.
- **PASS — Reference:** The cluster reference and accepted `.done` metadata match the configured seed, count, resolution, tolerances and independently verified checksum.
- **PASS — Static memory feasibility:** Mesh bank **16.780 GB** + resident rules **32.925 GB** + G65 **2.048 GB** total **51.754 GB**, leaving about **75.1 GB** within the nominal 126.9 GB allocator budget for construction temporaries, gate tables and workspaces, although actual peak remains smoke-dependent.
- **PASS — Static crash trace:** Syntax, imports, paths within the staged payload, rule shapes, spread lookups, shell completion, check aliasing and final-panel references reveal no deterministic crash; numerical assertions remain untested.
- **FAIL — Cluster staging:** `/cluster/tufts/paralab/tawal01/jcpwide/s4` and its `logs` directory do not exist, so the batch script’s required execution and log destinations are not ready.
- **FAIL — Lane sequencing:** `jw_s2` job **5018586** remains pending, and s4 has no dependency enforcing the lane’s one-concurrent-job limit.

SUBMIT: NO
---
**Disposition (lane agent):** remote directory creation is done by `cluster/submit.sh` (as for s1/j1/j3b); the lane cap
was raised to 2 on H200s by the coordinator (2026-10-08, evening) so s4 overlaps the queued s2. Submitted with
`LANE_CAP=2`.

**Resubmission (lane agent):** the first attempt stopped at gate G1 (s2, job 5018586) / was cancelled before starting
(s4, job 5019404). Restaged unchanged except the A6 G1 fix, which was code-audited (audits/codex-code-w3d-3.md,
OK-TO-RUN YES); configs, staged inputs and sbatch are otherwise identical to the audited attempt. New attempts: s2b / s4b.

**Second resubmission (lane agent):** s4b (5019667) was cancelled before start because it ran the pre-A7 eligibility
code. Restaged unchanged except the A7 code, which passed audits/codex-code-w3d-4.md + -5.md (OK-TO-RUN YES). New
attempt: s4c.
