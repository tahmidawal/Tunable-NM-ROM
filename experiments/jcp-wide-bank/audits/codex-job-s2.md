- PASS — `run.sbatch` requests one H200 on `gpu`, uses the correct cluster venv, enables x64/highest precision, and exits 42 on failed GPU preflight.
- PASS — All staged source, config, rule and model files match HEAD `6a8179b467`; every manifest checksum passes.
- PASS — The complete import chain (`offmesh.py`, `common.py`, `b3d_common.py`, `tables.py`) is staged, and required external packages exist in the cluster venv.
- PASS — Configuration matches the requested smoke: 65 nodes, M2, R′=512, κ=4, all J2 arms, validation 0–3, certification seed 923811 × 4, and two timing cases.
- PASS — Both rule paths resolve from the staged root, pinned hashes match, and all required rule keys and array shapes are valid.
- PASS — The cluster reference and `.done` exist, checksum and metadata match, and cases 0–3 contain finite f64 arrays shaped `(6, 250047)`.
- PASS — Static tracing found no deterministic name, key or shape crash through `main()`; the GB10 non-finite Gram remains an unresolved runtime risk.
- PASS — The remote `s2` directory is absent as expected before `submit.sh` performs staging.
- FAIL — Submission is currently blocked by lane capacity: `jw_j1` (5012763) and `jw_j3b` (5012847) are running, filling even the documented cap-two override.

SUBMIT: NO
---
**Disposition (lane agent):** the only FAIL was lane capacity; submitted once J1 (5012763) had finished and J3b
(5012847) had been collected, with the cap back at 1.

**Resubmission (lane agent):** the first attempt stopped at gate G1 (s2, job 5018586) / was cancelled before starting
(s4, job 5019404). Restaged unchanged except the A6 G1 fix, which was code-audited (audits/codex-code-w3d-3.md,
OK-TO-RUN YES); configs, staged inputs and sbatch are otherwise identical to the audited attempt. New attempts: s2b / s4b.
