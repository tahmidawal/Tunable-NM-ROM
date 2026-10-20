**SUBMIT — technical verdict, conditional on the coordinator releasing this second smoke.** No blocking defect found in the staged job. Completion within the hour remains an accepted, fail-closed risk.

| Item | Verdict | Evidence |
|---|---|---|
| Resource requests | **CORRECT** | `run.sbatch` requests `gpu`, one H200, `200G`, eight CPUs and `01:00:00`: a scheduled allocation of at most **1 GPU-h**. |
| Environment | **CORRECT** | Absolute cluster venv; `JAX_ENABLE_X64=true`; precision `highest`; GPU preflight exits **42** on CPU fallback. Driver also asserts GPU and highest precision. |
| Output/cache isolation | **CORRECT** | Logs, outputs, temporary files, XDG/CUDA/JAX caches and output checksums resolve beneath `/cluster/tufts/paralab/tawal01/jcp-hrb/smk2`. E0 is an external read-only input. |
| Manifest and HEAD | **CORRECT**, with documented exception | All **49** manifest entries verify. All **45 non-config provenance entries** match committed HEAD `b4371eda55c97ae9be8ea3f1cd1787eeea17877c` byte-for-byte. Config differs only by the added matching `commit` field, as A13.5 specifies. Its staged hash and size match provenance. |
| Frozen inventory | **CORRECT as amended** | Both config cells match `smoke_accept.py::FROZEN`; ranks also match A5.4. Meshes are **1024² / 64³**, training counts **192 / 342**, maximal ranks **512 / 1024**, maximal advection dimensions **2048 / 4096**. ECSW uses A15’s replacement support knobs `[2,4]`, running at `4r`, rather than obsolete A5.4 tolerance knobs. |
| E0 pin | **CORRECT** | `JCPREF_STORE` is set; `JCPREF_META` is unset. Staged `e0/store/MANIFEST.sha256` and `STATUS.json` hashes exactly match `score.py`’s pins. Loader metadata resolves to that staged directory. |
| Submission safeguards | **CORRECT**, with concurrency qualification | `submit.sh` checks the queue before/after, refuses an existing attempt directory, verifies hashes, and refuses submission at lane count ≥2 or account count ≥8. Its lock is **lane-local**: simultaneous submissions by other lanes still require coordinator serialization to guarantee the account cap. |
| Deadline/checksumming | **NEEDS-RESTATEMENT** | `wall_seconds=3300` is a phase-entry guard, not an interruptible deadline. Shell timeout sends TERM after **3360 seconds**, then KILL after another **30**. This reserves roughly **210–240 seconds minus setup time** for checksumming. A13.5’s “always checksummed” is too strong: neither setup duration nor checksum duration is bounded. |
| Fit within 55 minutes | **NEEDS-RESTATEMENT: unproven** | smk1’s driver took **3139.1 seconds**, despite failed 2D advection/Q-DEIM work, failed 3D maximal ECSW work and a skipped 3D grid. Its 3D smoke-rank NNLS alone took **1174 seconds**. smk2 changes that algorithm substantially; full-QR paths continue to **16r**, with possible retries. NNLS, maximal-shape pilots and consequently later 3D grid/control/timing phases are particularly at risk. There is no defensible completion certificate yet. |
| Phase-I evidence and acceptance | **CORRECT** | Required phases, raw trajectories/fields, Q1 evidence, derivative controls, undersampling controls, gradient/scoring controls and timing attempts feed offline recomputation. Staged CPU gates pass and their source hashes match. Missing/skipped/error phases fail acceptance; unavailable required controls remain unfulfilled. `ALL-DONE` alone cannot establish success. |

Two scope qualifications matter:

- “Every control at real size” means the **amended registered inventory**. Full-basis/FOM reproduction remains a small CPU gate; the maximal Q-DEIM pilot may use uncertified `pmax` under A14; ours’ adapter execution is not phase-II G-ours.
- smk1 already consumed the brief’s single permitted smoke. A15.6 explicitly records that smk2 needs coordinator release. This verdict assesses technical readiness and does not grant that release.

Unmanifested Python 3.12 bytecode caches are present from staging; they are outside the provenance claim and incompatible with the specified cluster Python 3.13.

No files changed, no scratch created, no jobs submitted and no GPU work run.