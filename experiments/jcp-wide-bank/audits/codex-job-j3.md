- PASS — Resources: `gpu` partition, one H200, 420G RAM, and 10-hour limit match registration.
- PASS — Environment: absolute cluster venv, `JAX_ENABLE_X64=true`, highest precision, and GPU preflight exiting 42 on CPU are present.
- PASS — Wrapper: `/usr/bin/time -v` records MaxRSS, the sidecar samples training-child RSS every 30 seconds, and exit-code tests preserved 0/7/42 under strict shell settings.
- PASS — Provenance: every manifest checksum matches, and all five staged payloads equal HEAD `005b025abbda92665de356742068ae8c16bfc57e`.
- PASS — Paths: training script, both vendored dependencies, and the verified 512-column comparison bank resolve correctly from `TASK_ROOT`; staged `logs/` exists.
- PASS — Registration: config differences are exactly the eight registered changes, and training code contains only the two authorized functional changes.
- PASS — B1: finite checkpoint checks and unchanged worst-validation selection populate `bank.selected_step` and `bank_log`.
- PASS — B2: `ordering.inverse_check` and `bank.condition_65` support the inverse and numerical-rank gates.
- PASS — B3: same-job validation fields produce new-bank and comparison-bank full-grid floors at 129 nodes for rank 512.
- PASS — B4: `floors_full` includes 129-node results for 512/768/1024, supporting strict decrease and material-gain checks.
- PASS — Diagnostics: dropped POD tail is recorded per mesh and whitening appears in `bank_log`.
- PASS — Step-rate evidence: checkpoint seconds at 1000 and 2000 support the registered seven-hour projection gate, which requires external monitoring.
- PASS — Smoke execution: the supplied log reports GPU, x64, highest precision, finite training, ordering, both floor paths, and `TRAINING COMPLETE`.

Read-only audit completed; no files modified or jobs submitted.

SUBMIT: YES
---
**Post-submission failure (lane agent):** job 5012802 failed after 12 s with exit 127: `/usr/bin/time` exists on the
login node but not on the compute node pax010; nothing ran. Missed by the audit (it checked the login node's view).
Fix: the wrapper uses `/usr/bin/time -v` only if present; host memory is still recorded by the 30 s RSS sidecar and by
`sacct` MaxRSS. Restaged as `j3b` (new directory); `j3` remote directory removed.
