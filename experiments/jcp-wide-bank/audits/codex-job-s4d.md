- **PASS — Resources/runtime:** `gpu:h100:1`, `240G`, GPU partition, four hours; correct absolute cluster venv, x64, highest precision, memory fraction `0.90`, manifest verification and GPU-only preflight.
- **PASS — Staging/provenance:** All 12 payload files match HEAD `3c695d0ad` byte-for-byte, including `w3d.py`, W1024 bank and both rule archives. Complete manifest verifies.
- **PASS — A6/A7 fixes:** Present; `w3d.py` is unchanged from the code approved in audit `codex-code-w3d-5`.
- **PASS — Config unchanged:** Matches HEAD, original audited commit `243ef4806`, and staged s4. Uses 129 nodes, R′=1024, M=4097; no deployment tensor; gl48/check shares storage.
- **PASS — Memory smoke feasibility, not guaranteed fit:** Resident mesh bank **16.780 GB** + rule blocks **32.925 GB** + G65 **2.048 GB** = **51.754 GB**, leaving approximately **20.25 GB** against the nominal 72 GB budget. Gate tables are built before the full rule bank and explicitly released. Construction duplicates, compiled executables and workspaces make peak uncertain.
- **PASS — Jacobian/target scaling:** Reached-state Jacobian is only **33.56 MB**. Rho targets use **64-state batches** and **65,536-point chunks**: approximately **3.22 GB** for each chunk’s B/D/P arrays, plus intermediates. Largest resident-rule P is **5.76 GB**; construction temporaries remain a credible OOM risk. A clean OOM is acceptable for this diagnostic smoke.
- **PASS — Submission preparation:** Shell/Python syntax checks pass; local `logs/` exists. No `jw_` job is currently queued. Remote s4d is not uploaded yet; the existing submission wrapper stages it and its log directory before sbatch.

Read-only audit completed; nothing modified or submitted.

SUBMIT: YES