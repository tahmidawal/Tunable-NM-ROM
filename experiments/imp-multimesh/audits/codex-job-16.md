**FIX-FIRST — resource certification does not cover the retry protocol.** The identity, timing, and output wiring otherwise pass this audit.

- **Runtime blocker:** [resources.py:35](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/resources.py:35) budgets one attempt per mesh, while `mmtime` permits two. Its first-pass projection is **155.46 minutes**, including probes. Applying the same estimator to two attempts gives **274.92 minutes**. Moreover, the timing subprocess has a **120-minute timeout**, versus **119.46 minutes projected for first-pass timing alone**. Reconcile retry costs and subprocess limits with the 190-minute allocation before submission. This is a certification gap, not evidence that the normal run will actually take that long.
- **Checkpoint gap:** [mmtime.py:217](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmtime.py:217) saves after whole ours blocks and competitor arms, not after each configuration as A1.7 requires. A timeout can discard completed configurations. Persist each configuration as it finishes.

Checks passed:

- **Inputs:** archive inspection verified all 18 J5 files’ case order and identity-array shapes, all 49 qualified competitor files’ `C`/`it`/`reason` keys, and required lattice/source files. Only small identifiers and array headers were read.
- **Protocol:** A1 → initial sentinel → competitor arms with sentinels → A2; correct repetition counts, burn-in, timed-result identity, drift band, retry retention, and combined validity gate.
- **Memory:** code releases the full ours bank before competitors; prefix caching retains one prefix copy. The stated H200 resident-array budget is plausible, though this read-only audit does not measure transient peaks.
- **Compilation/analysis:** fresh JIT objects, backend-event recording, disabled persistent cache, 12 separate compile processes, and expected `time.json`/compile output schemas are wired correctly. A7.1 supersedes the older charging rules.
- **Provenance:** staged manifest passes; config hash matches the certificate; selected configurations exactly match pulled target JSONs.

Small efficiency improvement: load J5 `W`, `states`, `it`, and `reason` once per configuration; current indexing repeatedly decompresses entire NPZ members.

No files changed, jobs submitted, or lab-log/main writes.