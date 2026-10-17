# Lane brief: jcp-speed3d (RESUME)

Read `briefs/COMMON-BRIEF.md` first; it binds you. This brief adds your lane specifics.

## Situation
- Worktree `worktrees/2026-10-14-jcp-speed3d`, branch `exp/2026-10-14-jcp-speed3d`, lane dir
  `experiments/jcp-speed3d`, namespace `/cluster/tufts/paralab/tawal01/jcp-speed3d`. The previous lane
  agent was **stopped mid-lane**. Last lane commit `f98843cb1` ("f1 (5301274) collected; g1 submitted 5301580").
- Production jobs g1 5301580, g2 5301264, g3 5301270 all COMPLETED on 2026-10-15/16 but were **never
  collected**; their outputs are still in `/cluster/tufts/paralab/tawal01/jcp-speed3d/{g1,g2,g3}`
  (≈1.4–1.6 GB each). f1 5301274 was collected.
- You are not starting a new design. You **resume** the existing lane under its own `DESIGN.md`
  (amendments A1–A6), `runs/JOB-LEDGER.json`, `audits/` (codex-select-3 READY was the last gate) and
  the last commits. Read those first, plus `git log` since `9f8093d64`.

## Your sequence
1. Collect g1–g3 (check `sacct` states and `jax_backend=gpu` in each log; checksum remote vs local;
   keep large arrays out of git, SHA256 in `LARGE_FILES.sha256`; check `df -h /home` first). Update the
   ledger and the GPU-hour ledger (`runs/GPU-LEDGER.json`, measured from `sacct`, including all past jobs).
2. Score offline **on CPU** against E0 (`JAX_PLATFORMS=cpu`).
3. Run selection and matching (`select_s3.py`, `fom_match.py`) exactly as audited.
4. Codex results audit (recompute from raw outputs; every control fired). Fix WRONG, re-audit.
5. Then p1 and m1 as already designed and projected (A6 included), each with its Codex job audit
   before submission. Lane concurrent-job cap: **3**. Respect the campaign-wide cap of 8.
6. Results audit for p1/m1, then `make_report.py` → `report.md`, report audit, hand back.

## Constraints from the campaign plan
- Your outputs will be consumed by other lanes per `PLAN-v3-amendments.md` V3-4 (reuse map) and V4-6:
  a finite FOM ladder gives an **upper bound on optimal FOM cost, hence an upper bound on the claimed
  ROM speed-up** (use this wording); matched labels do **not** transfer across meshes; m1 is the
  **mesh-free / scoring-lattice** contract and must be labelled so (it is not the full-field query).
- Make your report state, per FOM family (upwind/BE, central-CN, spectral) and against their union, the
  3D H-comp cells (V3-5): which (setting, family) cells reach ≥ 2× and which do not. The early unaudited
  f1 observation "spectral FOM clearly beats the ROM at 64³" must be confirmed or retracted from the
  audited numbers. If K2 (governing plan) fires, say so prominently; that is a valid result.
- Supply, in a machine-readable `runs/EXPORT.json`, the per-arm Q1 certificate outcomes (with the
  Q1 numbers), the 3D cost-vs-mesh numbers from m1, and the matched speed-ups, so the coordinator and
  other lanes can consume them without re-deriving.
- Do not extend scope (no new arms beyond the design). No new sealed/OOD access.
- Heartbeat at `experiments/jcp-speed3d/HEARTBEAT.json`.
