# Lane brief: jcp-fom2d (priority 2: 2D C1 FOM ladder and C2 crossover; 2D H-comp)

Read `briefs/COMMON-BRIEF.md` first; it binds you. Lane specifics below.

## Where
- New worktree `worktrees/2026-10-18-jcp-fom2d`, branch `exp/2026-10-18-jcp-fom2d`, created by the
  coordinator from `21c175a1b` (sparse; `sync_github.sh` configured). Lane dir `experiments/jcp-fom2d`.
  Namespace `/cluster/tufts/paralab/tawal01/jcp-fom2d/`.
- Concurrent-job cap **3**. GPU-hour cap **30** including contingency, with reservation.

## Build on (read report.md + DESIGN.md)
- `worktrees/2026-10-14-jcp-speed3d` — the 3D version of exactly this work: `fomcore.py`, `fomrun.py`,
  `fom_match.py`, `psjax.py`/`pscore.py`, its DESIGN amendments on matching (A2.3 scoring-lattice
  distance; A6 mesh-specific matching; finite-ladder upper-bound wording, `DESIGN.md:373–379`). Port the
  design to 2D; copy code with provenance.
- `worktrees/2026-10-06-jcp-references` (E0 `refcore.py` central+CN-R, `pscore.py` sine pseudo-spectral;
  `refstore.py`), `worktrees/2026-10-13-jcp-combined/fom2.py` (2D FOM used for the 13.6× H5 result),
  `worktrees/2026-10-08-jcp-time2` (fom1k fair second-order FOM), `worktrees/2026-10-17-jcp-p1-scaling`
  (timing harness, frozen ROM configurations).

## Items owned (plan V2 C1, C2; V3-5 H-comp; V4-6)
- **C1:** 2D FOM ladder over three families — upwind+BE, central+CN-R, sine pseudo-spectral — × Δt ×
  solver tolerances × **FOM mesh** (the FOM mesh varies freely, coarser than the ROM's mesh allowed). The
  comparator per family is selected on **validation** with pre-declared median **and** worst criteria
  against E0 ST with ε_ref ties, then confirmed by paired timing. Positive gate: reproduce a trusted FOM
  (E0 reference agreement at the reference resolution). Negative controls: a non-converged/too-loose
  comparator rejected; a falsely matched accuracy pair rejected.
- **C2:** crossover for each family and their union: (a) same-mesh crossover; (b) best accuracy-matched
  crossover (FOM on any mesh) against the ROM front `runs/FRONT-2D.json` from jcp-qualify (the
  coordinator will tell you its commit; until then build and qualify the ladder and its accuracy table).
  Paired ROM–FOM timing in one job on one GPU per mesh (A–B–A). Matched labels never transfer across
  meshes. Wording: a finite ladder gives an **upper bound on optimal FOM cost, hence an upper bound on
  the claimed ROM speed-up**.
- **H-comp (2D) decision rule (V3-5):** per (setting, family) at the largest fixed mesh (4096²) and also
  1024²: "≥ 2× faster at matched accuracy" stated only for cells that pass; union reported. **K2**
  (governing plan, `reports/2026-10-06-jcp-offmesh-paper-plan.md`): evaluate as written for every linear
  setting at the largest mesh; if it fires, say so prominently — a valid result.
- Record both median-matched and worst-matched comparisons. Report cost per FOM at each mesh.
- **Defer order** if the cap binds: upwind family at 4096² → tolerance sweep breadth → 256² crossover.
  Never dropped: spectral and central-CN at 1024² and 4096².
