# Lane imp-multimesh — E7 (ideation W4): full multi-mesh cost, against a fair control

Worktree `worktrees/2026-10-21-imp-multimesh`, branch `exp/2026-10-21-imp-multimesh`, lane dir
`experiments/imp-multimesh/`, namespace `/cluster/tufts/paralab/tawal01/imp-multimesh/`.
**GPU-hour cap: 8. Lane concurrent-job cap: 2.** Wave 1. Read `COMMON-BRIEF.md` first (binding).

## Problem it answers
Problem 2, and the paper's remaining cost claim (mesh-independent cost, mesh-free evaluation). Ideation F14b
and §5 row W4.

## Experiment
Registered workloads: 1–3 meshes × {10, 100, 1000} queries (choose the mesh set from 256²–4096² in DESIGN),
**full cost = offline setup (training/snapshots/basis/rule build, amortised as stated) + all queries**.
Arms: ours (one bank, one off-mesh rule, reused across meshes); **per-mesh POD + ECSW** (tuned per mesh, rebuilt
per mesh); the **fair control: continuous POD built once and reused** (with refit and with reused rules).
All accuracy-matched (worst and median error on the same cases, E0 ST), timed on one GPU type.

## Stop rule (verbatim; pre-register it)
If the continuous-POD control matches us (no more cost at equal or better worst error) on every workload,
**drop the comparative multi-mesh cost claim** (mesh-independent evolution and mesh-free evaluation remain
properties of the method, not comparative advantages).

## Build on
jcp-hr-baselines (POD + ECSW build/tune and its measured offline costs), jcp-p1-scaling (mesh-cost panel),
jcp-budget (offline cost inventory), jcp-combined (`offline.py`, `inventory.py`), jcp-references.
Re-use measured offline costs where they were measured on the same GPU type and say so; otherwise re-measure.
