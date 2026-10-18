# Lane brief: jcp-p1-scaling (priority 1: P1a–P1g + Q3)

Read `briefs/COMMON-BRIEF.md` first; it binds you. This brief adds your lane specifics.

## Where
- Worktree `worktrees/2026-10-17-jcp-p1-scaling` (exists: setup commit `8bd0d20b6` — vendor, sync
  script, README; forked at `21c175a1b`). Branch `exp/2026-10-17-jcp-p1-scaling`. Lane dir
  `experiments/jcp-p1-scaling`. Namespace `/cluster/tufts/paralab/tawal01/jcp-p1-scaling/`.
- Concurrent-job cap: **4** (jcp-speed3d holds 3; campaign total 8).
- **GPU-hour cap: 20 GPU-h including contingency**, cumulative ledger; "P1 incomplete" if the core can't
  finish (see the design).

## What you own and how
Your design is fixed in outline by the coordinator's restatement, **as corrected by its Codex audit**:
- `campaign/P1-RESTATEMENT.md` (the restated design: items, frozen configurations, decision rules,
  controls, resources, missing code, deliverables);
- `campaign/P1-RESTATEMENT-AMENDMENTS.md` (the coordinator's fixes after the Codex audit; supersedes the
  restatement where they differ);
- `campaign/codex-p1-restatement-*.md` (the audits themselves; read them for the traps).
Your `DESIGN.md` implements all of it, with every decision rule, threshold, control and defer step stated
before any data. Then your own Codex design audit, as in the lane rules.

## Inputs (read report.md and DESIGN.md of each)
- `worktrees/2026-10-13-jcp-combined` (2D/3D code: `c2common.py`, `t2core.py`, `c2grid.py`, `c2bud.py`,
  `score.py`, `make_e0_registry.py`; reused 1024² arrays `runs/c2grid/archive/output/W_R{128,256,384}.npz`;
  3D `runs/c3grid/archive/output/W_grid.npz`; finalist table).
- `worktrees/2026-10-08-jcp-wide-bank` (configurable M, `w2d.py`, `w3d.py`, W1024 bank).
- `worktrees/2026-10-06-jcp-references` (E0 store, `refstore.py`).
- `worktrees/2026-10-14-jcp-speed3d` (pattern for `resources.py` cost certificates, Q1 contract,
  timing harness `timing.py`). **Read-only:** another agent is actively working there.
- Copy code you need into your lane dir with a provenance note (do not edit `vendor/`).

## Hand-back
Per the common brief §10, plus `runs/EXPORT.json` with every hypothesis verdict (per r, scheme, panel),
the key numbers behind each, and the plots list.
