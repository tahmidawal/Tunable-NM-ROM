# ops-tune-grid — handoff

**Read `DESIGN.md` first.** It is the pre-registration and it is authoritative; this file
only says where the lane got to. Branch `exp/2026-09-22-ops-tune-grid`, forked from
`exp/2026-09-22-ops-deeponet-b2d` at `306c939d`. Cluster namespace
`/cluster/tufts/paralab/tawal01/opstune_grid_20260922/`.

## What this lane is for

The user's concern: *"I think we should try and properly tune them so that our comparison
doesn't get questioned by reviewers."* Two objections, in priority order:

1. **Data parity.** The published FNO / U-Net / Transolver arms trained on **128 cases**; our
   own Burgers head trained on **4608 trajectories**. Never controlled.
2. **Tuning.** No operator family was ever tuned; every hyperparameter was inherited from the
   FNO lane.

## Status

| | state |
|---|---|
| `DESIGN.md` | written and pre-registered before any GPU job |
| Codex design audit | ran; its sandbox could not read files (the recorded `bwrap` failure), so it audited the prompt's numbers only. Two findings, **both accepted and fixed** — see §A2 |
| Independent subagent design audit | commissioned with the same adversarial brief |
| Local gates | contract smoke, both training smokes and `smoke_tune.py` over all 23 configs: **pass** |
| `gen01` (data) | staged, not yet submitted |
| `ladder01` (data ladder) | spec written |
| `grid01` (tuning grid) | spec written |

## The one thing a reader should not miss

The 128-vs-4608 gap is **not** a design choice — it is a data-generation-cost artefact. The
operators' training targets were held to a far stricter reference than our own model's:

- operator training case: 4096 intervals, $\Delta t = 1.5625\times10^{-4}$, restricted to
  256 — **135 s per trajectory**;
- NM-ROM bank/head trajectory: 256 intervals, $\Delta t = 0.005$, solved directly —
  **0.19 s per trajectory**.

128 cases is what ~4.8 GPU-hours buys at the first rate. This lane generates the extended
bank at 1024 intervals / $\Delta t = 3.125\times10^{-4}$ (**4.8 s per case**), which the
frozen calibration measured at ≤0.31 % from the pinned reference — still **five or more times
finer than the targets our own head trained on** — and controls that choice two ways (G1
measures it on the real training cases; G2 retrains at fixed data size on both fidelities).

## Job order and why

`gen01` → `ladder01` → `grid01`, at most one running, six total.

`ladder01` deliberately runs **before** `grid01` and uses each family's **published**
configuration unchanged, so the only thing varying along a ladder is the number of training
cases. Putting the tuned configurations in the ladder would confound tuning with data, and
would also make the ladder wait on the grid.

## Rules this lane operates under

- No speed number from this lane is admissible. Arms worth timing go to `ops-timing-panel` in
  its harness format; see `DESIGN.md` §8.
- Nothing is selected on a final or held-out cohort. Selection is on validation-32, which is
  therefore the **tuning** cohort and is labelled optimistic wherever a tuned number appears;
  diagnosis-8 is the cohort nothing was selected on.
- Never push, never merge, never create worktrees. Do not touch root `paper/`, `main`, other
  worktrees or other namespaces.

## Next session

1. If `gen01` has not run: submit it (`cluster/stage.py gen01 gen01.json --gpu a100`, then
   `sbatch`), check `squeue -u tawal01` before **and** after.
2. Check its reproduction gate and control G1 before trusting any ladder rung.
3. Then `ladder01`, then `grid01`.
4. Append to `LAB-LOG.md` (via the `flock` on `.lablog.lock`) including anything retracted.
5. Delete `/cluster/tufts/paralab/tawal01/opstune_grid_20260922/` when the lane closes — the
   shared cache it leaves behind is ~20 GB on a 91 %-full share.
