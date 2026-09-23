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
| Independent subagent design audit | **done: 4 blockers, 17 majors, every finding accepted.** `reports/design-audit-2026-09-22.md`; disposition in `DESIGN.md` §A3 |
| `gen01` (data) | **submitted, job `4183681`**, a100, 13 h limit; `jax_backend=gpu` confirmed |
| `ladder01` (data ladder) | generated from `gen01`'s real indices by `make_ladder_spec.py` — do not hand-write it |
| `grid01` (tuning grid) | `specs/grid01.json`, 15 arms + 1 budget control |
| report generator | `reports/generate_report.py` builds today and names each missing job as a gap |

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

## The audit finding that mattered, so it is not re-learned

The ladder's learning-rate schedule and early stopping were indexed on **epochs**, and an epoch
is 16 gradient steps at 128 training cases but 576 at 4608. Left alone, the 4608 rung would
have reached ~55–70 epochs, fired `ReduceLROnPlateau` at most once, never been able to
early-stop, and **finished at its initial learning rate**, while the 128 rung annealed to the
1e-5 floor. The ladder would then have measured a schedule difference and reported it as a
data effect — the lane's headline, with a bias of unknown sign, unrepairable after the jobs ran.
`train.py` now takes `plateau_patience` and `make_ladder_spec.py` scales both patiences per
rung so they are constant in gradient steps. **Never hand-write a ladder spec.**

## Next session

1. Watch `gen01` (`4183681`). Its gates, in order: `jax_backend=gpu`, `disk_free_bytes=`,
   `REPRODUCTION arrays_identical=True`, `INDEX index-*.json` lines as each prefix lands,
   `DISCRETISATION` rows, `GENERATION FINISHED`, `WORKER FINISHED`, `ALL-DONE`.
2. **Check the reproduction gate and control G1 before trusting any ladder rung.** G1's
   pre-registered threshold is 0.5 % worst; above it the cheap-fidelity rungs are labelled
   confounded.
3. Collect `gen01`, then `python make_ladder_spec.py --cache-json
   runs/gen01/archive/gen01/out/cache.json`, commit, stage, submit `ladder01`.
4. Then `grid01`. If the budget runs short, **cut `grid01`, not `ladder01`** — data parity is
   the reviewer-facing question.
5. Regenerate the report (`python reports/generate_report.py`) — never hand-type a number.
6. Append to `LAB-LOG.md` (via the `flock` on `.lablog.lock`) including anything retracted.
7. Delete `/cluster/tufts/paralab/tawal01/opstune_grid_20260922/` when the lane closes — the
   cache it leaves behind is ~20 GB and the lane peaks near 60 GB on a nearly-full share.

## One thing the constraint cost, recorded for whoever sets the next budget

`ladder01` trains the **published** configurations, so it has no dependency on `grid01`; the
two could have run **concurrently** in their own directories and finished roughly 18 hours
sooner. This lane is allotted one running job, so they run in series. If the deadline binds,
that allotment is the thing to change — not the science.
