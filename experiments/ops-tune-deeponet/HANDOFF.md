# ops-tune-deeponet — handoff

Branch `exp/2026-09-22-ops-tune-deeponet`, worktree `worktrees/2026-09-22-ops-tune-deeponet`,
forked from `exp/2026-09-22-ops-deeponet-b2d` @ `306c939d`. Cluster namespace
`/cluster/tufts/paralab/tawal01/opstune_don_20260922/`. Nothing pushed, nothing merged.
Pre-registration: `DESIGN.md` — read it, and §A2/§A3 in particular, before touching anything.

The question: was the paper's DeepONet baseline weak because it was **data-starved** (128
training cases against the 4608 trajectories our NM-ROM's head saw) or **untuned** (an inherited
U-Net schedule), and how strong can it honestly be made before the ICLR deadline?

## State

*(kept current; newest first)*

- **2026-09-22, evening — generation running; no accuracy result yet.**
  - `gen02` (job 4183561) is generating train cases 0…4607 on an A100-80G. It **chose the
    1024-interval, $\Delta t = 3.125\times10^{-4}$ reference setting** (the finer of the two
    candidates) because the in-job profile projected 19479 s against a 23689 s budget; realised
    rate ≈ 4.4 s/case, so ≈ 5.7 h for the full 4608. Its worst label deviation from the pinned
    4096-anchor targets, as measured by the calibration on 8 independent cases, is 1.87e-3 —
    and the job measures the real one on the first 128 cases, which are the pinned cache's own
    draws.
  - `gen01` (job 4183329) **FAILED after 45 s** and is recorded in `runs/gen01/FAILED.md`:
    `data.provenance` derives source paths from `data.py`'s own location, which does not exist
    in the flat staged `code/` tree. No data, 45 s of A100. Counts as a submission.
  - **Two independent design audits ran before the first job** and both found the same two
    blockers: the pinned 128-case cache was produced by `refine.py` + `protocol-refined.json`,
    not `data.py` + `protocol.json` as the design claimed, and the `s-rank` sweep arm could not
    instantiate (`trunk_width >= rank`). Codex: 37 findings. Claude subagent: 7 blockers, 15
    major. Dispositions in `DESIGN.md` §A2 and §A3; reports in `reports/`.

## Jobs

Budget **≤ 1 running, ≤ 6 total submitted, counted without exception**. The account cap of 6
running is shared with three other lanes, so check `squeue -u tawal01` before and after every
submission.

| attempt | spec | job id | state | what it is |
|---|---|---|---|---|
| `gen01` | `specs/gen01.json` | 4183329 | FAILED 00:00:45, logs pulled, remote deleted | generation; died in `data.provenance` |
| `gen02` | `specs/gen02.json` | 4183561 | RUNNING | generation, 4608 cases at 1024/3.125e-4 → `.../opstune_don_20260922/pool02` |
| `lad01` | `specs/lad01.json` | — | staged, waiting on `gen02` | the data ladder: `c-pinned128`, `c-new128`, base at 512 / 2048 / top, 3000 s each |
| `tun01` | `specs/tun01.json` | — | waiting on `lad01` | the sweep: `s-base` + 11 one-factor arms at 1500 s, then the composed arm |
| `fin01` | to be written after `tun01` | — | — | the selected recipe (and the best-worst-case arm if different) at 9000 s |

## The one thing that must not be lost

**The generated cache `/cluster/tufts/paralab/tawal01/opstune_don_20260922/pool02` is ~34 GB and
lives outside every job directory.** It is made read-only when complete and its `DATA.sha256`
covers the per-case files, the index and the packed pool arrays. It is **deleted only after
`fin01`**, and its deletion must be recorded with what regenerates it: commit, the recorded
reference setting and `data.case_seed('train', i)`. Its **index and generation report are
archived** with every job that uses it; its 17 GB of fields are not.

## What exists

- `gen_more.py` — extends the train split past 128 cases on the pinned generator. Vendors all
  six files the pinned index records as its provenance, calls `refine.configure()`, and reads
  the expected hashes **out of the pinned index at run time**. `self-check` proves its
  warm-up-free `solve` is bitwise `data.solve`.
- `build_pool.py` — runs `dataset.load_pair` over the whole split (every checksum, schema,
  boundary and disjointness gate) once per cache, then packs three `.npy` arrays whose hashes
  join the cache manifest. `train.py --pool --pool-limit N` trains on the first N cases.
- `train.py` — wall-fraction validation cadence (200 evaluations per budget, patience 50), an
  optional cosine schedule, per-output-time output scaling, a forced final evaluation, the
  training-subset error at the selected checkpoint. Every option defaults to the inherited
  behaviour.
- `compose.py` — DESIGN §5.3's composition rule as code, so it cannot drift.
- `audit.py` — **rewritten**, not inherited: the parent's version would abort every job here on
  its FNO-index and hyperparameter literals. It checks each arm against its own declared data
  source and prefix, rebuilds each arm's config from base + declared override, and recomputes
  S1/S2/S3 independently.
- `check_inherited.py` → `checks/inherited-sources.json` — proves what is byte-identical to the
  fork and to the pinned generator, and that nothing undeclared is present. **Run it after every
  edit**; it currently passes.
- `reports/accounting.py` → `reports/accounting.json` — DESIGN §2's accounting derived from
  pinned artefacts, including the validation-to-training nearest-neighbour distances per rung.
- `checks/pinned/` — byte copies of the pinned train and validation indices, whose sha256 **are**
  the `5333584b…` / `468b9e70…` literals every operator lane asserts, plus the calibration gate
  slice and the NM-ROM checkpoint's configuration block.

## Next, whoever picks this up

1. **When `gen02` finishes:** check `out/gen-summary.json` for the realised case count, the
   chosen setting and the measured label discrepancy; then `python cluster/stage.py lad01
   lad01.json --gpu a100`, rsync, `sbatch`, checking `squeue` before and after. If it returned
   fewer than 4608 cases, edit `specs/lad01.json`'s `pool_limit`s to the milestones it reached
   — the rungs must be prefixes that exist.
2. **Collection, every time:** `cluster/collect.py <attempt>` → `audit.py <attempt>` →
   `cluster/collect.py <attempt> --preserve` → `--cleanup`. The archive is chunked into 45 MiB
   parts; at 13 arms it may be ~2 GB, and if that is too much for the repository, commit the
   audit and the JSONs and record the archive's hash and what regenerates it rather than
   silently dropping it.
3. **What this lane will not have done**, and what a hostile reviewer will still ask for: a
   trunk/branch representation diagnostic, independent seeds, a fresh confirmation cohort after
   all this validation-32 reuse, and — if any arm reaches ~2 % error — a re-derivation of the
   top rung on pinned-protocol targets, because the cheap targets stop being negligible exactly
   there (DESIGN §5.2 T0).
4. **No speed number is admissible from this lane.** If a tuned arm is worth timing, hand its
   checkpoint path and SHA256 to `ops-timing-panel` in the format of
   `experiments/ops-deeponet-b2d/reports/timing-handoff.json`.
