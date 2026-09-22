# ops-deeponet-b2d — handoff

Branch `exp/2026-09-22-ops-deeponet-b2d`, worktree
`worktrees/2026-09-22-ops-deeponet-b2d`, forked from `exp/2026-09-17-no-second` @ `ea812685`.
Cluster namespace `/cluster/tufts/paralab/tawal01/opsdon_20260922/`. Nothing pushed, nothing
merged. Pre-registration: `DESIGN.md`. Read that before touching anything here.

## State

*(kept current; newest first)*

- **2026-09-22 — lane complete on its own question; 1 of 4 jobs used, nothing running.**
  `don01` (job 4179556) COMPLETED, collected, audited, reported, remote deleted, namespace
  empty. **DeepONet is the weakest of the four operator families on 2D Burgers at 256² by a
  wide margin** — selected arm `don-small`, validation-32 mean 14.7877 %, median 11.7522 %,
  worst 54.7444 %, against `fno-large`'s 2.2811 / 1.8054 / 6.3825 and `unet-medium`'s
  1.4341 / 1.3169 / 3.9622. Pre-registered D1 and D2 both **fail**; D3 (the honest negative)
  is the outcome. All four arms **early-stopped** at 543–898 s of a 3000 s budget, so the budget
  did not bind — but read §1/§2 of the report for what that does and does not establish, because
  two audits cut this claim down: early stopping means only that 250 epochs produced no new best
  *validation selection* score under *this* schedule (training loss was still falling in all
  four), three coupled capacity configurations do not rule out under-capacity, and there are only
  **128 training cases**, so data-limited generalisation is a live third explanation this lane
  does not separate out. The trivial persistence control scores 64.6850 % mean / 90.6492 % worst
  on the same cases, so these arms are ~4.4× better than doing nothing and ~7–11× worse than the
  other three families. Report `reports/2026-09-22-ops-deeponet-b2d.md`, rows
  `reports/summary.json`.
  **No speed claim is made** (DESIGN §A4); the handoff for the admissible route is
  `reports/timing-handoff.json`.

## Jobs

| attempt | spec | job id | GPU | state | what it is |
|---|---|---|---|---|---|
| `don01` | `specs/don01.json` | 4179556 | A100-80G pax050 | COMPLETED 00:45:45, collected, audited, remote deleted | DeepONet capacity screen: `small`/`medium`/`large` at 3000 s each + `refine` |

Job budget for this lane: **4 total, 1 running at a time.** Preamble deaths with zero
training GPU time do not count (the `no-second` rule), but they must be recorded with their
logs in `runs/<attempt>/`.

## What exists

- `families.py` — `DeepONet2d` added beside the inherited U-Net and Transolver.
- `check_inherited.py` → `checks/inherited-sources.json` — proves which files are byte-identical
  to `no-second` at the fork and that only the declared six differ. Run it after any edit.
- `reports/generate_report.py` — writes the report and `summary.json` from audit JSONs only;
  it reads this lane's `runs/don01/audit.json` plus four hash-pinned external sources (U-Net,
  Transolver, FNO, ROM/FOM). Every **measured** number is derived; the report's own preamble
  lists the handful of non-measurement literals (criterion bars, job ids, the two FNO split
  hashes) and says where each is pinned.
- `reports/codex-design-audit.md` — Codex could not run (sandbox failure); the independent
  audit is `reports/design-audit-2026-09-22.md`, disposition in `DESIGN.md` §A2.

## Next, whoever picks this up

1. **The one thing outstanding: a same-allocation timing row.** Hand
   `reports/timing-handoff.json` to the `ops-timing-panel` lane (it has 2 unused jobs, the
   gates, the comparators and the audit already built) rather than copying its 46-file harness
   here. That file has each checkpoint's path and SHA256, re-verified against the hash
   `train.py` recorded, and the `families.py`/`model.py` that harness needs for the family.
   **Never divide a time from `don01` by a time from another job** — DESIGN §5/§A4.
2. If anyone wants to argue the DeepONet deserves better, both audits agree on what the fair
   experiment is, and it is **not** a longer budget: a patience/schedule ablation, more training
   cases (there are 128), independent seeds, and a trunk/branch representation diagnostic — the
   per-output-time profile in §2 already points at the early, sharpest field rather than at error
   growth. Give that its own worktree and its own pre-registration; do not amend this one after
   the fact.
3. `runs/don01/archive/don01/` is `.gitignore`d. It rebuilds from
   `runs/don01/archive-parts/part-*` (`manifest.json` has each part's SHA256): concatenate,
   `tar -xzf`, then re-verify against `MANIFEST.sha256` / `OUTPUTS.sha256`.
