# ops-deeponet-b2d — handoff

Branch `exp/2026-09-22-ops-deeponet-b2d`, worktree
`worktrees/2026-09-22-ops-deeponet-b2d`, forked from `exp/2026-09-17-no-second` @ `ea812685`.
Cluster namespace `/cluster/tufts/paralab/tawal01/opsdon_20260922/`. Nothing pushed, nothing
merged. Pre-registration: `DESIGN.md`. Read that before touching anything here.

## State

*(kept current; newest first)*

- **2026-09-22** — lane created; DeepONet family written, smoked locally, pre-registered,
  independently audited. Job `don01` staged/submitted. See §Jobs.

## Jobs

| attempt | spec | job id | GPU | state | what it is |
|---|---|---|---|---|---|
| `don01` | `specs/don01.json` | *(see below)* | A100-80G | *(see below)* | DeepONet capacity screen: `small`/`medium`/`large` at 3000 s each + `refine` |

Job budget for this lane: **4 total, 1 running at a time.** Preamble deaths with zero
training GPU time do not count (the `no-second` rule), but they must be recorded with their
logs in `runs/<attempt>/`.

## What exists

- `families.py` — `DeepONet2d` added beside the inherited U-Net and Transolver.
- `check_inherited.py` → `checks/inherited-sources.json` — proves which files are byte-identical
  to `no-second` at the fork and that only the declared six differ. Run it after any edit.
- `reports/generate_report.py` — writes the report and `summary.json` from audit JSONs only;
  it reads this lane's `runs/don01/audit.json` plus four hash-pinned external sources (U-Net,
  Transolver, FNO, ROM/FOM). Nothing is typed.
- `reports/codex-design-audit.md` — Codex could not run (sandbox failure); the independent
  audit is `reports/design-audit-2026-09-22.md`, disposition in `DESIGN.md` §A2.

## Next, in order

1. Collect `don01`, run `audit.py don01`, regenerate the report, commit.
2. Decide on timing: a same-allocation panel (`b-panel` harness) or an explicit statement that
   no speed claim is admissible. **Do not divide a time from `don01` by a time from any other
   job** — DESIGN §5.
3. Append the lab-log entry under `flock`.
