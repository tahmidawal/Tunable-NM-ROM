# HANDOFF — burgers-eqcert (kept current; read this first after an interruption)

**Lane:** `experiments/burgers-eqcert/` in `worktrees/2026-09-21-burgers-eqcert`
(`exp/2026-09-21-burgers-eqcert`, fork `exp/2026-09-20-hires-burgers` @ `0ab60014`). Cluster namespace
`/cluster/tufts/paralab/tawal01/bcert_20260921/`. Contract: `reports/2026-09-20-speed-accuracy-campaign-protocol.md`
(on main). Budget: ≤ 2 running (lane), ≤ 8 total, account-wide wait if ≥ 6 running. Hard stop 2026-09-24 12:00 EDT.
Design: `DESIGN.md` (+ Addendum A1, Codex dispositions; audit record `reports/codex-design-audit-2026-09-21.md`).

## State (2026-09-21, resumed after the pause)

- Resumed at user request. DESIGN.md written, Codex audit done (12 findings, dispositions A1), code + configs ready,
  local 64² smoke passes every path; audit script tested on the smoke (NumPy ρ recomputation agrees to 5e-12).
- **Exploration (local, not results; `explore/`)**: on initial-fit states $w_0$ of `params_draw(0,128)` trajectories
  8–47, lat64 ρ_max = 0.52 / 0.30 / 0.22 / 0.18 at 256² / 512² / 1024² / 2048² (bar 0.116); worst trajectories
  18, 22 (hires-burgers sampled only 8–15). So **hires-burgers' lat64 "certified" at 2048²/4096² is a single-draw
  result** — flagged in the lab log; not re-certified by this lane unless spare budget (DESIGN §6). Evolved states
  at 256²: worst is k=1 (lat64 0.139, every-2nd-node 0.078), k≥2 ≤ 0.033 → the exact-first-step arms (`xfast.py`).
- Jobs used: see table. Nothing else running for this lane.

## Files

- `eqcert.py` driver (from hires.py @ 0ab60014), `xfast.py` (exact-first-steps), `audit_eqcert.py` (NumPy audit),
  `config-{256,512,1024}.json`, `config-smoke64.json`, `cluster/{stage.py,collect.py,submit.sh}`.

## How to run a job

```bash
cd worktrees/2026-09-21-burgers-eqcert          # commit first: stage.py refuses uncommitted files
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/burgers-eqcert/cluster/stage.py <attempt> config-<L>.json --gpu a100-80G|h200 --mem 240G
experiments/burgers-eqcert/cluster/submit.sh <attempt>      # waits for slots; squeue before/after
# when done:
$PY experiments/burgers-eqcert/cluster/collect.py <attempt>
$PY experiments/burgers-eqcert/audit_eqcert.py experiments/burgers-eqcert/runs/<attempt>/archive \
    --checkpoint experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl \
    --out experiments/burgers-eqcert/checks/<attempt>-summary.json
ssh tufts-login 'rm -rf /cluster/tufts/paralab/tawal01/bcert_20260921/<attempt>'   # only that attempt dir
```

## Jobs

| attempt | job id | mesh | GPU | state | summary |
|---|---|---|---|---|---|
| bc256 | 4139288 | 256², dev6 | A100-80G | submitted 16:11 EDT 09-21 (staged at f80f0a88) | — |
| bc1024 | 4139290 | 1024², dev6 | H200, 240G | submitted 16:12 EDT 09-21 (staged at f80f0a88) | — |

Next: when one finishes → collect, audit, delete remote dir, commit summary; then stage/submit `bc512`
(`config-512.json --gpu h200 --mem 240G`). Jobs used: 2 / 8.
