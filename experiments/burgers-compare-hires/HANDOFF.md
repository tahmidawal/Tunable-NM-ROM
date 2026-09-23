# HANDOFF — burgers-compare-hires (read DESIGN.md first; amendments A1–A6 at its end)

Branch `exp/2026-09-23-burgers-compare-hires` (LOCAL ONLY — coordinator: never push; the base carries unpushable
archive history). Namespace `/cluster/tufts/paralab/tawal01/bcmp_20260923/`. Lane cap: 2 running jobs.

## State

| attempt | job | what | state |
|---|---|---|---|
| t1024 | 4196040 | operators at 1024^2 (4 families) | done, collected, remote deleted → `operators-1024.json` |
| t2048 | 4196041 | operators at 2048^2 | U-Net, Transolver OK; FNO, DeepONet OOM (A100) |
| t2048r | 4200680 | retry on H200 | cancelled while PENDING (no H200 free) |
| t2048s | 4206643 | retry on A100 + expandable_segments | DeepONet OK; FNO OOM again |
| **p1024** | **4204019** | **1024^2 panel** | **done, 0 failed gates, local audit = remote audit (1.9e-15)** → `checks/p1024-summary.json` |
| p2048 | 4206695 | 2048^2 panel (1st staging) | cancelled while PENDING |
| p2048b | 4207177 | 2048^2 panel + in-job FNO training | JAX process killed (hash copy of 72 GB bank, A5); job OOM in FNO training; FNO checkpoint salvaged (A6); no numbers used |
| p2048c | 4215837 | 2048^2 panel rerun | died silently in POD-512 quick run (A6); no numbers used |
| p2048d | 4218300 | 2048^2 with late FNO pickup | cancelled while PENDING |
| **p2048e** | **4218390** | **2048^2 panel, H200 — ACCEPTED (A7/A9)** | **done, 0 failed gates, collected, remote deleted** → `checks/p2048e-summary.json`; qman r=64 dropped |
| **p2048f** | **4219197** | **2048^2 panel, A100 fallback (A7)** | **done, 0 failed gates, collected, remote deleted** → `checks/p2048f-summary.json`; POD-512 and qman r=64 dropped (A100) |

## Commands

```bash
cd worktrees/2026-09-23-burgers-compare-hires/experiments/burgers-compare-hires
PY=/home/tahmid/Dev/.venv/bin/python
$PY cluster/collect.py p2048e
$PY audit_cmp.py runs/p2048e/archive/output --operators runs/p2048e/archive/output/operators-runtime.json --out checks/p2048e-summary.json
$PY reports/make_report.py checks/p1024-summary.json checks/p2048e-summary.json \
    --out-md reports/2026-09-23-burgers-compare-hires.md --out-json reports/summary.json --status final
ssh tufts-login 'rm -rf /cluster/tufts/paralab/tawal01/bcmp_20260923/p2048e'
```

## Findings worth carrying

- repanel's `no_order_effect_between_arms` failure (br1024) is a case-mix artefact: all after-slow samples sit in their
  own case's timing range; paired within-case gap ≤ 1.2 % (`checks/order-gate-control.json`).
- Bank-span ρ: `lat64` within the bar at every R' and both meshes; `lat128` exceeds it, always at k = 0 (the
  initial-fit state); over time-stepped states (k ≥ 1) every arm is ≤ ~0.03.
- Operators at 1024^2/2048^2 got far fewer epochs than at 256^2 under the equal 3000 s budget (see the operator table).

## TO FINISH (A7): when p2048e ends

If it completed with its audit before 2026-09-24 12:00 EDT, it replaces p2048f as the 2048^2 panel:
```bash
$PY cluster/collect.py p2048e
$PY audit_cmp.py runs/p2048e/archive/output --operators runs/p2048e/archive/output/operators-runtime.json --out checks/p2048e-summary.json
$PY reports/make_report.py checks/p1024-summary.json checks/p2048e-summary.json --out-md reports/2026-09-23-burgers-compare-hires.md --out-json reports/summary.json --status final
ssh tufts-login 'rm -rf /cluster/tufts/paralab/tawal01/bcmp_20260923/p2048e'
```
Otherwise p2048f stands and the report status becomes final with p2048f. Either way: commit, delete the remote dir,
append a short lab-log line. Never mix ratios across the two jobs.
