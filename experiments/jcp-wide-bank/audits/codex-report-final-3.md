Read-only re-audit complete; no files modified.

| Item | Disposition |
|---|---|
| Failed-job provenance | **CLOSED** — generator reads `runs/JOB-LEDGER.json`; report matches its six attempts. |
| Floor-table status | **CLOSED** — caption explicitly identifies native-grid bank-validation span floors, distinct from refined-reference accuracy. |
| Opening status wording | **CLOSED** — qualification now applies to rollout accuracy, with bank floors separately identified. |
| Nominal κ glossary | **CLOSED** — distinguishes exact 2D ratio from shell-completed 3D test counts. |
| Registered diagnostics | **NOT-CLOSED** — 3D exits/rejected sweeps and 2D iterations/exits/memory are present, but **2D rejected-step counts remain omitted without an omission statement** ([report.md:107](experiments/jcp-wide-bank/report.md#L107)). JSON `rejected_total` sums to zero for every deployed 2D setting; report that explicitly. |
| Memory plot hollow-series identity | **CLOSED** — legend identifies W1024, 128³. |
| A8 historical-lane comparability | **CLOSED** — explicitly warns that J4 differs from both J2 and the historical three-adaptive-step lane. |
| `3d_j2_mstar_vs_M.png` | **CLOSED** — visually verified: rank-labelled points, no cross-rank connecting lines. |

**Five new numerical spot-checks: 5/5 PASS.**

| JSON source and setting | Report value verified |
|---|---|
| J1 `result.json`, R512_M2048, deployed Gauss192 | Median iterations: **55** |
| Same | Budget/damping-limit exits: **0** |
| Same | Sampled memory: **52.7 GB** (52.650793472 GB raw) |
| J2 `result_A7.json`, R512_k2, converged arm | Non-stationary steps: **3** |
| J4 `result.json`, 129-node W1024 R1024_k4, converged arm | Non-stationary steps: **2** |

REPORT-OK: NO