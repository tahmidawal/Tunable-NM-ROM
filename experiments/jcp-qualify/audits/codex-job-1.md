**FIX-FIRST.** Read-only audit; no files changed, scripts executed, or GPU jobs launched. Paths below are relative to `experiments/jcp-qualify/`.

Must-fix, ranked:

1. **WRONG — fallback timing coverage checks the wrong identity.** `fbstate` records original keys; fallback timing subjects append `|fb`. An ordinary panel containing the original configuration therefore passes without its fallback. Reproduced in memory using `configs/time256.json`. Fix both staging and reservation checks.  
   `ledger.py:74,92,134,161`; `cluster/stage.py:59–60`.

2. **NEEDS-FIX — pending fallback obligations remain underpriced.** In-memory validation accepts replacing the 2-hour allowance with `fb256placeholder: 0.25`, leaving `time256` unchanged. Neither fallback coverage nor an enlarged timing projection is required for uncertified pending jobs. A3.1/A4.1/A5.1 remain bypassable.  
   `ledger.py:108–115,130–138`.

3. **NEEDS-FIX — incomplete analysis can release fallback allowances.** `failing_from_analysis('256')` returns `[]` when analysis contains no 256 records. The transaction then treats Q2 as known with no fallback required. Atomic recording is fixed; completeness/provenance validation is missing.  
   `ledger.py:89–93,179–193`; `analyze.py:191–193`.

4. **NEEDS-FIX — order/3D proxies are not demonstrated conservative.** Order scales a pooled 599.83-second block containing three single-case arms and six TH06 cases as though their costs were interchangeable. Q3D uses 71.87 seconds/member, or 107.8 after its multiplier, while the measured W1024 member took 118.8 seconds. Fixed overhead and scheduler margins may cover this, but do not establish conservative per-member proxies. Reprice using measured upper envelopes or justified stratification, then update certificates/reservations. This is **not** evidence that the existing jobs will time out.  
   `resources.py:73–78,105–112`; `runs/smkA/archive/output/qacc_smkA/result.json:5350`; `runs/smk3/archive/logs/5366486.out:15–19`.

| Job/evidence item | Verdict | Finding |
|---|---|---|
| 2D registered matrix | **CORRECT** | 30/28/30 new configurations at 256/1024/4096: 88 total; registered widths, rules, schemes and larger steps. `configs/acc256.json:1`, `configs/acc1024.json:1`, `configs/acc4096.json:1` |
| 4096 ext22 | **CORRECT** | Four inherited R128/R256 BE/CNR configurations add exactly val32 cases 10–31, with prod/tight execution. `make_configs.py:51–54`; `qacc.py:175–201` |
| Order configuration | **CORRECT** | R128/R384, dev6, registered ladders and anchors; R384 TH06 control with its four-point ladder. `configs/order.json:30`; `qacc.py:264–289` |
| Q3D configuration | **CORRECT** | All 17 registered members plus control; cohort 923801 ×64, finer check cases 0–7. `configs/q3d.json:1` |
| Staged trees | **CORRECT** | All five match HEAD `6f44d5d719`; each has 137 matching provenance entries and 141 passing manifest entries. Certificate config hashes match. `runs/{acc256,acc1024,acc4096,order,q3d}/PROVENANCE.json:1` |
| Sbatch requirements | **CORRECT** | Requested GPU types, gpu partition, exit-42 preflight, x64/highest, isolated paralab paths. Each `run.sbatch:3,16,27` |
| Smoke evidence | **CORRECT** | **85/85** required checks pass; all three GPU logs complete, stderr empty; all 44 output checksums match. `checks/smoke-accept.json:1`; smoke logs `:4` |
| 4096 memory gate | **CORRECT** | **96.592 GB <120 GB**, feasible. `runs/smkB/archive/output/qpilot_pilot/result.json:564` |
| Installed budget | **CORRECT, conditional on projections** | Spent 0.5039 + pending 31.5 = **32.0039 GPU-h**, leaving **2.9961**. Sequential reservation of all five preserves that total. `runs/GPU-LEDGER.json:41–54` |
| Concurrency | **CORRECT at inspection** | Queue: 0 lane /3 account jobs. Three accuracy submissions would reach 3/6; order/q3d must await lane slots. Checks enforce 3/8. `cluster/submit.sh:18–24` |

All scheduler limits exceed 1.5× their stored projections:

| Job | Projection → limit, hours |
|---|---:|
| acc256 | 1.718 → 2.75 |
| acc1024 | 1.542 → 2.50 |
| acc4096 | 3.284 → 5.00 |
| order | 1.098 → 1.75 |
| q3d | 0.872 → 1.50 |

Evidence: each `checks/resources-<attempt>.json:3–4`.

| `codex-code-8` item | Resolution |
|---|---|
| Baseline reset bypass | **PARTIALLY** — zero rejected; enlarged pending pricing still unenforced. |
| Unverified coverage/placeholders | **PARTIALLY** — tiny values rejected; 0.25-hour placeholders still pass. |
| Enlarged-panel lifecycle | **PARTIALLY** — state handling exists, but identity mismatch invalidates coverage. |
| Subject counts charged as hours | **RESOLVED** — legacy keys rejected. |
| Failed fallback can disappear | **PARTIALLY** — completed coverage checked; unverified pending placeholder still substitutes. |
| Staging-to-submission gap | **PARTIALLY** — reservation recheck added, but checks original identities. |
| Unverified Q2 state | **PARTIALLY** — atomic, analysis-derived transaction added; incomplete analysis still releases allowance. |

The configurations support the registered qualification/order questions. B5 fronts still require the separately reserved timing jobs. **Do not submit under the current audit verdict.**