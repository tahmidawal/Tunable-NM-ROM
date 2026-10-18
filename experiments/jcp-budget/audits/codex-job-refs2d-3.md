**No remaining blockers found for the five restaged jobs at HEAD `9a7f8d28a`.**

| Item | Assessment | Evidence |
|---|---|---|
| 1. Coverage | **CORRECT** | Exactly 38 distinct FD cases: dev6 0–5 and val32 0–31, split 13/13/12. PS splits {0,1}/{5,19}. No sealed/OOD cases; outputs are job-local, never E0. |
| 2. Scripts and staging | **CORRECT** | H200, `gpu`, 240G, GPU preflight, x64/highest, isolated directories. All five manifests verify; each complete 150-file source inventory matches pinned HEAD byte-for-byte. |
| 3. Certificates | **CORRECT** | Independently reproduced all projections from the 278.1113-second smoke measurement and PS model. FD reserve now covers the previously omitted retry/refined-tight paths under A7’s **two difficult cases per job** assumption. Memory estimates are recorded. |
| 4. Collection and analysis | **CORRECT** | Writer/promoter/analyzer layouts agree. A7 specifies serial verified promotion, full ref2a collection, promotion before `--no-fine` cleanup, and joint analysis of all five archives. |
| 5. Release | **CORRECT** | A7 permits 2D production on smk1 alone; acceptance reports `all_passed: true`. |

The resource projections remain estimates, not runtime guarantees. PS’s former **3 hours for four cases was insufficient**; the two 3.5-hour jobs cover the revised attempt-0 estimate. Resolution escalation requires a separate certificate.

| Job | Projection | Required 1.5× | Staged limit | Verdict |
|---|---:|---:|---:|---|
| ref2a | 3.5322 h | 5.2983 h | 5.50 h | **SUBMIT** |
| ref2b | 3.3656 h | 5.0483 h | 5.25 h | **SUBMIT** |
| ref2c | 3.2883 h | 4.9325 h | 5.00 h | **SUBMIT** |
| ps2a | 2.2900 h | 3.4350 h | 3.50 h | **SUBMIT** |
| ps2b | 2.2900 h | 3.4350 h | 3.50 h | **SUBMIT** |

Blocking fixes: **none**.

Read-only audit completed: no file changes, SSH, submissions, or array copies.