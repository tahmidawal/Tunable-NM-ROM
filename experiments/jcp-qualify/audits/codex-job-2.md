**FIX-FIRST.** Audited HEAD `34d7df35f`. No files changed, project scripts run, GPU use, or cluster jobs launched. Paths below are relative to `experiments/jcp-qualify/`.

| Previous must-fix | Verdict | Evidence |
|---|---|---|
| 1. Fallback timing identity | **PARTIALLY** | Reservation now requires `key\|fb` (`ledger.py:167`), but staging still compares original keys (`cluster/stage.py:53,60`). Reproduced: ordinary panel passes staging without fallback subjects. |
| 2. Underpriced fallback obligations | **PARTIALLY** | Pending fallback minimum raised to 2 hours; first-panel recertification added (`ledger.py:135,193–204`). However, a completed original timing panel plus an uncertified **0.25-hour repeat** still passes validation (`ledger.py:141–144`). |
| 3. Incomplete analysis releases allowance | **PARTIALLY** | Missing configuration keys now refuse (`ledger.py:94–98,200`). Provenance remains unchecked: reproduced all expected keys with unrelated source/attempt and “pass” labels returning `[]`, permitting allowance release. |
| 4. Conservative order/3D proxies | **PARTIALLY** | Q3D now uses the measured **118.8-second maximum** (`resources.py:111–117`). Order still divides a heterogeneous pooled block by six; that is not an established upper bound for its singleton arms (`resources.py:74–78`). Increasing the estimate does not establish the claimed conservatism. |

**New diff bug:** `t_live` includes **FAILED/TIMEOUT/CANCELLED** timing jobs (`ledger.py:190`). Reproduced a failed `time256`, recorded fallback requirement, unchanged original timing reservation, and 2-hour fallback placeholder passing validation. This bypasses first-panel enlargement/recertification through `ledger.py:202`.

**Staging passes:** all five trees match HEAD: **137/137 provenance entries and 141/141 manifest hashes each**. Config hashes, run lists, commit records, reservation hours, and sbatch limits agree (`runs/<attempt>/PROVENANCE.json:1`, `RESERVATION.json:1`).

| Job | Projection → limit (hours) | ≥1.5× |
|---|---:|---|
| acc256 | 1.718 → 2.75 | Pass |
| acc1024 | 1.542 → 2.50 | Pass |
| acc4096 | 3.284 → 5.00 | Pass |
| order | 1.487 → 2.25 | Pass |
| q3d | 1.046 → 1.75 | Pass |

Certificates: `checks/resources-<attempt>.json:3–4`.

**Installed budget arithmetic passes**, conditional on projections (`runs/GPU-LEDGER.json:41–54`):

- Spent **0.5039** + pending **32.25** = **32.7539 GPU-h**, leaving **2.2461** under cap 35.
- After the three accuracy reservations: reserved **10.25**, pending **22.00**.
- After reserving order and q3d: reserved **14.25**, pending **18.00**.
- Total remains **32.7539** before replacing completed reservations with actual elapsed time. Later jobs must await lane slots; live queue availability was not checked.

Staging and arithmetic pass; the unresolved guards and order proxy prevent **SUBMIT**.