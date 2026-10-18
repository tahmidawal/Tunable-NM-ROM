**FIX-FIRST** — HEAD `739c96149`. Read-only; no files changed, scripts run, GPU use, or cluster access.

Paths relative to `experiments/jcp-qualify/`.

| Prior finding | Verdict | Evidence |
|---|---|---|
| Fallback timing identity | **RESOLVED** | `cluster/stage.py:53`, `ledger.py:179` both require `key\|fb`. |
| Underpriced timing repeat | **RESOLVED** | Original certificate floor inherited at `ledger.py:123–126`. |
| Analysis provenance | **RESOLVED** | Exact `lane <completed-attempt>` required at `ledger.py:98–101`. In-memory checks reject unrelated sources and failed attempts. |
| Order/3D proxy claims | **RESOLVED by restatement** | Estimates disclosed at `DESIGN.md:583–586`; misleading order-bound comment corrected at `resources.py:75`. |
| Failed timing panel counted live | **RESOLVED** | `ledger.py:202–203` excludes failed terminal attempts. |
| A6 retry-name conflict | **RESOLVED** | Retry whitelist and mandatory-name mapping at `ledger.py:117–133`; doubled-limit replacements validate. |
| A6 overall | **PARTIALLY** | Smoke evidence confirms 85/85; retry reservation enforcement has the new defect below. |

**New blocker:** accuracy/order/q3d retries bypass inherited certificate floors. `ledger.py:123–126` inherits floors only for timing repeats; `:127–129` checks other retries only when their own certificate already exists, while `:130–133` immediately accepts them as mandatory coverage.

Reproduced independently for timed-out `acc256`, `order`, and `q3d`: replacing the mandatory pending entry with `<attempt>r1: 0.25` returns **no validation errors**. A6 requires doubled limits of **5.50 / 4.50 / 3.50 hours**, respectively (`DESIGN.md:586–588`). Other submissions can consume that unreserved obligation. Enforce the retry floor before accepting replacement inventory.

All five staged trees equal HEAD: **137/137 provenance entries and 141/141 manifest hashes each**. Commit records, config hashes, run lists, certificates, reservations, and sbatch limits agree.

| Job | Projection → limit, hours | ≥1.5× |
|---|---:|---|
| acc256 | 1.718 → 2.75 | Pass |
| acc1024 | 1.542 → 2.50 | Pass |
| acc4096 | 3.284 → 5.00 | Pass |
| order | 1.487 → 2.25 | Pass |
| q3d | 1.046 → 1.75 | Pass |

Installed inventory arithmetic passes (`runs/GPU-LEDGER.json:41`):

- Spent **0.5039** + pending **32.25** = **32.7539 GPU-h**; headroom **2.2461**.
- After accuracy reservations: reserved **10.25**, pending **22.00**.
- After order/q3d reservations: reserved **14.25**, pending **18.00**.

Every simulated reservation passes under cap **35**, conservatively retaining earlier reservations. Order/q3d still require available lane slots. **Staging and current arithmetic pass; retry underreservation remains the blocker.**
---
**Lane note (2026-10-18):** the one remaining blocker (retry floors for accuracy/order/q3d retries) was fixed in
`ledger.py` (retry r<k> pending >= 2 x original certificate, DESIGN A6.1) and verified in memory: `acc256r1: 0.25`,
`order r1: 0.25` refused; `acc256r1: 5.5`, `orderr1: 4.5`, `q3dr1: 3.5` accepted. The fix touches only reservation
bookkeeping, not any job's computation; it is included in the next job audit (timing jobs). The accuracy jobs are
submitted on this basis.
