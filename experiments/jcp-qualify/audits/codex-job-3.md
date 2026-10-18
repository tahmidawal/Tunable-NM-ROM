**FIX-FIRST** — audited HEAD `61bc6067d`. No files changed, project entry points executed, GPU use, or cluster access. Checks used read-only inspection and isolated functions in memory.

Paths below are relative to `experiments/jcp-qualify/`.

| Prior finding | Verdict | Evidence |
|---|---|---|
| 1. Fallback timing identity | **RESOLVED** | Staging now requires `key\|fb`, matching reservation enforcement (`cluster/stage.py:53`, `ledger.py:174`). |
| 2. Underpriced repeat obligation | **RESOLVED** | Repeat inventory inherits the original certificate floor (`ledger.py:121–124`). Reproduced: 0.25-hour repeat rejected; 3.25-hour repeat accepted. |
| 3. Analysis provenance releases allowance | **PARTIALLY** | Completed attempt is checked, but source only needs the substring `"lane"` (`ledger.py:98–100`). In-memory reproduction: all expected keys, `attempt=acc256`, `source="lane unrelated-attempt"`, and pass labels return `[]`, permitting allowance release. Bind source to the actual completed attempt. |
| 4. Conservative order/3D proxies | **RESOLVED by restatement** | A6 explicitly identifies these as estimates, not established bounds (`DESIGN.md:583–586`). Order’s claimed per-arm bound in `resources.py:75–78` remains unjustified and should be corrected. |
| Previous new bug: failed timing panel counted live | **RESOLVED** | Failed terminal states no longer bypass first-panel recertification (`ledger.py:197–212`). |

**A6: PARTIALLY implemented.** Estimate disclosure is explicit; recorded smoke evidence confirms **85/85** required checks and **96,592,342,016 bytes** pilot peak (`DESIGN.md:594–595`, `checks/smoke-accept.json:894`). Provenance enforcement remains incomplete.

**New amendment conflict:** A6 promises a doubled-limit retry under a new attempt (`DESIGN.md:586–588`), but the inventory whitelist rejects accuracy/order/q3d retry names and still requires the original mandatory name (`ledger.py:116–128`). Reproduced replacing failed `acc256` with `acc256r1=5.5`: **“unknown pending key acc256r1”; “missing mandatory acc256.”** Implement retry identity mapping before relying on this recovery policy.

All five staged trees match HEAD: **137/137 provenance entries and 141/141 manifest hashes each**. Commit records, config hashes, run lists, reservations, and sbatch limits agree.

| Job | Projection → limit, hours | ≥1.5× |
|---|---:|---|
| acc256 | 1.718 → 2.75 | Pass |
| acc1024 | 1.542 → 2.50 | Pass |
| acc4096 | 3.284 → 5.00 | Pass |
| order | 1.487 → 2.25 | Pass |
| q3d | 1.046 → 1.75 | Pass |

Installed inventory validation passes (`runs/GPU-LEDGER.json:41`):

- Spent **0.5039** + pending **32.25** = **32.7539 GPU-h**; headroom **2.2461** under cap 35.
- After three accuracy reservations: reserved **10.25**, pending **22.00**.
- After order/q3d reservations: reserved **14.25**, pending **18.00**.

These are conservative reservation totals before reconciliation; later submissions require available lane slots. Arithmetic and staging pass; provenance and A6 retry handling remain blockers.