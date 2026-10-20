**SUBMIT — e2t0.** Findings 3, 4 and 9 are resolved.

- **3 — Reference evidence:** Actual `front()` probes returned A’s cost as **1.0** with valid evidence. Failed, missing or duplicate equivalence records, and failed invocation hashes, each returned **missing** for both selected and fixed rows.
- **4 — Inventory:** **1,566 unique entries, zero duplicates**: 720 certification runs, 414 split triplets, 414 fused triplets and 18 doubled-work repetitions. A is certified but excluded from subject timing.
- **9 — Time limit:** Write-intercepted `certify.py` execution with `checks/smoke-timings.json` produced **0.7764 h**, including its 1.25× margin. Staging requires **1.1646 h**; the requested **2 h passes**. A10.3 overhead applies only to `r2acc`, as intended.

Selection binding and committed file contents match. No additional submission blocker found within this audit. The certificate and inventory files still need routine generation before staging; neither currently exists.

No files changed, scratch written, jobs submitted, or SSH/cluster commands run.