Wave 6 passes at HEAD `646de21f8`.

- **Arithmetic:** independently recomputed **80.977639 GPU-h ≤ 87**, leaving **6.022361 h**. Ledger rounding is negligible.
- **Allowance:** nine retained cells include test-once funding. The 2.22 h/cell estimate matches measured W40_A2 reference + timing + controls costs to rounding. Runtime estimates remain empirical, not enforced ceilings.
- **Specifications:** both body hashes match RELEASE.json: **H200, 240G, 10:00:00**. Full cohorts, reference resolutions and controls are enabled; no smoke overrides.
- **Arms:** W08_A2 trains/runs the bank; W16_A1 skips it through the registered width predicate.
- **A16:** executable sources remain unchanged from code audit 14’s READY commit. Both A3 restarts completed with passing control registries, including G-HOIST.
- **Operations:** queue empty; 3.3 TB available; both destination directories absent. Submission and detached collectors are therefore planned, not yet verified running. Use `submit.sh` and launch one `wait_collect.sh` per job.

No files changed, lab log appended, or jobs submitted.

**SUBMIT**