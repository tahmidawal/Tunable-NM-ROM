**J4 — 4096²: SUBMIT.** No remaining blockers in this narrow audit.

- **Fixes:** Exhausted NNLS allowance preserves the reason and marks both κ configurations `UNRESOLVED`. Per-attempt allowances match A10.2.
- **Memory:** Prefix release precedes each refit construction; full-rank access clears it too. Construction retains Φ₃₈₄ without the cached prefix, contributions are blocked, and training states remain on the host.
- **Scope/time:** All **24 configurations** remain registered. Recomputed certificate: **2.127 hours**, below the **210-minute command timeout** and **215-minute allocation**. This remains an A9.2 planning projection, not a measured 4096² bound.
- **Staging:** All **144 manifest entries** verify; all **141 provenance files** match HEAD byte-for-byte. `COMMIT.txt` names parent `4855d220e`; HEAD `786c7f3d9` adds staging and ledger settlement without changing those sources.
- **Ledger:** Settled usage + active reservations + J4 totals **7.0728 / 8 GPU-hours**.

No files, jobs, or lab-log entries were changed.