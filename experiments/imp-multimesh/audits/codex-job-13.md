| Job | Verdict |
|---|---|
| J1 — 256 | **SUBMIT** |
| J3 — 1024 | **SUBMIT** |

No blocking findings in this narrow audit.

- **Exhausted budget:** isolated CPU checks confirm both κ configurations are saved as `UNRESOLVED`, preserving `job budget exhausted`, without `KeyError`.
- **A10.2:** implementation matches the revised per-attempt allowance. The allowance is calculated after contributions and divided to accommodate the initial attempt plus retry.
- **A10.3:** prefix release precedes contribution construction; full-rank access also clears the cached prefix. Small lifetime checks pass.
- **Staging:** both manifests verify. All **141 provenance-listed files per job** match HEAD (`28d1396d3`). Recorded source commit `315dd2f3f` correctly identifies the pre-restaging source.
- **Certificates:** config hashes match. Projections **0.603 / 1.121 hours** fit allocations **1.417 / 1.917 hours**, respectively. S1c acceptance passes.
- **Ledger:** both full allocations plus recorded spending and J5’s full reservation total **7.4122 / 8 GPU-hours**. Submission remains subject to the existing concurrency gate.

J4 was excluded. No files, jobs, or lab-log entries were changed.