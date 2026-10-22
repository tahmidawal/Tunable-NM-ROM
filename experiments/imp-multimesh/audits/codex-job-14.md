| Job | Verdict |
|---|---|
| J4 — 4096² | **FIX-FIRST** |

Two submission blockers remain:

- **Staging is incomplete.** `runs/j4` contains 29 files, all byte-identical to HEAD, but lacks `run.sbatch`, command scripts, manifest, provenance, reservation, commit marker, and required dependency trees. Complete staging and verify its manifest against HEAD.
- **The ledger still reserves J5’s full allocation.** Adding J4’s 215 minutes currently totals **9.079 / 8 GPU-hours**, so submission would be rejected. Settle completed J5 using confirmed accounting. Using its recorded elapsed time would reduce that total to approximately **6.773 hours**.

The requested code fixes pass this narrow audit:

- Small CPU probes confirm exhausted allowance preserves `job budget exhausted` and both κ configurations as `UNRESOLVED`, without `KeyError`.
- NNLS matches A10.2’s revised **per-attempt** allowance, calculated after contributions.
- Prefix release precedes each refit construction; full-rank access clears it too. At 4096², construction uses the **51.51 GB** full basis without the cached **34.34 GB** rank-256 prefix. Contributions are blocked, and training states remain on the host.
- Full scope remains **24 configurations**. Recomputed projection is **2.127 hours**, below the **3.5-hour command timeout** and intended **3.583-hour allocation**, leaving five minutes between those limits. The certificate’s config hash matches. This remains a planning projection under A9.2, not a measured 4096² runtime bound.

No files, jobs, or lab-log entries were changed.