**SUBMIT.** Both round-1 blockers are resolved.

| Check | Result |
|---|---|
| Staging integrity | PASS: all **173 manifest entries** verify; all **169 provenance files** match Git commit `e3be6fa9f9c8e548498d1687a210898f0a38250c`. |
| Config | PASS: staged config is byte-identical to committed/current `configs/q3r6.json`; SHA-256 matches the certificate: `17db85a8…e6295774`. |
| Sbatch | PASS: **01:45:00**, one H200, GPU partition, isolated paralab directory, f64/highest precision, GPU preflight, manifest verification. Shell syntax passes. |
| Certificate | PASS: independently reproduced **1.0219396 h** projection; 1.5× = **1.5329094 h**, covered by **1.75 h**. |
| Ledger | PASS: pending q3r6 **1.75 h**; `ledger.validate` returns no errors. Total commitments **53.7137/55 GPU-h**. |

Submission-time queue and reservation guards remain applicable. No files changed, jobs submitted, or LAB-LOG.md writes.