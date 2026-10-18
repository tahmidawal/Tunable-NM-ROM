**Overall: SUBMIT** for re-staged smk3 at `bfb098c54351118db6e4a1c73e85941f387cbe19`. No change in `b29e8be..HEAD` breaks smk3 or invalidates round 2’s submission verdict.

| Item | Verdict | Evidence |
|---|---|---|
| A6.1 / `dcert.gcert` | **PASS** | Implements the amended raw-residual floor; default scale preserves the previous behavior. |
| Call sites | **PASS** | `dacc`, `dtime`, `d3run`, and `dscore` supply the appropriate square-LSPG scalar or GAL per-step scale. Other certificate thresholds remain unchanged. |
| Offline smoke G-cert | **PASS, minor caveat** | Read-only execution of smk2 timing acceptance produced 29 checks with no failures. The changed logic applies only to smk2; smk3’s mandatory G-cert checks remain intact. |
| Tests | **PASS** | All eight certificate tests passed, including acceptance of round-off and rejection of a larger mismatch. Changed Python files parse successfully. |
| Other changes | **PASS** | A6.2 documents an offline ρ-check subset; it changes no smk3 execution. Added evidence, audit text, ledger accounting, and unused-variable removal introduce no submission blocker. |
| Staged provenance / manifest | **PASS** | All 160 provenance entries match HEAD byte-for-byte. Manifest verification passes; `COMMIT.txt` and reservation agree with HEAD. |
| `run.sbatch` | **PASS** | Shell syntax passes. H200, `gpu` partition, two-hour limit, f64/highest precision, GPU preflight, manifest verification, and dedicated paralab paths remain correct. |
| Reservation / budget | **PASS** | Two-hour reservation matches sbatch. Recorded spent time + smk3 + pending core totals **26.6747 / 55 GPU-h**. |

Minor caveat: offline smk2 timing acceptance now checks GAL raw agreement without separately retaining the prior η-agreement check. I independently verified that both archived GAL timing subjects also pass that prior check, so this does **not** alter their acceptance or block smk3.

This is submission readiness, not numerical acceptance. No files changed, jobs submitted, GPU computations run, or lab-log entries written.