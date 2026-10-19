**Verdict: FIX-FIRST for all three jobs.** Two submission blockers remain. Audited through HEAD `7b7d8ce274`; the checkout advanced during the audit. No files changed, scratch created, jobs submitted, or SSH/cluster commands executed.

1. **NEEDS-FIX — BLOCKING: all resource certificates are stale against HEAD.**  
   Each certificate records code-tree hash `c4973543…`. At the final audited HEAD, `certify.code_tree_sha()` returns `b72fdb37…`. [stage.py:73](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/cluster/stage.py:73) therefore rejects every job with `code changed since the certificate`. Job-spec, config, inventory, and timing-file hashes **all match**. Regenerate certificates after the final staged-tree commit; do not bypass the assertion.

2. **NEEDS-FIX — BLOCKING: the current production-release condition is unmet locally.**  
   The checkout now includes **A8**, beyond the A1–A7 mentioned in the request. [DESIGN.md:793](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/DESIGN.md:793) requires accepted `smk3` evidence before these 2D submissions. `checks/smoke-accept-smk3.json` is absent; the local ledger still lists `smk3` as `SUBMITTED`. Accepted `smk1b` and code audit 12 do not satisfy this additional condition.

3. **CORRECT — registered primary population and solve coverage.**  
   All three configs equal the in-memory output of `make_configs.e1_2d(mesh)`: R128; draws `d0`–`d15`; `fib4181s<s>` with matching `fib6765s<s>`; **38 cases at every mesh**; four LSPG arms and three diagnostic GAL arms. Every draw/arm has production, tight, and successor runs on all38, plus tighter on dev6. Every arm has gauss384 targets on all38 and gauss320 checks on dev6. [make_configs.py:27](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_configs.py:27)

4. **CORRECT — controls, with a scope clarification.**  
   The CPU-established fixture is attached **only at 1024²**, with the correct SHA256 `777df398…`; its six CPU indicators are 2.458–3.170, and accepted `smk1b` records in-job agreement and firing. Sobol has 16 seeds, BE factor 1, all38, only at 1024². Identity rejection and permuted replay are configured at every mesh.  
   **The bad-rule rollouts themselves occur at all three meshes**, seven arms ×38 cases each. Only the fixture check is restricted to 1024². This agrees with A3.10’s per-mesh inventory; interpreting “1024² only” as applying to every bad-rule rollout would be incorrect.

5. **CORRECT — saved evidence supports the registered verdicts and controls.**  
   [r2acc.py:210](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/r2acc.py:210) saves target/check coefficients; the `R_*.npz` files save production/tight/tighter/successor coefficients, separate validity flags, case mappings, normalizers, per-step certificates, and both indicators. Check validity is retained in `result.json`; control results are also retained there. These match the readers in `e1analyze.py` and `e1verdicts.py`. No missing job-side output was established. Offline analysis still requires the pinned bank, E0 store, CPU establishment, and historical reproduction archives.

6. **CORRECT — inventory and numerical reservation checks.**  
   Independently regenerated inventories match exactly, without duplicate execution keys. Recomputed projections match the certificates:

   | Job | Executions | Projection | Required ≥1.5× | Limit | GPU / host RAM |
   |---|---:|---:|---:|---:|---|
   | e1m256 | 14,052 | 0.2810 h | 0.4215 h | 1 h | A100-80G / 200G |
   | e1m1024 | 14,698 | 0.3220 h | 0.4830 h | 1 h | A100-80G / 200G |
   | e1m4096 | 14,052 | 0.3196 h | 0.4794 h | 1 h | H200 / 240G |

   All timing-source files exist locally and match their recorded hashes. H200 is correctly selected for the approximately 68.7 GB full bank at 4096². These are runtime projections, **not measured production peak-memory guarantees**; the certificates contain no peak-memory bound.

7. **CORRECT — bank, rule names, mode policy, cohorts, and precision.**  
   Staged checkpoint and rotation hashes match accepted `smk1b` (`18f0266a…`, `51149166…`). The driver explicitly uses frozen 1024² modes with \(M=512\); the smoke mode hash matches the CPU fixture. Draw names use the vendored generators. Cohort construction is restricted to dev6/val32. Staging sets x64/highest and performs GPU preflight; the driver independently asserts them. All 249 staging inputs examined exist and equal their committed blobs.

8. **CORRECT — reservation and concurrency safeguards.**  
   Submission reserves rounded GPU-hours, enforces lane cap 3 and account cap 8, uses the shared campaign submission lock, and retains ambiguous reservations. The local ledger shows 0.5267 GPU-h spent plus 2 reserved; adding these three one-hour jobs remains within 75. Live queue state was not checked, as requested.

| Job | Verdict |
|---|---|
| `e1m256` | **FIX-FIRST** — findings 1–2 |
| `e1m1024` | **FIX-FIRST** — findings 1–2 |
| `e1m4096` | **FIX-FIRST** — findings 1–2 |