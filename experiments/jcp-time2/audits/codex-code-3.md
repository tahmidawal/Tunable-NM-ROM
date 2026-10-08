**Overall verdict for SUBMITTING THE SMOKE: NOT CLEAN** at `5c96b098a`.

1. **Item 5 — STILL-WRONG, partially repaired.** Exact G1a/timing inventories, mandatory cache keys, explicit times and hash comparisons are now checked. However, expected coverage still comes from `result.json['config']`, never authenticated against the staged configuration (`audit_t2.py:193–201,266`). Reducing that embedded configuration can reduce required evidence. Bind it to the pinned staged config.

2. **Item 7 — RESOLVED.** Each submission uploads into a unique staging directory; queue checking, final-directory exclusion, move and submission remain protected by the namespace lock.

3. **Item 18 — STILL-WRONG, partially repaired.** Checkout/certificate/blob checks now precede directory creation. But the current untracked `audits/codex-code-3.log` still makes staging fail. Also, config parsing and reference validation remain *after* `mkdir` (`stage.py:54–82`), so failures there still leave a poisoned attempt directory.

4. **Item 19 — machinery checks pass; submission gate remains blocked above.** In-memory generated `smk` sbatch, `submit.sh` and `collect.sh` pass `bash -n`. Confirmed GPU partition/A100-80G, absolute venv, f64/highest, GPU preflight, paralab paths, both settings, cases 0/2, 232 runs per case/setting and 82 timing records per setting. Collection verifies output checksums before remote deletion. `ALL-DONE` establishes execution completion only; the corrected independent audit must pass before either 1024 job. That prerequisite is procedural—`submit.sh` does not enforce it.

5. **New fields — consistent for `smk`.** `A_sha=[ha1,ha2]` and `A_expected_sha` match the auditor’s schema. Both use identical `output_times=[.05*j for j in range(6)]`, consistent with the solver schedule; the auditor also checks step counts.

6. **Items 12–17 — deferral accepted.** They gate reporting and scientific interpretation, not smoke execution. Fix them before reporting results or making selection/order claims.

No files modified or jobs submitted.