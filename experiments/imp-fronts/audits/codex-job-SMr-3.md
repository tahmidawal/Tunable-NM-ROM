**NEEDS-RESTATEMENT — core execution fixes pass; reference completion and projection claims remain qualified.**

- **CORRECT:** `gpu` partition, one H200, no `preempt`; 24-minute wall enforces 0.4 GPU-h.
- **CORRECT:** Required cluster venv, x64 and `highest`; GPU preflight exits 42 on CPU and propagates failure.
- **CORRECT:** All 117 manifest entries verify; batch script passes shell syntax checking.
- **CORRECT:** Outputs, logs, temporary files, CUDA/JAX/XDG caches target the job namespace.
- **CORRECT:** Solves cover FC1 dev 0–1, the 16 fresh dev/val cases, and FC1|0 / FC2|0 references only.
- **NEEDS-RESTATEMENT:** Full cohort metadata still loads and hashes test descriptors; “no test solves” holds, “no test-split access” does not.
- **CORRECT:** Tighter-anchor parsing now preserves `FC1|0` using `rsplit('|', 3)`.
- **CORRECT:** Policy exceptions and failed evidence produce nonzero job status; C-B1/C-B6 remain evidence gates.
- **NEEDS-RESTATEMENT:** Reference exit 124/137 becomes `INCOMPLETE` without failing the job; `ALL-DONE` can therefore coexist with unfinished references.
- **CORRECT:** Outer policy timeouts now cover setup, compilation and certification through the 1,296-second cutoff, with 30-second kill grace.
- **CORRECT:** Policy projection now uses whole-driver wall time, including setup and fresh-Jacobian certification.
- **NEEDS-RESTATEMENT:** `resources.py` drops Amendment 12’s separate reachable-shape compilation allowance; update the design or restore that term.
- **CORRECT:** Explicit initial GPU burn-in now executes before policy measurements.
- **CORRECT:** Generated `output/timing.json` matches the consumer’s default collection path and schema.
- **NEEDS-RESTATEMENT:** Missing/incomplete reference timings are flagged but still permit historical-only certification; completed partial timings are discarded.
- **NEEDS-RESTATEMENT:** Two-reference completion remains unsubstantiated: 432 seconds after the ROM cutoff versus approximately 577 historical mean-cost seconds, before startup.

Read-only audit; no file changes, jobs, SSH, scratch or lab-log writes.