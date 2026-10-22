**WRONG — fix before submission.**

- **CORRECT:** `gpu` partition, one H200, no `preempt`; 24-minute wall equals the 0.4 GPU-h ceiling.
- **CORRECT:** Required cluster venv, `highest` precision and x64 are enabled.
- **CORRECT:** GPU preflight prints `jax_backend=` and exits 42 on CPU; shell propagates that failure.
- **CORRECT:** All 117 manifest entries match, including `run.sbatch` and `COMMIT.txt`.
- **CORRECT:** Explicit outputs, logs, temporary files and XDG cache target `/cluster/tufts/paralab/tawal01/imp-fronts/SM-r/`.
- **NEEDS-RESTATEMENT:** CUDA cache remains unredirected; “all writes under the job directory” is not enforced, despite Amendment 17.
- **CORRECT:** Configured solves cover FC1 dev 0–1, the 16 fresh dev/val cases, and only FC1|0 / FC2|0 references; no sealed/OOD evaluation.
- **NEEDS-RESTATEMENT:** Full cohort metadata, including test descriptors, is loaded and hashed. No test solves occur, but literal “no test-split access” is stronger than implemented.
- **CORRECT:** C-B1 exercises swapped hashes, config mismatch and a matched positive control; C-B6 uses the production certifier on a perturbed state.
- **WRONG:** `fpolicy.py:292` parses `FC1|0|cert|rule|factor` into `cid='0'`. If the configured tighter anchor runs, `solve()` accesses `self.U0['0']` and raises `KeyError`; preserve `FC1|0`.
- **WRONG:** Policy/reference failures are swallowed; failed evidence is ignored with `|| true`; `ALL-DONE` is unconditional. This violates smoke failure propagation.
- **CORRECT:** Downstream evidence independently rejects missing controls or an incomplete dry run; the unconditional marker does not itself certify production.
- **NEEDS-RESTATEMENT:** Policy cutoffs are 216 / 864 seconds from job start; reference cutoff is 1,296 seconds. Policy setup, compilation and certificates lack an outer timeout.
- **WRONG:** Policy timing rows exclude fresh-Jacobian certification, rule setup and control work; `resources.py` projects from those solve-only rows without accounting for those costs.
- **NEEDS-RESTATEMENT:** No explicit burn-in occurs in `fpolicy.py`; the burn in `r2acc.main()` is bypassed when calling `r2acc.run()`.
- **NEEDS-RESTATEMENT:** Reference timing schema matches the consumer, but collection must place it at `runs/SM-r/timing.json`; the staged output location is different. Missing reference smoke timings do not block `ref_item()`.
- **NEEDS-RESTATEMENT:** Completion is unsubstantiated: references receive only 432 seconds if ROM work reaches its cutoff, versus roughly 577 seconds for two historical mean-cost cases, before startup overhead. Earlier ROM completion could recover time; the hard wall limits spending.

Read-only audit completed; no changes, jobs, SSH, scratch or lab-log writes.