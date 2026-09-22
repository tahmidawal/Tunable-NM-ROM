1. **PASS — trimming is safe.** `panel.py` is byte-identical to the specified source. Evaluating its configuration-only functions produces **21 subjects**, with no missing priority names, fidelity subjects or fast-parity partner. Rule loading iterates `eq_q`, so unused dictionary entries are harmless; all four active rule files have matching hashes and correct \(q,M\). Evidence: `panel.py:111–182, 408–433, 809–844`. **Separate prerequisite:** `checks/comparators.json` is absent; the default audit will fail all four fidelity gates until supplied (`audit_panel.py:358–365`).

2. **PARTIAL — separate rows work; integrity coverage is incomplete.** Each explicitly listed operator becomes its own row; default and explicit single-arm calls still work (`audit_panel.py:174–206, 241–247`). However:
   - The unchanged repetition gate checks **JAX invocations only**, before operators are added. Operator `zip()` silently truncates unequal timing arrays; neither five repetitions nor all six cases is enforced (`118–123, 183–190`).
   - The hash gate correctly checks saved field bytes. The \(t_0\) gate merely trusts a recorded Boolean, rather than comparing against this panel’s supplied input (`191–195`).
   - All operators are mislabelled `family='fno'`; missing arms are nonblocking. All nine names must be supplied explicitly.

3. **PASS for staged runtime files; one unstaged external dependency.** All **37 listed files** exist and match HEAD, including the decoder, library modules and every configured rule; all **nine operator checkpoint hashes match**. Import closure is covered, assuming the cluster venv provides the third-party packages. Checkpoint mismatch raises during staging; `sha256sum -c` under `set -euo pipefail` aborts the generated batch script (`stage.py:78–98, 136–151`). **But** the training index is an external absolute path, neither staged nor hashed; if absent, the training-overlap check silently disappears (`stage.py:44; panel.py:299–306`).

4. **FAIL — section 4 does not match the reported timing protocol.**
   - Operator timing does use **5 repetitions / 5 burn-in calls**, CUDA synchronization and retained arrays (`stage.py:60–61, 118; fno_panel.py:50–61`).
   - Audit table costs are **pooled medians**, not medians of case medians (`audit_panel.py:248–249, 308–326`).
   - JAX “complete” includes **input allocation/upload plus query and output download**; operators use query plus separately timed download. These scopes differ (`panel.py:737–745; audit_panel.py:188`).
   - Five forward calls do not establish thermal/clock stability. Fixed sequential operator order and host-side case processing/compression can leave clock-ramp or drift contamination (`fno_panel.py:90–108`).
   - Accuracy comes from an **extra untimed query**, not a retained timed invocation; operator repetition-output equality is unchecked (`99–103`).

5. **FAIL — plausible-looking “admissible” rows can escape the checks.** CPU fallback is explicitly blocked in both frameworks. Normal execution shares one allocation and generated cohort, and scoring correctly uses this panel’s `fft_tight` and refined references. However, operator metadata contains no verified job/GPU identity or cohort binding; the audit labels same-allocation timing true from descriptive strings. Stale operator files from another job could pass, as could an incomplete cohort indexed by matching case numbers (`audit_panel.py:183–197`). Operator/FOM rows are unconditionally `admissible=True`, even when operator gates fail (`280–285`). A failed `fft_tight` convergence gate does **not** abort the driver, and audit failures do not produce a nonzero exit (`panel.py:847–855; audit_panel.py:451–459`).

6. **FAIL — several claims or criteria need tightening.**
   - §8.2 compares the FNO “row” across allocations “by repetition spread”: this invites forbidden timing comparison and specifies no spread statistic.
   - §6 does not fix which error metric selects the FOM; §7 does not name the qualifying accurate arm or timing scope. The audited code does not implement that comparator-selection rule.
   - §8.1 supplies no numerical disagreement tolerance; §8.3 treats ordering changes across different cohorts as disagreement despite the cohort caveat.
   - §10’s blanket **f64** claim conflicts with the explicitly f32 U-Net/Transolver networks.
   - “Nothing is trained” should distinguish frozen neural checkpoints from the POD basis fitted in-job (`panel.py:317–323`).
   - “Three previously untimed cells” is undefined against nine operator arms, and the stated binary deliverable is not enforced when arms are absent.

No files were written, no jobs were run, and no network was accessed.