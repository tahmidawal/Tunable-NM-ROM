1. **CORRECT** — Commit `8ad7e649f4a308e03e1c122d3bc80ebfe748b1d3`: all 60 staged source/assets match Git blobs, allowing only the specified config `commit` injection. All 63 manifest entries verify; no extra files except the manifest itself.

2. **WRONG** — `stageb.py`, `ours.py`, and `analyze.py` match `s2d1024b`; `freeze_stageb.py` differs. Its sole change replaces `block × ncase/6` with `block × (ncase + 2 + ncase × n_members)/32`, plus explanatory comments. This matches your requested correction, but falsifies source equality. Freeze/analyze are offline sources, absent from both runtime bundles. **A43.1 is also absent from DESIGN.md at the recorded commit.**

3. **CORRECT** — Recomputed from tuning certificates, recorded distances, case errors and warm timings: **836 eligible**, **26 absolute-front**, **24 alternating near-front additions**, **53 union members**. Every X-specific front matches exactly; front sizes are **26/27/28/28/29/28/28/25/25**. Admission is **five diagnostic batch-1 plus four batch-2 X**. All six screening references are store-eligible. Frozen-file and tuning-JSON hashes match. Raw coefficient-distance recomputation was not repeated.

4. **CORRECT** — Recomputed projection: **7.822774 h ≤ 8.433333 h usable**. Memory model: **47.775 GB device / 14.223 GB host**, below effective **119.342 / 180 GB** limits.

5. **CORRECT** — `run.sbatch` satisfies all listed resource, driver, path, precision, preflight and reference-store requirements; wall time matches config.

6. **CORRECT** — Independently reverified **77/77 remote dependency hashes**, zero mismatches.

7. **CORRECT** — Exactly one unbound reservation, `s3d64a`; all eight preceding jobs have reconciled measurements. **16.163 + 8.5 = 24.663 GPU-h ≤ 30**.

**DO-NOT-SUBMIT-s3d64a**