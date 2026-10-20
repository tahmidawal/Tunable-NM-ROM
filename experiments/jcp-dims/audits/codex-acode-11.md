FIX-FIRST

Audited `de7c459c3..6b3e5a122`.

| Priority | File:line | Finding |
|---|---|---|
| P1 | [analyze2d.py:134](experiments/jcp-dims/analyze2d.py#L134) | **Retry signature omits numerical dependencies.** In-memory provenance mutations to `sep_common.py`, `engines.py`, and `bkfast.py` leave the signature unchanged. These affect decoder features, initial conditions/cohorts, and bank projection. Include the transitive numerical dependencies; reject missing hashes. The checkpoint-hash fix passes. |
| P1 | [analyze2d.py:155](experiments/jcp-dims/analyze2d.py#L155) | **Configuration validation covers settings only.** Changing archived `config.mesh` from 1024 to 512 still admits all 168 `f2a` arms. Validate numerical job-level fields against `configs/<job>.json`, allowing only explicit retry-packaging differences. Per-setting mismatches correctly abort. |
| P1 | [analyze2d.py:36](experiments/jcp-dims/analyze2d.py#L36) | **Numerical changes can escape cache invalidation.** `_code_version()` excludes `qcore.py` and `hari_quadrature.py`, although cached independent ρ uses their quadrature generators. Changing `RHO_SUBSET_MAX` also leaves the version unchanged. Bind these dependencies and numerical globals. |
| P2 | [resources.py:81](experiments/jcp-dims/resources.py#L81) | **Certificates cannot regenerate after offload.** The remaining `f2br1/archive` directory selects production calibration, but its `result.json` is absent; both `f2kA` and `b2f` projections raise `FileNotFoundError`. Resolve retained slim metadata. Smoke-path lookup also blocks current 3D certificate recomputation. |

CPU-only verification with the required interpreter:

- All **208** slim files pass manifest checks; recorded raw hashes match archived checksum listings.
- All **208 cached configuration summaries match** `checks/analysis-2d.json`.
- One eight-rung Q2 ladder reproduces every qualification label. Floating-point distances differ by at most approximately **3.3e-19**; bitwise reproduction was not established. Raw archives are offloaded.
- No missing downstream `KEEP2D` key found on the checked paths.
- Retry ordering and older-attempt fallback pass.
- Both configuration splits preserve their original inventories. Ledger projection-floor checks pass.
- Reading retained slim metadata reproduces certificates: `f2kA` **11.2667 h**, `b2f` **1.9548 h**.

No files changed, jobs submitted, large arrays copied, or LAB-LOG written. Full analysis was not run.