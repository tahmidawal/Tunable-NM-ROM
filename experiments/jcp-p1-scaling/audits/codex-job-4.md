Audited HEAD `86f7535a69d3`.

| Attempt | Verdict | Finding |
|---|---|---|
| `kern2d` | **SUBMIT** | P1a fixtures, frozen modes, all three panels and nominal rules match DESIGN. Output schema matches `analyze.kernels()`. |
| `kern3d` | **SUBMIT** | P1b fixtures, fixed `M=2052`/`gl24`, nominal shell completion and W1024 rule mapping match DESIGN. Output schema matches `analyze.kernels()`. |
| `p1e3d` | **FIX-FIRST** | Configuration matches, but A1.12 validity and reporting are incomplete. |

All three staging directories pass manifest checks; all **124 provenance files per attempt** match HEAD byte-for-byte. Batch scripts correctly request H200, isolated paralab directories, f64/highest precision and GPU preflight. Certificates match configs and driver invocations; allocations exceed 1.5× projections: **0.3015/0.447/0.7725 h**, respectively. These are estimates, not runtime bounds.

Memory is feasible: `gl64`, r=512 has **6.451 GB** of off-mesh tables; the simultaneously resident `gl56` comparator adds approximately **4.322 GB**. Nominal r=1024 tables occupy **3.222 GB**, plus **2.048 GB** for the full mesh bank. Construction intermediates add memory, but no H200 capacity blocker appears.

For `p1e3d`, fix:

- **Undetected damping saturation:** `p1three.py:245–246` relies on reason 3, but the fixed-sweep solver only flags nonfinite residuals there; saturation can remain reason 0. Persist and check saturation throughout the rollout.
- **Missing prescribed judgments:** distances and own-state ρ are saved correctly, but neither the driver nor `analyze.py` evaluates the **2.5e-4 / 0.116** bars. Analysis merely copies `p1e`. Add validity-aware failing/deployed/finer outcomes, preserving its descriptive status.

Submission must respect the four-job lane cap; the recorded ledger already has three active reservations. No files changed or jobs submitted.