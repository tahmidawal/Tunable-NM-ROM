1. **CORRECT — t2d256a bundle identity.** All 61 manifest entries match; no extra files. Provenance payloads match Git commit `13fbab1d645b2b9f376cf02634ea9d77c61f12c2`. Staged `tune.py`, `nnls.py`, `build.py`, and `pipeline.py` are byte-identical to job-audit-4’s t3d64b bundle.

2. **CORRECT — t2d256a batch script.** Requests `gpu`, one A100 with `a100-80G` constraint, `128G`, and `02:00:00`, matching config wall. Attempt-specific paralab paths, prescribed venv, `highest`, x64, GPU preflight exit 42, and `JCPREF_STORE` are present. Driver deadline is 116 minutes.

3. **CORRECT — t2d256a config.** Equals `configs/tune_2d256.json` plus recorded commit: saturation enabled; extension rank 512; ranks `[32,64,128,256,384]`; BE/CNR at 0.005; eight P1 comparators; search allocation **1.9797222222 h**; ECSW path budget **2700 s**; budget/wall **1.75/2 h**.

4. **CORRECT — t2d256a memory fit, by allocation analysis.** All-step snapshots contain 29,376 × 65,025 doubles: **15.2814 GB host**. Their Gram matrix is **6.9036 GB**. Snapshot transfers, eigensolver matrices/workspace, and retained basis data fit nominally within the 80 GB device and 128G host allocation. This is an analytical fit, not a measured peak or completion guarantee.

5. **CORRECT — s2d1024a eligibility and membership.** Independently checked store eligibility: all six screening cases qualify. Recomputed Q1/timing acceptance for **172** completed configurations from certificates and saved coefficients; decisions match, with maximum distance discrepancy **1.45519e-11**, below A41.1’s `1e-9`. Recomputed eligible set: **111**; absolute front: **17**; alternating near-front additions: **18**, exhausting candidates below cap 24.

6. **CORRECT — every X-specific front and exact union.** Using the median of six per-case `t_B/t_X` ratios, front sizes are:

   | Rank | BE | CNR |
   |---|---:|---:|
   | 128 | 18 | 19 |
   | 256 | 18 | 18 |
   | 384 | 17 | 17 |
   | 512 | 17 | 18 |

   Every front matches the freeze. Union membership matches **exactly: 35 members**, without missing or extra IDs.

7. **CORRECT — ECSW and replay dependencies.** All **18 ECSW members** pass online-node/weight, selected-checkpoint, terminal residual, applicable KKT, and quality-label validation. Required dependency inventory matches exactly: **72 artifacts**. All local hashes match; remote paralab checks also match **73/73**, including `tune.json`. Frozen JSON matches its `.sha256` and staged config pin.

8. **CORRECT — Stage-B resource projection.** Independently reproduced A41.2: **10,550.15492 s = 2.930599 h**, below **3.433333 h** usable driver time. Recomputed memory models match: **53.7476 GB device**, **17.2477 GB host**, below H200 admission **119.3424 GB** and host admission **180 GB**. Runtime remains an estimate.

9. **CORRECT — s2d1024a bundle, script, and config.** All **63 manifest entries** match, with no extras; provenance matches `0ce8cf761dd7578618840afa97dc9370422e0b17`. Script requests `gpu`, one H200, `200G`, **03:30:00**, and `stageb.py`. Config equals source plus commit, with correct frozen-file/hash, remote tuning directory, and **3.25/3.5 h** budget/wall. Precision, preflight, venv, and paralab settings are correct.

10. **CORRECT — audit-27 source continuity, with explicit differences.** `stageb.py`, `analyze.py`, `stats.py`, `score.py`, and `cluster/stage.py` are unchanged from `f59050a68`. Differences are:
    - `ours.py`: three lattice methods now explicitly use `.copy(order="C")`, satisfying audit 27’s restatement.
    - `freeze_stageb.py`: A41.1 distance-agreement handling/tolerance and A41.2 calibrated time projection.
    - Bundled construction code: audited NNLS blocked deletion/deadlines and forwarding changes in `build.py`, `pipeline.py`, `tune.py`, plus regression-test changes. These construction sources match t3d64b’s audited bundle.

11. **CORRECT — ledger and sequencing.** Each requested attempt has exactly one unbound reservation: **2 h** and **3.5 h**. Measured expenditure plus reservations totals **21.7053 ≤ 30 GPU-h**, including t3d64b’s **12 h**. A41.3 requires t2d256a first, then s2d1024a, respecting the two-job lane cap.

No files changed, scratch created, jobs submitted, or GPU work performed.

**SUBMIT-t2d256a**

**SUBMIT-s2d1024a** — in the registered sequence.