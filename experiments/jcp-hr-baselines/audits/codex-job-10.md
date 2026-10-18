1. **CORRECT** — At commit `0cb714d186a870c6622fb19fd8c28e6353c4d1d9`, all 60 provenance-listed files match their blobs, allowing exactly the config’s injected `commit`. All manifest hashes and the bundle file set match. Runtime sources are identical to `s3d64a`. Differences: `COMMIT.txt`, `PROVENANCE.json`, `run.sbatch`, and the cell-specific config/frozen JSON/sidecar filenames and contents.

2. **CORRECT** — Independently recomputed from tuning evidence and remote reference-store eligibility:
   - Eligible screening cases: **0–5**; **164** configurations rechecked, **154** eligible.
   - Absolute front: **21**; alternating near-front additions: **24**; union: **46**, all matching exactly.
   - Eight P1, batch-1 X-specific fronts match exactly: sizes **22, 24, 21, 21, 21, 22, 21, 22** for R128/R256/R384/R384nom, BE then CNR.
   - Maximum distance-recomputation discrepancy: **4.55e-13**; all qualification decisions agree. All **26 ECSW members** pass artifact validation.
   - Frozen hash/sidecar match. Saturation fails at rank 384: **0.000895807 > 1.10 × 0.000688355**. The recorded tuning source installs the all-step basis before saving artifacts; hashed `Phi` and `Phi_lat` agree exactly.

3. **CORRECT** — Independently reconstructed A43.1 projection: **3.577421 h ≤ 3.933333 h usable**. Device model **5.556 GB ≤ 67.712 GB** admission limit; host model **2.433 GB ≤ 115.2 GB**.

4. **CORRECT** — `run.sbatch` specifies `gpu`, `a100-80G`, 128G, four hours matching config, `stageb.py`, paralab paths, the required venv, highest precision, x64, GPU preflight exit 42, and `JCPREF_STORE`.

5. **CORRECT** — Read-only remote re-verification matched **all 83 dependencies**, including `tune.json`, under `/cluster/tufts/paralab/tawal01/jcp-hrb/t2d256a/output`.

6. **WRONG** — Accounting passes: one unbound `s2d256a` reservation; `s3d64a` bound to **5917329**; earlier jobs reconciled consistently with `sacct`; **16.163 + 8.5 + 4 = 28.663 ≤ 30 GPU-h**. However, **DESIGN A43.3 explicitly requires submission after the 3D job is reconciled**. Job 5917329 is **RUNNING**, with `measured_gpu_hours: null`. Reserving its full wall time proves budget safety but does not satisfy that sequencing requirement.

**DO-NOT-SUBMIT-s2d256a**