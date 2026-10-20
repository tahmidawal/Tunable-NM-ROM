**DO-NOT-SUBMIT.** Configuration and input provenance pass. The blocker is incomplete evidence for independently recomputing solve validity; the resource certificate also needs qualification.

1. **CORRECT — configuration.** All 19 groups / 33 arm instances are present:
   - Four head sizes, three seeds each, `min_pass=2`.
   - Fixed-M=128 head groups for K=4,8,16; K=32 reused as specified.
   - Incumbent and all eight linear rungs, `min_pass=1`.
   - Fixed-M linear groups for r=4,8,16.
   
   Ladder is `{16,24,32,48,64,96,128,160,192,256,320}`, selectable maximum 192, population sub16, check dev6. The driver uses BE with dt=0.005 and selects the first rung passing the amended predicates.

2. **CORRECT — trained heads.** All 12 compact archives’ SHA256 values match the train result’s `compact_sha256`; every recorded `compact_vs_emitted` is zero. All 12 and the incumbent loaded successfully through `hcommon.Head` in a read-only CPU check, including the source-checkpoint assertion. Dimensions, 8,192 candidate codes, and 200,000 training steps agree. Recomputed training acceptance passed.

3. **CORRECT — staging and sbatch.** GPU partition, A100-80G constraint, GPU preflight exiting 42, absolute cluster venv, x64, highest precision, namespace-local outputs/cache/tmp, and 03:30:00 limit are present. The limit matches the reservation and certificate. All **140 manifest entries** verified; all **136 provenance entries** match their committed blobs at `1352a19313b7047794090f856d902dc221393884`. Production release is recorded.

4. **NEEDS-RESTATEMENT — resources.** The certificate’s arithmetic is consistent: **2.2029 hours projected**, rounded 1.5× allowance gives **3.5 hours**. It counts all nine selectable rungs for every arm.

   However, [resources.py:62](experiments/jcp-heads/resources.py:62) scales only by the main rule’s point count. Successor/check work at 256²/320² is implicit in a smoke-derived average, and there is no explicit scaling for linear r=384/M=1536 or trained-head iteration counts. This is a plausible heuristic, **not demonstrated worst-case runtime coverage**.

   Memory appears plausible: the same-mesh smoke peaked at **17.25 GB**, with a **4.29 GB** bank, against a **78.29 GB** allocator limit. Production’s larger tables and three-head groups are not directly measured by that smoke; I found no clear inevitable OOM.

5. **WRONG — complete offline predicate recomputation.** Saved unknowns, norms, case identities, and head parameters support reconstructing distances and rho. But [hrule.py:91](experiments/jcp-heads/hrule.py:91) reduces each solve to `W`, fields, coefficients, and a **validity boolean**; [hrule.py:156](experiments/jcp-heads/hrule.py:156) archives only `keys`, `W`, `valid`, and `n0`.

   Consequently, the archive cannot independently reconstruct the validity checks involving damping-saturation exits, per-step nonfinite stationarity, or the head initial-fit exit reason. Those diagnostics are discarded.

   Additionally, the advertised `hscore.rule_check` does not exist. [analyze.py:74](experiments/jcp-heads/analyze.py:74) copies the selected rules after checking `complete`; it does not independently verify selection.

6. **NEEDS-RESTATEMENT — recovery.** Solve arrays and selection trails are persisted only after an entire group finishes. Timeout/OOM loses the current group’s cached evidence; earlier completed groups survive. There is no resume path. This makes an underestimated runtime costly, though it does not silently qualify an incomplete job.

Required fixes before submission:

- Persist sufficient per-solve diagnostics to reconstruct validity, including exit counts/reasons, step stationarity values, initial-fit status, and initial unknowns.
- Implement independent rule verification and require it to pass before freezing `RULES.json`.
- Reassess the certificate with explicit successor/check work and dimension-dependent allowances; regenerate and restage affected artifacts.

No files were changed, no generators or jobs were run, and no lab-log entry was written.