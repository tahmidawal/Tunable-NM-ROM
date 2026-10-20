**Verdicts: `e13d` — FIX-FIRST. `e2t3d1` — FIX-FIRST.** The registered workloads are correct, but metadata, control enforcement, and the training handoff need fixes.

1. **CORRECT — E1 population and generated configuration.**  
   I compared `configs/e13d.json` with the in-memory result of `make_configs.e1_3d()`; they match exactly. It includes:
   - E1-3D-a: W256, lat8192 → lat16384, QN BDF2 0.025 k4.
   - E1-3D-b: W512, lat16384 → lat32768, same solver.
   - Both use 64 cases, draws 0–15, gl64 targets, gl56 checks and tighter solves on cases 0–7.
   - A11.2 supplements c/d have the correct settings, lat32768 → lat65536, and `supplementary: true`. Verdict rows preserve that flag without pooling them.

2. **CORRECT — E1 execution, fixture, and saved evidence.**  
   [e1run3d.py](experiments/jcp-robust/e1run3d.py:59) runs tight ADT on every draw/case, tighter ADT on the registered subset, draw-matched successors, per-step certificates, target/successor indicators, reversed-case replay, identity controls, and 16 Sobol diagnostics on E1-3D-b.  
   The badlat8192 fixture matches its pin. CPU establishment records six finite states, positive target norms, and \(\rho_{\rm tgt}=1.503\)–\(1.899\), satisfying admission. A1.3 correctly makes fixture agreement the required control outcome; bad-rule rollout failure is additional evidence.

3. **CORRECT — Main E1 scoring and reproduction paths are compatible.**  
   Saved coefficient keys and `cases|…` mappings match the analyzer’s production, successor, tight/tighter, target, and check lookups. **Use `e2score3.py`**, despite the stale `score3.py` instruction in the analyzer’s docstring: `check_scores()` explicitly requires bank-aware provenance.  
   I verified the local q3d3 reproduction archive’s checksum and both required arrays: `(64, 6, 256)` and `(64, 6, 512)`, with cases 0–63.

4. **WRONG — E1 final verdicts can ignore failed identity controls.**  
   [e1verdicts.py](experiments/jcp-robust/e1verdicts.py:74) checks C-E1.2 only from E1-3D-a. The driver runs that control separately for every configuration, but a failed C-E1.2 on b/c/d does not block its verdict.  
   **Fix:** enforce each configuration’s recorded identity-control outcome, with the registered blocking precedence.

5. **NEEDS-FIX — E1 resource inventory and estimate disclosure.**  
   [certify.py](experiments/jcp-robust/certify.py:61) hardcodes `W1024` into every E1 execution key, including E1-3D-d, whose actual bank is M2. The certificate also has `"estimates": []`, omitting the explicit A11.3 requirement to label W512/M2 rates as estimates.  
   **Fix:** derive bank identity from `settings`, disclose the A11.3 scaling, and regenerate inventory/certificate.

6. **CORRECT — E1 requested resources satisfy the current arithmetic.**  
   I independently reproduced **14,240 executions** and **0.272772 h**, matching the certificate’s 0.2728 h. One hour exceeds the staging requirement of \(1.5\times0.2728=0.4092\) h. Current code-tree, job-spec, and timing hashes match. Staging maps the request to A100 with the `a100-80G` constraint.  
   The W256 smoke completed in 77.8 s with a 13.13 GB device peak. Wider configurations remain extrapolations, not measured memory/time guarantees.

7. **CORRECT — Training recipe and checkpoint selection match j3b.**  
   I compared the training script and base config byte-for-byte against j3b’s staged copies: both match. Seed 1 changes **only**:
   - `model_seed`: 923715 → 924715.
   - `subset_seed`: 923002 → 924002.

   The unchanged trainer selects the checkpoint with lowest worst bank-validation projection error, rather than automatically selecting the final step. j3b selected step 11,000. Host QR and least-squares/SVD ordering remain unchanged.

8. **WRONG — Effective training memory-fraction setting differs from j3b.**  
   [stage.py](experiments/jcp-robust/cluster/stage.py:156) defaults to `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`. [run_seed.sh](experiments/jcp-robust/train3d/run_seed.sh:32) preserves that existing value; its `0.90` fallback never applies. j3b explicitly used **0.90**.  
   **Fix:** set `mem_fraction: "0.90"` in the training job/generator and regenerate its certificate. The JSON recipe is seed-only; the current effective execution environment is not the claimed incumbent environment.

9. **CORRECT, WITH UNRESOLVED PILOT RISK — 420G and four hours are defensible under A9.2.**  
   The actual host acceptance ceiling is **336 GiB MaxRSS**, because \(336\times1.25=420\). The recorded smk3 value, 176,766,044 KiB, is **168.58 GiB**, about **40.1%** of the request; A9’s “176.8 GB / 42%” mixes units. That leaves approximately **167.4 GiB to the acceptance ceiling**, but smk3 never reached bank fitting or ordering.

   j3b’s measured 6,660.18 s × 1.4 gives **2.590 h**. Four hours supplies about **1.410 h additional allowance** and satisfies the certificate’s \(1.5\times2.6=3.9\) h requirement. However, smk3’s unfinished host POD stage means this extrapolation does not establish completion.

   **A9.2 must remain binding:** timeout, host MaxRSS above 336 GiB, or device peak above **120 billion bytes** defers seeds 2–3 without increasing the limit. `SEED-COMPLETE` records successful execution, **not memory-gate acceptance**. The wrapper records process/device memory; authoritative Slurm MaxRSS still needs separate collection.

10. **NEEDS-FIX — Training artifacts suffice, but the later bank handoff is not connected end-to-end.**  
    j3b’s bank is **113,318,738 bytes**, so the analogous output belongs in the checksum-verified cluster archive. The trainer saves the bank, ordering, config, selection record, and completion/checksum evidence needed for reuse.

    However, [make_configs_e2_3d.py](experiments/jcp-robust/make_configs_e2_3d.py:31) expects `inputs/banks3d/s1/BANK.json`; no 3D registration writer is present. It places `cluster_inputs` inside evaluation **configs**, while [stage.py](experiments/jcp-robust/cluster/stage.py:133) reads them only from the **job spec**.  
    **Fix:** provide the registration step and explicitly propagate the archive path/hash into later e2q/e2f job specs. Local bank-aware scoring will also need a verified local bank copy; cluster staging alone does not supply that.

No files were changed, no jobs submitted, and no cluster or SSH commands run.