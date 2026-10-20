**Verdicts at `8fcc237d1`: `e13d` — SUBMIT; `e2t3d1` — SUBMIT as the A9.2 memory pilot.** No remaining blocking defect found for these two jobs.

1. **CORRECT — Per-configuration C-E1.2 enforcement.**  
   `e1verdicts.py` now blocks failed or missing identity controls on a/b/c/d, including supplementary configurations. In-memory fault tests confirmed this and verified that invalid-job precedence remains intact.

2. **CORRECT — E1 bank identities and estimate disclosure.**  
   `inv_e13d()` derives banks from settings. The regenerated inventory contains 10,968 W1024-bank executions and 3,272 M2-bank executions. The certificate explicitly labels the W512/M2_512 rates as unmeasured estimates at twice W256.

3. **CORRECT — Both certificates match the committed tree.**  
   Independently verified code-tree, job-spec, config, inventory and timing hashes. The certificates reference `3ac76535e`; their code-tree hash also matches current HEAD because `8fcc237d1` changes only excluded certificate/inventory files.

4. **CORRECT — Inventory and timing arithmetic.**  
   E1 regenerates exactly: **14,240 executions, all unique; 0.272772 h**, rounding to the certified 0.2728 h. W256 rates and setup reproduce from raw smoke rows; wider rates match A11.3. One hour exceeds the required 0.4092 h.  
   Training retains its declared **2.6 h** projection; four hours exceeds the required 3.9 h. Its inventory remains `{}`, consistent with the certifier’s shell-step handling; the training command, completion marker and projection basis reside in the hashed job spec.

5. **CORRECT — Training memory fraction and recipe.**  
   All three generated training specs match their committed JSON and explicitly set `"mem_fraction": "0.90"`, which staging exports and the wrapper preserves. Trainer provenance hashes pass. Seed 1 still changes only `model_seed` and `subset_seed`; checkpoint selection remains unchanged.

6. **CORRECT — A12.4 units and collection-time gate.**  
   Host acceptance is **MaxRSS ≤336 GiB**; device acceptance is **peak ≤120,000,000,000 bytes**. The corrected smoke value is **168.58 GiB**, approximately **40.1%** of 420 GiB.  
   This remains a **manual collection-time gate**, not an automated staging interlock. Record authoritative Slurm MaxRSS and device peak in the prescribed gate file before staging seeds 2–3. Timeout or either threshold breach defers those seeds. `SEED-COMPLETE` alone does not establish acceptance.

7. **CORRECT — Bank handoff is connected.**  
   Registration cross-checks the bank checksum against `LARGE.sha256`; its archive path matches the collector’s layout. Config generation supplies the registry and `cluster_inputs`; job generation propagates those inputs into the spec consumed by staging. The verified local-fetch destination matches bank-aware scoring. This verifies the code path, not a completed seed’s artifacts.

8. **CORRECT — Previously accepted behavior has not regressed.**  
   The E1 config still equals its generator output. Population, supplementary separation, driver execution, fixture, scoring/reproduction paths, training implementation and checkpoint selection are unchanged by these fixes. Continue using **`e2score3.py`** for bank-aware scoring. Wider E1 resource estimates and full training completion remain unproven until execution.

No files changed, scratch created, jobs submitted, or cluster/SSH commands run.