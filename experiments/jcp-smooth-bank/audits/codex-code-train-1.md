**Do not launch unchanged.** The numerical core largely checks out; the full-phase Sobolev sampling and job-success handling do not. No repository files were modified.

Paths below are relative to `experiments/jcp-smooth-bank/`.

1. **WRONG — verification gates; value-only implementation is correct.**  
   `smoothtrain.py:66–117,172–312,388–408` matches the original data selection, initialization/key splits, schedule, AdamW mask, global normalization, EMA and reconstruction selection. Pick: 16,384 states, including 3,456 early states. Vendored hashes match A1. A tiny CPU comparison covering subsampled/full phases produced **zero parameter/code and reconstruction differences**.  
   However, `tests/test_r2b.py:13–38` runs only eight steps, never runs λ=0.1, and merely prints indices from five draws; it does **not** compare 50 steps across both paths. `smoothtrain.py:279–284` also never enforces R2a’s historical losses.  
   **Fix:** compare actual trainer indices for 50 steps with both λ values; assert loss/gradient parity against the reference; enforce base losses `2.402e+00` and `2.125e-03`. Historical correction: the original JSON specifies `time_cap=12600`, not zero, but records `time_capped=false`.

2. **WRONG — full-phase gradient sampling.**  
   `smoothtrain.py:120–131,188–201,228–235` correctly implements x-first neighbours, boundary sentinels, \(h=1/(n-1)\), and global squared-gradient normalization. `feat_grad:160–167` is per-row correct because features do not couple rows; independent Jacobian checks passed. The separate key stream is correct under A1.  
   But `smoothtrain.py:246–261` samples only 4,096 gradient points **even during the full phase**, violating `DESIGN.md:94`; A1 does not override that requirement.  
   **Fix:** use all points for the final phase, preferably chunked. Current direct gather is **0.25 GiB**, not an all-state expansion; resident U is **7.875 GiB**. Full-phase gradient gather would be **3.938 GiB**, before derivatives/activations. Measure peak memory after fixing it.

3. **CORRECT — D2/D4 indices.**  
   `smoothtrain.py:147–155`: padded slice `3:m+1` selects physical indices `2:n-2`, so fourth-order neighbours reach the walls but never ghost cells. Signs, axes and scaling match explicit stencil checks.  
   **Fix:** none required; preserve this independent stencil check as a regression test.

4. **NEEDS-RESTATEMENT — rotation algebra correct; diagnostics incomplete.**  
   `smoothtrain.py:324–340` correctly computes \(Y=UQ\), \(C=R_G^{-1}Y^\mathsf T\), normalized-row SVD, T and L, matching the requested substitution into `make_rotation.py`. The **7.875-GiB** host U remains resident; LS reconstruction is chunked into **0.492-GiB** arrays (`:327–331`), with additional temporaries—not a full reconstructed U.  
   **Fix:** add A1.2’s missing trust radii to `:344–351`: `0.01*max(norm(C @ L[:r].T - mean, axis=1))` for each deployed rank. Reject zero/nonfinite normalization rows.

5. **CORRECT — R0 fingerprint.**  
   `smoothtrain.py:105–107` sums every trajectory/time/full-grid value, including boundaries, before selection. `:394–397` uses the original JSON’s exact sums and \(10^{-9}\) relative tolerance.  
   **Fix:** no numerical correction; replace `assert` with an explicit failure so optimized Python cannot disable the gate.

6. **WRONG — batch failure handling and resource claims.**  
   `cluster/stage.py:53–54,87–91` loses child failures: the trailing `echo` succeeds and bare `wait` does not aggregate statuses. A shell reproduction returned batch status zero after a failed child. `collect.py:21–30` verifies checksums but not task completion; partial outputs can therefore pass collection.  
   `stage.py:53,82–83` also overwrites Slurm’s device mask with assumed indices 0–3; these need not identify the allocated devices.  
   **Fix:** select entries from the inherited allocation mask, preflight those exact devices, retain/wait every PID, persist exit codes, and require every expected `train.json` to have `complete=true`. Keep failure archives explicitly marked incomplete.  
   **320G** is ample for the identified host arrays, but peak RSS remains unmeasured. `jobs/tr1.json:5` requests **8 h / 32 GPU-h**, contradicting A1.9’s **5 h / 20 GPU-h**. Benchmark both training phases and postprocessing; amend the budget or meet it. The original 2.97 h does not establish Sobolev runtime.

7. **WRONG — lane cap is not atomic; staging provenance is sound.**  
   `cluster/stage.py:33–45,94–96` stages pinned-commit bytes, checks working copies and hashes the generated bundle correctly.  
   `cluster/submit.sh:10–17` permits two different attempts to pass the queue check concurrently. It also ignores active states such as `CONFIGURING` and `COMPLETING`.  
   **Fix:** serialize queue-check-through-submission with a namespace lock on shared cluster storage, count every queued lane job regardless of state, and create attempt directories atomically.