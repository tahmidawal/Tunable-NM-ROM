**SUBMIT — technical job audit passes at `385de12b217dc80ea53c2a5804a4315eea0102fa`.** This does not override the separate coordinator release requirement for production.

Round-1’s resource blocker is resolved. The missing probe ratio, incorrect Q-repro band, and missing identity acceptance checks are fixed. The compact-check and interruption caveats remain as described below.

1. **CORRECT — configuration.** All twelve `(K, seed)` combinations use recipe `mid`, 200,000 steps, with the identity arm and K16/seed0 2,000-step sensitivity probe. Initialization, optimizer, normalization and diagnostics match the copied recipe. Output names match downstream `heads/head_mid_k{K}_s{s}.npz` paths after collection and installation.

2. **CORRECT — sbatch and provenance.** GPU partition, A100-80G, absolute venv, exit-42 GPU preflight, highest precision, x64, namespace outputs, 128 GB host memory and **03:45:00** limit are correct. All **122 provenance entries** match their committed blobs and staged bytes; the manifest passes. Memory is plausible from the real-size endpoint smoke, though no production peak-memory bound exists.

3. **CORRECT — input and compact safeguards.** The driver asserts the coefficient-file and incumbent SHA256s, checks all 4,608 training descriptors against dev6/val32, and checks pick/code alignment. Reloaded compact head weights must produce exactly identical outputs. Remaining qualification: the comparison uses in-memory emitted parameters, not a reloaded pickle, and does not independently validate every compact metadata field.

4. **CORRECT — resource certificate.** Independent recomputation from checksum-verified smoke outputs reproduces:
   
   | Quantity | Value |
   |---|---:|
   | Maximum smoke fit rate | 0.00355596995 s/step |
   | Projection | 2.48827255 h |
   | Rounded 1.5× reservation | 3.75 h |

   The unexplained `/2` is gone. This is a plausible conservative estimate, not a guaranteed runtime bound. Minor documentation correction: “Every constant is read from the smoke archives” is inaccurate—the setup allowance is hard-coded.

5. **CORRECT — training outputs.** The job preserves the training metrics, Q-repro oracle inputs and all twelve compact heads. Analysis now reports the probe/full ratio and uses the ROM ratio band **0.8–1.25**. ROM-dependent conclusions still require later qualified rollout results.

6. **NEEDS-RESTATEMENT — acceptance and recovery limits.** Identity acceptance now implements A1.7’s relative bars: reconstruction ≤ `1e-9`, oracle ≤ `1e-3`. However, `training_acceptance.accepted` is exported as a flag; it does not automatically prevent downstream conclusions. Require it to be true before promotion. Completed arms persist, but interrupted arms remain non-resumable; `result.json` updates are non-atomic. These are residual recovery limitations, not submission blockers with the corrected allowance.

**Required pre-submission fixes: none.** Before accepting results, verify training acceptance, completion and collected checksums, and install the compact heads before generating downstream configurations.

Read-only throughout: no generators, jobs, file changes, scratch copies or LAB-LOG writes.