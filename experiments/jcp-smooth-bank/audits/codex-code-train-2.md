**Mostly fixed, but completion enforcement still needs correction.** I generated the sbatch string in memory and passed it to `bash -n`: **PASS**. No repository files were modified; no training was run. Paths below are relative to `experiments/jcp-smooth-bank/`.

1. **NEEDS-RESTATEMENT — R2 verification.**  
   `tests/test_r2b.py:22–41` now checks exact final parameter/code parity across subsampled and full phases, and runs both λ values for 50 sampling steps. However, `smoothtrain.py:258` returns only `pts_idx[:8]`: the test checks **8 of 4096 indices**, not the complete draws.  
   **Fix:** return all indices when instrumenting the test; explicitly compare loss histories too. R2a is implemented correctly for `tr1` (`smoothtrain.py:287–293,422–428`): step 1 aborts; step 5000 only records disagreement. Both reference strings match the original log. There is no apparent code-path reason for a spurious abort, though exact formatted equality across JAX/hardware versions is not a mathematical guarantee. `.3e` checks four significant figures, stricter than A1’s three.

2. **CORRECT — stochastic Sobolev term under A2.1.**  
   `smoothtrain.py:247–262` draws 2048 states × 4096 points from the separate stream in both phases. The previous full-phase objection is withdrawn. Gradient construction and global normalization remain consistent (`:189–202,229–236`).  
   **Fix:** none numerically; update the stale module description at `:9`, which implies sharing the value points.

3. **CORRECT — D2/D4 indexing.**  
   `smoothtrain.py:147–155` uses the intended common support, correct signs and spacing, and no ghost values.  
   **Fix:** none. Interpret the saved statistic as A2’s **stencil sensitivity**, not a certified derivative-error bound.

4. **CORRECT — rotation and diagnostics.**  
   `smoothtrain.py:335–367` implements the registered LS/normalized-row SVD construction, rejects zero/nonfinite coefficient rows, and records conditioning, orthonormality and trust radii for ranks 128/384.  
   **Fix:** none in training; evaluation must enforce A2.4’s conditioning/orthonormality eligibility thresholds. `complete=true` does not establish eligibility.

5. **CORRECT — R0 fingerprint.**  
   `smoothtrain.py:105–107,409–413` fingerprints the complete generated dataset and now uses an explicit abort.  
   **Fix:** none.

6. **WRONG — collection still permits incomplete results; batch handling substantially fixed.**  
   `cluster/stage.py:84–100` preserves the allocation mask, preflights each selected device, stores every PID, waits individually, records exit codes and exits nonzero after any child failure.  
   But `cluster/collect.py:16,34–39` defaults to **no expected outputs**, merely prints task statuses, and does not automatically mark `--partial` archives incomplete. Also, checksum-generation failure at `stage.py:99` can be masked by the final successful `echo`, because `-e` is disabled.  
   **Fix:** derive mandatory outputs from the job specification; require all four `train.json` files to have `complete=true` and all expected task statuses to be zero; always label partial collections. Explicitly fail if checksum generation fails.

7. **CORRECT — atomic lane submission and provenance.**  
   `cluster/submit.sh:11–20` locks queue-check-through-submission, counts all queued lane states and creates attempts atomically. Staging retains committed-byte verification (`cluster/stage.py:33–44`).  
   **Fix:** none for the original race. **Current staging will refuse the worktree:** `git status` shows untracked `banktime.py`, caught by `stage.py:35`; resolve that before staging.

**Generated sbatch:** braces survive both Python and Bash interpolation correctly, including the inner Python f-string. `DEVS`, `PIDS`, `NAMES`, output-path concatenation and per-PID status capture are valid. All four allocated devices receive preflight checks. `OUTPUTS.sha256` covers output files, including task statuses, after all children finish; its unchecked failure is the remaining shell issue.

**Evaluation artifacts:** the requested training inputs are present: history/R2a/B statistics (`smoothtrain.py:314–320`), FD sensitivity (`:415`), checkpoint, `rotation.npz` containing T/L/C, trust radii, hashes and base’s `rotation_frozen.npz` (`:432–448`). These enable the registered evaluation, but evaluation must still produce both-stencil reference errors, population ladders, spectra, rollout checks and timing repetitions. Training history is persisted only after training returns, so a timeout loses structured intermediate history; logs remain.

**Seven hours:** plausible, not established. The extra two derivative contractions cost approximately **25% of the value reconstruction contractions**; reverse differentiation through both coordinate JVPs adds substantial bank-network work. A rough operation count suggests **1.7–2.1× during subsampling**, with much smaller relative overhead during full-value steps. Using the original log’s approximately 2.13 h subsampled + 0.83 h full-phase split gives roughly **4.5–5.5 h training**, plus generation, compilation and postprocessing. Seven hours offers credible margin; A2’s **1.5×** estimate is optimistic. Measure both phases before treating the wall limit as assured.