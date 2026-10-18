**DO-NOT-SUBMIT.** Round-1 input and staging findings remain correct. Diagnostic storage is substantially fixed, but the verifier still mishandles legitimate failed rungs, and the resource estimate does not explicitly cover all check-rule work.

1. **CORRECT — configuration.** All 19 groups / 33 arm instances are present: four head sizes with three seeds and `min_pass=2`; fixed-M=128 heads for K=4,8,16; incumbent; eight linear rungs; fixed-M linear r=4,8,16. K=32/r=32 reuse follows A7. Ladder, selectable maximum 192, sub16/dev6 populations, and BE at dt=0.005 match the amended design.

2. **CORRECT — inputs.** Independently checked all 12 compact hashes against the accepted train result. All compact heads and the incumbent loaded through `hcommon.Head`, including source-checkpoint checks. Dimensions, 8,192 candidate codes, and 200,000 training steps agree. Recomputed training acceptance passes.

3. **CORRECT — staging and sbatch.** All **140 manifest entries** and **136 provenance entries** verify against committed blobs at `05e2f1433fc95b7794293ec97b5423ca6f7e19a7`. GPU partition, A100-80G, preflight exit 42, absolute venv, x64, highest precision, and namespace-local outputs are present. Sbatch, reservation, and certificate agree on **7 hours**. Production release is recorded.

4. **NEEDS-RESTATEMENT — resource coverage.** Revised arithmetic gives **4.5975 hours**, with a rounded 1.5× reservation of 7 hours. However, [resources.py:66](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/resources.py:66) scales by successors ending at **256²**, while the final selectable rung can require **320² check solves**. The added dimension/iteration allowance is a blanket 1.5×, without measured justification for r=384/M=1536. This remains a heuristic full-ladder estimate, not demonstrated worst-case coverage. Memory remains plausible, but production’s largest tables and three-head groups lack direct peak measurements.

5. **WRONG — complete independent freeze verification.** Round-1 missing diagnostics are now saved, and `analyze.rules()` gates freezing on structural and numerical verification. Remaining defects in [hscore.py:214](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/hscore.py:214):

   - Validity recomputation omits the saved `w0` finiteness check required by `P.run_valid`.
   - Nonfinite distances become JSON `null`; `np.allclose(d, a['dist'])` then raises `TypeError`. I reproduced that comparison read-only. An invalid coarse rung should count as failure, not prevent freezing a later passing rung.
   - Nonfinite rho likewise becomes `null`, which is always reported as disagreement. Additionally, the verifier excludes NaNs while the driver treats any nonfinite rho as failure.
   - The header still advertises nonexistent `hscore.rule_check`; the implemented function is `rule_verify`.

   The accepted smoke predates these storage/verifier changes and does not validate this new path.

6. **NEEDS-RESTATEMENT — recovery.** Evidence is now saved after every rung, improving on round 1. Writes still overwrite NPZ and JSON directly: interruption during writing can damage the current archive or result index. There is no resume path. This is a recovery limitation, not evidence of silently accepted incomplete results.

Required fixes:

- Make verifier validity and nonfinite handling match the driver, accepting correctly recorded failed rungs. Exercise passing, failed, and exhausted trails through the new archive/verifier path.
- Explicitly account for q++ work through 320² and justify dimension/iteration allowances; regenerate and restage the certificate and affected files.
- Correct the stale verifier reference. Atomic checkpoint replacement is recommended.

No files were changed, no generators or jobs were run, and no lab-log entry was written.