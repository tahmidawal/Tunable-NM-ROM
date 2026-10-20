**DO-NOT-SUBMIT.** One round-2 freeze-verification defect remains in a second checker.

1. **CORRECT — configuration.** All 19 groups / 33 arm instances match the amended design: four head sizes with three seeds and `min_pass=2`; fixed-M=128 groups for K=4,8,16; incumbent; eight linear rungs; and fixed-M linear r=4,8,16. K=32/r=32 reuse follows A7. The ladder, selectable maximum 192, sub16/dev6 populations, and BE at dt=0.005 agree.

2. **CORRECT — inputs.** All 12 compact-head SHA256 hashes match the train archive’s `compact_sha256` fields. All compact heads and the incumbent loaded successfully through `hcommon.Head`, including source-checkpoint/bank checks. Compact heads contain 8,192 candidate codes and record 200,000 training steps.

3. **CORRECT — staging and sbatch.** All **140 manifest entries** verify; all **136 provenance entries** match staged files, working sources, and committed blobs at `59bffa30debb51364d5e7c49d589d08386284870`. The script specifies GPU partition, A100-80G, absolute cluster venv, GPU preflight exit 42, x64, highest precision, and namespace-local outputs. Reservation, certificate, and sbatch agree on **11.25 hours**. Production release is recorded.

4. **NEEDS-RESTATEMENT — resource certainty.** The round-2 missing-320² coverage is fixed. Independently recomputed projection: **7.377805 hours**, rounded to **11.25 hours** after the 1.5× allowance. The dimension allowance now comes from smoke measurements: **1.627563×**, with L384 included in the comparison. This is a plausible full-ladder *estimate*, not a demonstrated worst-case runtime bound: short, compile-inclusive measurements do not bound trained-head iteration costs. Memory remains plausible; the smoke reported a **17.25 GB** device peak, but production’s largest tables and three-head groups lack direct peak measurements.

5. **WRONG — complete freeze verification.** The changes in `hscore.rule_verify` address all four specific round-2 code findings: `w0` finiteness, null distances, nonfinite rho handling, and the stale function reference.

   However, [smoke_accept.py:77](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/smoke_accept.py:77) still rejects `rollout_pass=True` with `rho_max=None`. The driver legitimately produces that combination when rho is nonfinite and JSON cleaning converts infinity to `null`.

   **Read-only reproduction:** using the actual checker functions with in-memory fixtures, `rule_verify` accepted both a nonfinite-rho failure followed by a passing rung and an exhausted trail. `rule_trail_ok` rejected both. Because `analyze.rules()` requires both checks, either outcome prevents freezing `RULES.json`. The saved evidence remains sufficient to repair and recompute offline.

6. **CORRECT, with recovery limitation.** Atomic replacement now protects the solve NPZ and result JSON individually, closing the direct-overwrite concern. They are not a transactional pair, and there is still no resume path. Interrupted work can require rerunning; this does not establish silently accepted incomplete results.

**Required fixes:** make the structural checker accept correctly recorded nonfinite-rho failures; exercise the combined structural/numerical gate on passing, failed-then-passing, and exhausted trails, including null distances/rho and nonfinite `w0`; commit and restage affected files.

No files were changed, no generators or jobs were run, and no lab-log entry was written.