**Verdicts: e2t2d — FIX-FIRST. e2s0 — FIX-FIRST.** The release prerequisite blocks both; e2s0 also lacks the required staged replay-reference hashes.

1. **NEEDS-FIX — Both jobs: A8’s release condition is not yet satisfied.**  
   [DESIGN.md:793](experiments/jcp-robust/DESIGN.md) requires accepted smk3 evidence before 2D production. `checks/smoke-accept-smk3.json` is absent, and the local ledger records smk3 as `SUBMITTED`. Accepted smk2 does not satisfy this condition. Complete smk3 acceptance before submission.

2. **NEEDS-FIX — e2s0: freeze the replay reference hashes before staging.**  
   A3.6 explicitly requires the P1 archive SHA256 values in the staged config. `configs/e2s0.json` contains no replay-reference pins; `e1analyze.pinned_sha()` instead reads another worktree’s listing at analysis time. Record the six acc1024 reference identities/hashes in the staged configuration and make replay consume those pins. I verified that all six reference files currently exist and match P1’s listing.

3. **CORRECT — e2t2d: registered seed changes and fixed data.**  
   `train2d/run_seed.sh` passes Stage A `--seed s`, extraction `SEED0=0`, and head-fit `SEED0=s`. Stage A fixes its state selection to seed 0. Extraction retains the canonical 576 trajectories plus 4032 trajectories from seed 1000. All 17 vendored training files match their provenance hashes.

4. **CORRECT — Full recipe and stopping records.**  
   Production uses Stage A’s 300,000-step default, including 10,000 final full-point steps, and the registered Stage B full call: 200,000 steps, `ARMS=mid,wide,k32`, `TIME_CAP=1500`, and `EMIT=mid`. It records actual steps, device, elapsed time, and whether the mid head stopped early. A time-capped seed must retain the registered “recipe deviation” label.

5. **CORRECT — Deployment rotation and registration artifacts.**  
   The emitted dense mid checkpoint feeds head-based `make_rotation.py`; Stage A’s least-squares rotation is renamed diagnostic-only. Checkpoint/rotation hashes, inverse and orthogonality diagnostics, stage manifests, and the final output manifest are saved. These supply the files and hashes needed for later `inputs/banks/s<seed>/BANK.json` registration.

6. **CORRECT — Seed isolation and reasonable training resources.**  
   Each seed has its own output directory and subprocess with `CUDA_VISIBLE_DEVICES=0/1/2`. Staging enables f64/highest precision and checks every GPU backend. Extraction streams 64 trajectories at a time; it does not retain 4608 full trajectories. Principal allocations are approximately 7.88 GiB for Stage A snapshots, 1.59 GiB per extraction chunk, and 0.50 GiB per coefficient matrix. **480G is ample by this allocation assessment**, although no measured three-seed host-RSS peak is available. Six hours is exactly 1.5× the declared four-hour per-seed projection; the reservation is 18 GPU-hours.

7. **CORRECT — e2s0’s numerical selection contract.**  
   The config has all 38 cases, ranks 128/256/384, both registered schemes at factor 1, the seven candidates in increasing point-count order, and their correct successors—including terminal gauss256. Target/check are gauss384/gauss320; tighter and check populations are dev6. `select2d.py` requires both schemes to pass and implements the diagnostic fallback when the target is uncertified.

8. **CORRECT — Selection, anchor, and replay outputs are available.**  
   `r2acc.py` saves production/tight/tighter/successor coefficients, validity flags, certificates, indicators, target/check coefficients, identities, and case IDs. `select2d.py` reports agreement with deployed rules for G-E2.1. `e2analyze.g_e23()` explicitly reads all six required acc1024 candidate files. Full G-E2.3 additionally awaits the registered 256² verdict run; e2s0 alone cannot complete that gate.

9. **CORRECT — Selection time allowance and archive handling.**  
   Read-only recomputation gives **5304 executions**, a **0.4268-hour** resource projection, and a **0.6402-hour** minimum at the staging multiplier; the one-hour limit clears it. This is an estimate, not a measured full-ladder runtime. `cluster/collect.py` archives large outputs, compares source/archive SHA256 listings, verifies the downloaded smaller outputs, and rechecks the archive before cleanup. Its threshold is technically **100 MiB**, using `find -size +100M`.

No files changed, no scratch created, and no cluster or SSH commands ran. Certificate freshness was not treated as a finding.