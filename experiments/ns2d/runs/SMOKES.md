# ns2d local smokes (GB10, through the shared slot helper)

Sub-minute-class runs used to shake out the drivers before each cluster job. None is a result;
every one is `SMOKE=1`, which forbids `complete=true` in its JSON.

| smoke | configuration | outcome |
|---|---|---|
| baseline (rule 7) | reproduce the audited Stokes S-FOM numbers from the parent lane | worst relative deviation 2.43e-15 (bar 1e-9) — `runs/BASELINE-SMOKE.txt` |
| phase 1 | 16²/32², T=0.2, 4+4+4 trajectories | found the biased mesh-order estimator and the independent-Jacobian sign; all other gates passed |
| phase 2 | K=4, R=16, 32², 300 steps | full path exercised; H-ORACLE fails as it must at 300 training steps (oracle = POD-K to 4 digits) |
| phase 3 (direct tensor) | K=4/R=16 checkpoint, q ∈ {0,2,4} | all machinery gates passed; tensor build 3.6 s; audit ALL_MATCH on 54 checks |
| phase 3 (FFT tensor, §A3) | same | R-TFFT 4.37e-16 vs the direct build, R-TQ 8.81e-16, R-TB exactly 0, build 0.90 s |

The phase-3 smokes report `R-LADDER: FAIL` (monotone = yes, gain 1.25 < the pre-registered 2.0):
that is the verdict gate working on a deliberately untrained checkpoint, not a lane result.

## After §A4 (2026-09-17, 11:2x EDT) — patched drivers

| smoke | configuration | outcome |
|---|---|---|
| phase 2 (§A4 driver) | K=4, R=16, `G_HIDDEN` default 2R=32, head 32×3, 32², 300 steps | B-RANKCAP PASS, B-ORTH rank 16/16, hash-or-value B-DATA in smoke mode, oracle formula-vs-field 1.3e-16; H-ORACLE fails as it must |
| phase 3 (§A4 driver) | the checkpoint above, q ∈ {0,2,4}, FOM ladder first, decomposition | R-TB 0, R-TFFT 6.8e-16, R-TQ 6.8e-16, R-LIN 1.2e-15; FOM ladder lands before the ROM arms; decomposition formula-vs-field ≤ 2.5e-16; `audit_phase3.py` ALL_MATCH on 54 checks; R-LADDER FAIL (gain 1.04) on the untrained checkpoint |
