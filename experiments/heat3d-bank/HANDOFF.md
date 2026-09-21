# HANDOFF — heat3d-bank lane (PAUSED 2026-09-21 by user instruction via coordinator)

Branch `exp/2026-09-21-heat3d-bank` (forked from `exp/2026-09-20-hires-heat` @ 4fb12a6d). Lane dir `experiments/heat3d-bank/`.
Cluster namespace `/cluster/tufts/paralab/tawal01/h3dbank_20260921/` — **never created; no GPU job submitted (0 of 5 used); nothing running.**
No lab-log entry appended (none required: no job running).

## State at pause
- Read: campaign protocol, repo CLAUDE.md, hires-heat HANDOFF/DESIGN (addenda 1-4)/SPEED-LOG/audit disposition, `train.py`, `core.py`, failed curves of 4071535/4072491, paper-h3d HANDOFF.
- Only local CPU (NumPy, no JAX, no GPU) diagnostics were run: `diagnostics/podfloor.py`, `diagnostics/podfloor2.py`
  (second one stopped at the pause; partial output in `diagnostics/podfloor2-partial-output.txt`). NOT results; orientation for DESIGN.md.
- DESIGN.md not yet written; no Codex audit yet.

## Local diagnostic findings (32^3 training grid, training-only POD, worst relative L2 projection)
POD from 512 training draws (seed 921000, 6 times), worst over cohort, all-times (= t=0, always the worst time):

| R | validation 921001 (16) | dev 920311 (16) | paper-h3d final 920399 (64) |
|---|---|---|---|
| 128 | 0.57 % | 0.67 % | 1.71 % |
| 256 | 0.09 % | 0.09 % | 0.42 % |
| 320 | 0.04 % | 0.04 % | 0.27 % |
| 384 | 0.02 % | 0.02 % | 0.16 % |

With 2048 training draws: R=256 -> 0.14 % final / 0.15 % on a new 256-draw validation cohort (seed 921777); R=320 -> 0.08 %/0.07 %.

**Disclosure (protocol):** these POD floors were computed on the paper-h3d FINAL cohort (seed 920399) as a representation diagnostic
(no ROM, no model, nothing trained or selected yet). To keep the final cohort clean for the verdict, all further choices must be made
on validation seed 921777 (256 draws), which tracks the final-cohort worst case closely (0.44 vs 0.42 % at R=256/512 draws). Record
this in DESIGN.md.

Conclusions for the design:
1. R=128 cannot reach 1 % on the final cohort even with the optimal linear subspace (POD 1.71 %); the paper-h3d bank (1.9 %) is near that optimum. Wider bank is required: R in {256, 320}.
2. The 16-case dev/validation cohorts under-estimate the 64-case final worst case by ~4x; training coverage matters: 2048 draws cut the R=256 floor 0.42 -> 0.14 %. Use >= 2048 training draws and a 256-draw validation cohort for the floor gate.
3. Data are exactly separable (amp * f1(x) f2(y) f3(z), semidiscrete propagation per axis), so fields at any grid are cheap to generate.

## Diagnosis of the previous 3D failures (from code + curves; to be confirmed by a local smoke)
- `train.py` is an auto-decoder (per-snapshot codes `eta` trained by Adam) with an exact coefficient refit every 10k/20k steps. At the refit the
  code Adam moments are zeroed but Adam's step count (bias correction) is not reset, so after each refit every sampled code row takes
  steps ~3-30x the learning rate (nu ~ 1e-3 g^2 while mu ~ 0.1 g). Loss rose between refits in both runs (4071535: 5e-3 -> 5e-2 -> 4e9;
  4072491: 1.6e-3 -> 2.3e-2 at lr 1e-3) and the bank's validation floor degraded with it (2.3 % -> 6 %) — code noise corrupts the bank.
- Capacity/spectrum: 32 random Fourier features at scale 1.5 in 3D with a 256-wide net asked to span 256 functions (hidden width == R, cond grows); the best floor seen was at step 20k.
- Planned fix (not implemented yet): no codes at all — variable-projection loss on the full 32^3 training grid, per-snapshot
  1 - ||L^{-1} G^T u||^2/||u||^2 with L = chol(G^T G) (f64), tail/worst-case weighting, weak whitening penalty on G^T G (span-invariant),
  optional POD-teacher pretraining (regress G onto top-R POD modes), hidden width 512 > R, more/structured Fourier features,
  >= 2048 training draws, pre-registered floor gate on the 256-draw validation cohort (e.g. <= 0.5 % at the training grid and at 64^3).

## Next step on resume
Write `experiments/heat3d-bank/DESIGN.md` (arms, floor gate, disclosure above, stop rules), Codex audit, local sub-minute smoke of the
varpro bank trainer on GB10 via jaxrun, then job 1 = bank+head training (H200/A100) with in-job floor gate; job 2 = 64^3/128^3(/256^3)
panel on dev then final cohort opened once.
