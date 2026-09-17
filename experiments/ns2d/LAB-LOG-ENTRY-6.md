## 2026-09-17

### ns2d — user decision "keep pushing": DESIGN §A9 pre-registers a NEW Phase-2 cell; four jobs submitted in parallel (ns301 diagnosis, ns302 4× data, ns303 3-mode family, ns304 exploratory ladder on ns204); cap raised 8 → 12

Branch `exp/2026-09-17-ns2d`, namespace `/cluster/tufts/paralab/tawal01/ns_20260917/`, commit
`31e0846f` (DESIGN §A9 + drivers + smokes) and the scheduler records after it. The §A4/§A6/§A7
negative for the pre-registered heads stands as recorded; this is a new cell, not a rescue.

**Submitted (A100, 180 GB, `gpu` partition, one attempt dir each, squeue checked before and after each sbatch; all PENDING on priority behind three other lanes' jobs):**

| attempt | job | driver / env | walltime | question |
|---|---|---|---|---|
| ns301 | 3808493 | `ns2d_headfit.py` / `headfit-k16r256-ns301.env` | 5 h | head-only on the frozen ns203 bank: subsets 128/256/512 × {plain, reg (wd 1e-4 + dev-select early stop), reg_small (128×2)}; dev-report oracle median vs n, log-log slope; train-oracle beside it |
| ns302 | 3808495 | `ns2d_phase2.py` / `phase2-k16r256-x4-ns302.env` | 12 h | 4× data (512 gated + 1536 from seed 20260920), K=16/R=256, ns203 recipe with per-step batch 13312; H-ORACLE at 2.0, `passes_at_1p5` recorded |
| ns303 | 3808498 | `ns2d_phase2.py` / `phase2-k16r256-m3-ns303.env` | 8 h | 3-mode / 6-amplitude family (new hashes recorded), 512 traj, K=16, ns203 recipe; same gates |
| ns304 | 3808502 | `ns2d_phase3.py` / `phase3-k32r512-explore-ns304.env` | 24 h | exploratory ladder q ∈ {0,32,64,128,256,512} on the ns204 manifold at fixed M=2176, POD-LSPG at 32/512 and matched K+q, same-job FOM ladder, 3 timed reps, decomposition |

**Pre-registered readings** (DESIGN §A9): ns301 slope ≤ −0.25 = "needs data", ≥ −0.10 under every
regime = "head-limited"; regularisation "moves it" if ≥10 % better at n=512. Whichever of
ns302/ns303 passes H-ORACLE gets the reserved Phase-3 job. ns304's R-LADDER is reported but
labelled exploratory after a failed gate; it tests whether the correction rank closes the
0.12 → 0.012 gap to the bank floor.

**Mechanics added** (all smoked locally at 32², K=4, R=16, before submission): `NMODES`,
`TRAIN_EXTRA`, `RECORD_HASHES`, blocked POD (`POD_BIG`, verified vs the GPU path to 7e-15) in
phase 2; `M_FIXED` in phase 3 (one shared device copy of the (M,R,R) tensor); the new
head-fit driver; checkpoints of ns203/ns204 committed under `experiments/ns2d/checkpoints/`.

**Retracted / corrected.** Nothing new this entry. **Open.** All four jobs pending; jobs used 9
of 12. Expected wall: ns301 ≈ 1.5–2 h once running, ns302 ≈ 6 h, ns303 ≈ 4.5 h, ns304 ≈ 8–12 h.
