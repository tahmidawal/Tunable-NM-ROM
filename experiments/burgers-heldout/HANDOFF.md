# HANDOFF — burgers-heldout (PAUSED 2026-09-21 ~11:10 EDT by user instruction via coordinator)

**State: PAUSED during reading. 0 / 5 GPU jobs used. Nothing submitted, nothing running.**
Cluster namespace `/cluster/tufts/paralab/tawal01/bheld_20260921/` not created. No DESIGN.md yet, no code, no
checkpoints copied, no lab-log entry (nothing ran). Worktree `worktrees/2026-09-21-burgers-heldout`, branch
`exp/2026-09-21-burgers-heldout`, forked from `exp/2026-09-20-hires-burgers` @ 0ab60014.

## Goal (from the lane brief)
New frozen Burgers 2D model = mesh-transferable learned bank from bank-floor (`cat1024`, fallback `cat2048`) + a
nonlinear head fitted with the INCUMBENT recipe + correction directions from training residuals + certified EQ rule
(held-out rho <= 0.116 on reached states; lattice rules as in hires-burgers) → accurate setting with <= 1 % worst
evolved error on hold64 = params_draw(20260916,64), faster than named Newton–BiCGStab FOM at 2048² and/or 4096².
Choose settings on dev6 only (hires-burgers DESIGN addendum A1); report hold64. Budget <= 5 GPU jobs, <= 2 running.

## What reading established (use this; do not re-derive)
1. **Incumbent head recipe is exactly reproducible from committed code**: `worktrees/2026-09-16-b-speed/experiments/
   separable-decoder/cluster/run_dn256b.sbatch` = `sep_coeff_extract.py` (N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5
   SEED0=0 LOOSE=1 EXTRA_SEED=1000 EXTRA_TRAJ=4032, CKPT = r3 bank `sep_burgers_r3_N256_K16_R512.pkl`) then
   `sep_hfit_run.py` (ARMS=mid → hidden 512 x 2, STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 SEED0=0, EMIT=mid).
   Training is in coefficient space via the exact Gram identity (*), so it is cheap. The b-head-train lane's failed
   reproduction used a DIFFERENT pipeline (235008 stride-1 codes vs the incumbent's 131072 picked states) — so the
   fidelity gate should rerun the incumbent's OWN scripts, and compare `mid` refit against arm `base` (incumbent as-is)
   in the same job. Proposed pre-registered tolerance: refit held-out/fresh-test oracle mean within 1.15x of `base`.
2. **Bank format mismatch**: bank-floor `cat1024.pkl` = `dict(blocks=[block, block], ...)` (bf_core.bank_block:
   B, g, out_scale per block; features = concat over blocks). Incumbent code (sep_common.SeparableDecoder,
   head-ablation arms.CoordBank, hops.build_bank) assumes ONE block. Either add multi-block support or merge the two
   blocks into one block-diagonal MLP (needs a feature-parity gate <= 1e-12). cat banks were trained on the first
   1024 trajectories of the incumbent draw (`C.incumbent_draw()[:1024]`, stride 2) — disjoint from dev6/hold64
   (asserted in bf_rep.py); still rerun `reports/checks/2026-09-21-burgers-train-eval-overlap.py`.
   The inc block inside cat1024 was fine-tuned (lr_warm 1e-4) → incumbent head does NOT transplant (defect 1.31 %).
3. **Memory at 4096²**: full-grid bank G is 64 GiB at R=512 f64; R=1024 → 128 GiB f64 (does not fit an H200 beside
   the FOM). Options: R=1024 at 2048² only (34 GB), or a labelled f32 decode-bank arm at 4096² (parity gate needed).
   R=2048 only fits at 2048². Learned-bank mesh transfer above R=512 is UNTESTED (bank-floor skipped the fine-mesh pass).
4. **Directions C** = qtd02's `directions_old.npz` (b-panel/inputs/directions_qtd02.npz, SHA 79d79458…), generator
   not yet located: next is `b-panel/inputs/make_provenance.py` and the qtd02 job (3757505) code, to rebuild C on the
   new bank+head from training residuals.
5. hires.py loads ckpt params (h, h_lin, B, g, out_scale) + Z_tr; population/certification states come from
   params_draw(0,128) trajectories 0–15 at mesh 512 via the dense topfix query; rules lat64 need no fit.

## Next step on resume
Locate the C (directions) generator; write `DESIGN.md` (fidelity gate + tolerance, arms, memory plan, stop rules);
Codex audit; then job 1 = training (fidelity gate + cat1024 head fit + directions), job 2/3 = 2048²/4096² dev6+hold64.
