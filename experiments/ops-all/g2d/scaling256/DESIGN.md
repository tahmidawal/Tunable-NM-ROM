# g2d / scaling256 — can a neural operator reach < 1 % on Burgers 2D with more data? (256² only)

Coordinator task, user-approved, 2026-09-25 ~00:00 EDT. Written before any job of this study. Parent: `../DESIGN.md`.

## Question
At 256² the best operators in pl256 (job 4301382) reach 2.2–3.9 % worst on dev6. Their training fit is much better
than their validation error (fno-w96f32 train MSRE 6e-6 vs validation worst 5.4 %), with only 128 training
trajectories. Does 576 or 2048 trajectories close that gap?

## Design (fixed now)
- Networks: `fno-w96f32` (width 96, modes 32, 4 layers, float32) and `unet-b64` (base 64, float32), the exact
  configs of t256ff32 / t256ub64. AdamW, lr 1e-3, weight decay 1e-4, batch 8, config seed 20260914.
- N ∈ {128, 576, 2048} training trajectories, **prefix-preserving**:
  - case i uses seed `SeedSequence([20260914, 22, 1, i]).generate_state(1)` and descriptors `params_draw(seed, 1)`,
    exactly the generator of the pinned 128-case index (`neural-operator-burgers/data.py` @ 5169c095, `case_seed`).
  - Checked locally: seeds 0–127 equal the pinned index. The job re-asserts that seeds and descriptors equal the
    pinned index for i < 128, to 1e-12 relative.
  - Disjointness from dev6 and the 32 validation cases is asserted.
- Validation: the same 32 pinned validation trajectories, used for checkpoint selection only.
- Reference for every trajectory (train and validation): `fft_tight` at 256 intervals (Newton–BiCGStab, dt 0.005,
  ntol 1e-6, ltol 1e-8). That is the same-grid solver the panel grades against, generated in host memory
  (`scripts/train_scale.py`, derived from `../scripts/train_mem.py`). **Deviation, stated:** the pinned 128 files
  were a 4096² reference restricted to 256², while pl256's opt201 checkpoints and t256* used 256²-generated data as
  well (t256*: `train_mem.py`, fft_tight at 256). Here all three N use the fft_tight-at-256 data.
- Budget **7200 s** per network, one network per GPU (6 jobs, cap 8).
- Schedule family: ops/train.py's plateau rule (factor 0.5, min lr 1e-5) and early stopping. Both patiences are held
  constant in **optimisation steps**, as train.py's own data-ladder note requires: N=128 has 16 steps per epoch and
  uses the configs' 20 / 250 epochs. N=576 (72 steps per epoch) uses 5 / 56 epochs. N=2048 (256 steps per epoch)
  uses 2 / 16 epochs. The plateau rule adapts to run length by itself, so the longer budget needs no other change.
- Per network, recorded from the best-validation checkpoint:
  - on the training set, the 32 validation cases and dev6: worst / mean / median over cases of the per-case max over
    evolved times of ||u − u_ref|| / ||u_0|| (the paper's Burgers metric);
  - epochs, best epoch, stop reason, optimisation steps, training seconds, and train MSRE per epoch.
- Panel `psc256` (one A100 allocation): `bankknob.py` with `../configs/pl256.json` unchanged (NM-ROM spans R' 32–512,
  head arms, FOM grid, dev6, same timing), then the 6 networks through `optime.py` / `opscore.py`, exactly like pl256.
- Output: `results.json` here. `../cost_points.json` gets a separate cell with `"variant": "data_scaling"`; pl256
  is not touched.
- Checkpoints are deleted on the cluster after the panel.
