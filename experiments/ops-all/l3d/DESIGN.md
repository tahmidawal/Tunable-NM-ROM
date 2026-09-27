# ops-all / l3d — neural operators for the Poisson 3D and Heat 3D rows of Table 1 (design, written before any cluster job)

Group B of `OPS-ALL-PROTOCOL.md`. Cells: Poisson 3D and Heat 3D at $32^3, 64^3, 128^3, 256^3$. Cluster namespace
`/cluster/tufts/paralab/tawal01/opsall_20260924/l3d/<job>/`. No number in this file is a measurement.
Amendments are appended as §A1, §A2, …; nothing above them is edited afterwards.

## 1. What each cell compares (one panel job, one GPU, one process)

| | Poisson 3D (source: `2026-09-23-poisson-bank-knob-3d`) | Heat 3D (source: `2026-09-23-heat-bank-knob`) |
|---|---|---|
| Table-1 jobs reproduced | 4200246 (32³), 4202245 (64³), 4207945 (128³), 4203255 (256³) | 4197416 (32³–128³), 4207497 (256³) |
| cohort | 32³/64³: reserved final seed 920499, 64 cases (held-out); 128³/256³: validation seed 920411, 16 cases (development) | held-out sealed seed 921099, 64 cases, every mesh |
| NM-ROM accurate / fast | span $R'=128$ / $R'=64$ (linear rung of the rotated nested bank, `pbk3_cube.make_linear`) | span $R'=320$ / $R'=128$, Crank–Nicolson $\Delta t=0.025$ (`hbk_core.make_linear`, init `field`) |
| FOM grid | unpreconditioned CG, rtol ∈ {0.3, 0.1, 0.03, 0.01, 0.003, 0.001}, cap 40 n | warm-started CN–CG, the source config's grid (Δt ∈ {0.025, 0.05, 0.1} × rtol ∈ {1e-2, 1e-3, 1e-4}, plus Δt 0.025 rtol 1e-6; at 256³ also Δt 0.0125 rtol 1e-4/1e-6) |
| truth / error | exact discrete solution (DST); worst over cases of the same-grid relative $L^2$ error | exact same-grid time evolution (DST); worst over cases of the max over **all six** output times of the relative $L^2$ error |
| timing scope | GPU query: device-resident forcing → device-resident field | GPU query: device-resident $u_0$ → six device-resident fields |

The NM-ROM and FOM arms are built by the source lanes' own code, copied byte-identical (`deps/`, `COPIED-FROM.json`),
from the same frozen checkpoints and pinned rotations. **Reproduction gate:** per-case errors of both NM-ROM arms must
equal the source job's per-case errors to ≤1e-6 relative, and the Table-1 FOM setting (Poisson CG rtol 1e-2; heat CN–CG
Δt 0.025 rtol 1e-4) to ≤1e-3 relative (loose-tolerance CG stopping is rounding-sensitive across GPUs; the other FOM
settings' gaps are reported, not gated). Local smoke (GB10, first 3 cases at 32³): NM-ROM gaps 6e-15 (Poisson) and
2e-11 (heat).

**FOM rule (the protocol's):** the fastest FOM setting in this job, all cases converged and finite, whose worst error is
≤ the NM-ROM accurate arm's worst error. Every speedup divides that one median by the arm's median, same job.

## 2. Operators

* **Code.** `ops_l3d.py` = `ns3d-operators/ops3d.py` (the NS 3D Table-2 lane, @766c3247) with only the Dirichlet
  changes: zero instead of circular padding, DeepONet trunk $\sin/\cos(\pi f x)$ instead of $2\pi f x$, no Leray
  projector, channel counts from the problem. FNO = official NeuralOperator 2.0.0 with the reviewed float64 spectral
  convolution (float64); U-Net / Transolver / DeepONet float32 with float64 I/O.
* **Contract.** Input = the query's own device-resident input ((n−1)³ interior field). Features on the $n^3$ grid
  (index 0 = boundary face, value 0): field / input scale and the three coordinates $i/n$ (the 2D cells'
  coordinate channels). Output cropped to the interior, times the output scale, float64. Heat: five channels = the
  fields at $t = 0.1,\dots,0.5$; $t=0$ returns $u_0$ exactly. Poisson: one channel.
* **Sizes — one per family, no tuning.** The NS 3D lane's validation-selected sizes (its `configs/panel{32,64,96}.json`):
  FNO-l (width 32, modes 16³, 4 layers), U-Net-l (base 32), Transolver-s (dim 128, 6 layers, 8 heads, 32 slices,
  16³ tokens), DeepONet-s (width 32, rank 256, trunk 384). At 256³ the smallest trained sizes of that lane:
  FNO-s (width 24, modes 12³), U-Net-s (base 16), Transolver-s, DeepONet-s.
* **Data.** The NM-ROM's training seed and count: Poisson seed 920410 × 512 draws; heat seed 921000 × 2048 draws.
  Split: index mod 8 = 7 → validation (checkpoint choice only), rest → gradients. Generated **on the fly on the GPU in
  float64** at the target mesh by `problems.py`, a line-by-line PyTorch port of the source generators (checked
  against them: gap ≤ 2e-16 at 16³/32³, `smoke_generators.py`). Evaluation-cohort rows asserted disjoint.
* **Protocol** (= ns3d-operators `train_op.py`): AdamW lr 1e-3, wd 1e-4, batch 8 (micro-batches + accumulation if 8
  does not fit), clip 1.0, cosine-in-wall-time to 1e-5, Transolver warm-up 160 steps, **3000 s wall budget**, epoch
  cap 4000, patience 250; checkpoint = best validation mean over cases of the per-case metric (Poisson: relative L2;
  heat: max over times). Loss = mean squared relative error (heat: over the five evolved times). Seed 20260914.
* **Declared deviation — shared GPUs.** GPU cap is 5 for this group, so the four families of a cell train
  **concurrently on one GPU** (32³ and 64³ of one PDE share one A100-80G, eight processes), each with a fixed torch
  memory fraction. The 3000 s budget therefore buys less work than a dedicated GPU would; epochs/steps are reported.
  An arm that cannot fit micro-batch 1 in its fraction, crashes, or is cut by the time limit is reported as **not
  trained**, never shrunk or retried with another size.

## 3. Jobs

Training: `tr_p3264` (Poisson 32³+64³, A100-80G), `tr_h3264` (heat 32³+64³, A100-80G), `tr_p128`, `tr_h128`
(A100-80G), `tr_p256`, `tr_h256` (H200, `--mem 240G`, attempted once). Then one panel job per cell (A100-80G for
32³–128³; H200 for 256³), checkpoints copied cluster-side from the training job dir. Timing: 2 untimed burn-in calls
per arm, then 3 rounds × every case, a fresh random arm order per (round, case), a 50 ms GPU burn before every timed
call; reported time = median of all samples; drift gate = median(last round)/median(first round) within 1.10.
Panel `status` = final only if reproduction, drift (selected arms) and FOM-found gates pass.
Only `summary.json`, `result.json`/`history.json`/`provenance.json` and logs are pulled; checkpoints are deleted
with the remote job dirs after the panels.

## A1 (2026-09-24 ~14:55 EDT, after the first 256³ attempts, before any 256³ panel) — 256³ harness failure, second attempt

The first 256³ training jobs (`tr_p256` 4286518, `tr_h256` 4288846, H200) ran the four families concurrently with
memory fractions FNO-s 0.36, U-Net-s 0.20, Transolver-s 0.08, DeepONet-s 0.30. FNO-s and DeepONet-s found no
micro-batch that fits in their share (both PDEs; recorded). Transolver-s (both PDEs) and U-Net-s (heat) died **in the
harness**, not in the network: the normalisation pass generated 16 draws at once (a 256³ heat batch of 16 alone is
~13 GB). Fix (memory only, no value changes beyond summation order): at n > 128 the normalisation and validation
passes generate one draw at a time. Second attempt `tr_256b` (H200): first a **whole-GPU memory probe** (fraction
0.95, micro-batch 1, two full AdamW steps, probe only, no training) of FNO-s and DeepONet-s for both PDEs, so the OOM
record is about the card and not about the share; then heat U-Net-s (0.45), heat Transolver-s (0.20) and Poisson
Transolver-s (0.20) train concurrently with the 3000 s budget. Poisson U-Net-s trains in `tr_p256` unchanged. The
fractions differ from the first attempt (the freed memory of the failed arms is given to the survivors). Nothing
else changes; no panel has been run at 256³.

## A2 (2026-09-24, after the 32³–128³ panels) — observed deviations, recorded, not acted on

* Heat 64³ (`pn_heat64`): the reproduction gate failed on the Table-1 CN–CG setting (Δt 0.025, rtol 1e-4; max
  per-case relative gap above the pre-set 1e-3; NM-ROM arms equal to the source). Source ran on an H200, this job on
  an A100; loose-tolerance CG stopping is rounding-sensitive. Panel stays PROVISIONAL by its own rule.
* NM-ROM medians are higher than in the Table-1 source jobs by more than the FOM medians (generated comparison table
  in REPORT.md); not diagnosed; the operator-vs-FOM ratios are same-job and unaffected.

## A3 (2026-09-24 ~15:05 EDT, before any 256³ panel) — Poisson U-Net-s also died in the harness; tr_256b replaced by tr_256c

`tr_p256`'s Poisson U-Net-s trained (micro-batch 1) but ran out of its 0.20 share in the first **validation** pass,
which generated 4 draws at once (fixed by A1's one-draw-at-a-time rule). `tr_256b` (4290426) was cancelled while
still pending (never started, directory removed) and replaced by `tr_256c` (H200): the same four whole-GPU probes,
then heat and Poisson U-Net-s (0.30 each) and heat and Poisson Transolver-s (0.15 each) train concurrently, 3000 s
each. This is the second and last 256³ training attempt.

## A4 (2026-09-24 ~17:15 EDT, before any 256³ panel) — whole-H200 probes pass; FNO-s and DeepONet-s get a dedicated H200 each

`tr_256c`'s whole-GPU probes (micro-batch 1, two AdamW steps): FNO-s peaks at ~118 GB (Poisson) / ~117 GB (heat),
DeepONet-s at ~96 GB / ~95 GB — they fit an H200 but not the 0.36/0.30 shares of the first attempt. So each gets one
dedicated H200 (fraction 0.95), same size, same protocol, 3000 s: `tr_256fp`, `tr_256dp`, `tr_256fh`, `tr_256dh`.
The 256³ panels were cancelled while pending (never started) and resubmitted to depend on all 256³ training jobs;
the panel takes each arm's record from its latest attempt (first-attempt failure records only if no later record).
Unlike the other meshes, these four arms train alone on their GPU (more work per 3000 s than the shared arms).

## A5 (2026-09-24 ~17:55 EDT, before any 256³ panel) — A4 withdrawn (coordinator: GPU cap 5 → 2, 2D cells first)

The four dedicated-H200 FNO-s / DeepONet-s trainings of A4 were cancelled while still pending (never started,
directories removed). At 256³, FNO-s and DeepONet-s are therefore reported as **not trained**: they do not fit the
shares of a shared H200 (first attempt), and the whole-GPU probes show they would need a dedicated H200 each (peak at
micro-batch 1 printed in the report), which the reduced GPU cap and the reporting window did not allow. The 256³ panels
depend on `tr_256c` only and time U-Net-s and Transolver-s (if trained) with the NM-ROM and FOM grid.

## A6 (2026-09-24 ~18:35 EDT) — correction to A4's probe numbers

A4 swapped the two families: from the pulled `probe.json` files, the whole-H200 micro-batch-1 peaks are
FNO-s ~88 GiB (Poisson) / ~90 GiB (heat) and DeepONet-s ~109 GiB (Poisson) / ~110 GiB (heat). The report's failure
strings are generated from those files.
