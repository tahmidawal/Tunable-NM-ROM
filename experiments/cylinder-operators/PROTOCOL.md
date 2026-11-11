# Protocol: learned-operator baselines on the unsteady-cylinder ROM data

Fixed 2026-11-11 before any operator was trained or any outcome was seen. Later changes are
appended under "Amendments" with the reason.

## Question
On exactly the snapshot data the unsteady-cylinder ROM used, how do standard learned surrogates
compare with the ROM and POD-Galerkin rows on the same six targets and the same rollout window?

## Data (identical to the ROM lane, exp/2026-11-03-cylinder-unsteady-rom @ fccd7658a)
- DFG 2D-2 cylinder, reference P4/P3 Taylor-Hood, saves every 1/400 s, recorded window 1.7 s,
  681 saves per trajectory (manifest inputs/manifest_production.json, mesh sha 528ffff3...).
- Train: Re 90, 100, 110 (3 x 681 saves). Nothing else is used for training: no extra Reynolds
  numbers, no settle-stage data, no pretraining.
- Development: Re 95. Held-out: Re 105 (sealed; see below).
- Every surrogate is rolled out / queried from the FIRST saved state of the window (plus Re)
  and must produce all 681 saves, like the ROM (which starts from the projected first state).

## Representations (prepared once by prep.py, NGSolve)
- N2 / N3: values of (u, v, p) at the element-local Lagrange nodes of degree 2 / 3 on the FE
  mesh (shared nodes merged). Map to the scoring quadrature: Lagrange interpolation in element
  reference coordinates (sparse matrix).
  Rule: the mesh-agnostic and coordinate-network arms use N2 if its interpolation floor on train
  Re 100 is e_u <= 5e-4 (20x below the e_u target), otherwise N3.
- G: cell-centred regular grid 1024 x 192 over the channel [0,2.2] x [0,0.41]; cells with centre
  inside the cylinder are masked (velocity 0, pressure = nearest fluid cell). Map to the scoring
  quadrature: bilinear interpolation on the grid augmented with the known boundary data (no-slip
  walls, the prescribed inlet profile, copy at the outlet).
- Interpolation floors (reference field -> representation -> quadrature) are reported for each
  representation; they bound what the arm can reach and are part of the result.

## Metric (where it is computed)
All arms are scored on the ROM's own scoring quadrature Q (order 12 on the curved mesh,
1,036,497 points) after mapping the native prediction to Q as above:
- e_u(t) = ||u_pred - u_ref||_L2(Q) / ||u_ref||_L2(Q) at each of the 681 saves (phase sensitive);
  targets RMS over saves <= 1e-2 and max <= 3e-2.
- e_p(t) = ||p_pred - p_ref||_L2(Q) / sqrt(|Omega|); target RMS <= 2e-2. Pressure is compared
  as predicted (no mean shift).
- Forces: C_D(t), C_L(t) from the predicted fields by one gradient-free volume functional applied
  identically to every arm (smooth cut-off eta = 1 on the cylinder, 0 beyond distance 0.1):
  F_i = -d/dt int u_i eta + int u_i (u.grad eta) + int p d_i eta + nu int u_i lap eta
        + nu int u_j d_i d_j eta,  C = 20 F  (4th-order differences in time on the save grid).
  The functional is first applied to the reference fields; its deviation from the reference
  solver's own cycle statistics is reported as the read-out floor.
- Cycle statistics exactly as the ROM lane (cycles.py: lift minimum to lift minimum, parabola
  refined, mean over complete cycles) against the reference solver history:
  St relative <= 1e-2, C_D,max relative <= 2e-2, C_L,max absolute <= 2e-2.
- Long-rollout stability: e_u(t) curve, last-period mean e_u, and the lift phase lag at the end
  of the window (best lag on the save grid, same support rule as rom_r4.py); "phase accurate"
  means |lag| <= 2 saves (1.5 % of a period) at the end of the window.
- A non-finite or missing rollout counts as failed and is kept in the table.

## Arms
1. Mesh-agnostic operator: Transolver-style transformer (physics-attention slices) on the node
   set, autoregressive, temporal bundle of K saves per call.
2. Regular-grid operator: FNO and U-Net on G (both are candidates; validation picks one),
   autoregressive, temporal bundle of K saves per call.
3. Coordinate network: DeepONet-style, branch = (first state at fixed sensor nodes, Re),
   trunk = (x, y, tau); not autoregressive: every save is one direct query from the first state.
   Trained on all ordered pairs (state_i -> state_j, tau = t_j - t_i >= 0) inside the training
   windows.

## Tuning and training (no development or held-out data involved)
- Candidate configurations per arm are listed in configs.py before training (at most 4 per arm).
- Selection: train each candidate on Re 90 + 110 for the tuning budget, roll out the full Re 100
  window from its first state, pick the lowest RMS over saves of the native relative velocity
  error. Then retrain the selected configuration from scratch on Re 90 + 100 + 110 with the
  final budget. No early stopping on development data; the last checkpoint is used.
- Budgets (wall time of the optimisation loop, one GPU): tuning 20 min per candidate, final
  60 min. Cosine learning-rate decay to zero over the budget. "Plateau" is reported as the
  relative change of the smoothed training loss over the last 20 % of the budget; an arm whose
  loss is still falling by more than 5 % there is reported as not converged (and may be given
  one longer run, recorded as an amendment).
- float32 training and inference (the usual setting for these models); the ROM rows are float64.
- Development Re 95 is evaluated after the final training of each arm. If the development
  result exposes an implementation fault, the fix and rerun are recorded as an amendment.
- Held-out Re 105: its arrays are written to a separate sealed directory by prep.py and are read
  by exactly one script (final_eval.py), once, after all arms are frozen (checkpoint hashes
  listed in the status file before that job is submitted).

## Cost
- Training cost: wall seconds of every tuning and final run, GPU model, summed GPU-hours.
- Inference: wall time to produce the 681 saves of one window in the native representation on
  the GPU (model calls only, state already on the device), one warm-up excluded, 5 repeats,
  median and min-max; GPU model recorded. The ROM/FE rows are the ROM lane's CPU timings; the
  hardware difference is stated next to the numbers.

## Comparison rows (read from the ROM lane's saved JSON, not retyped)
POD-Galerkin and bank ROM at r = 16/24/32 on Re 95 (and the held-out rows the ROM lane
produces). An operator that beats the ROM on these targets is reported exactly so.

## Amendments
- 2026-11-11 03:00 (before any training job): the candidate list lives in `ops_train.py`
  (`CONFIGS`) rather than a separate configs.py; it is committed before the first training job.
  Node rule outcome from prep01: N2 floor on Re 100 is below 5e-4, so arms 1 and 3 use N2.
- 2026-11-11 04:58 (after the 60-min finals of arms 1 and 3 and their development scores; held-out
  still sealed): both finals are flagged not converged by the protocol's own rule (training loss
  still falling over the last 20 % of the budget: node 32 %, coordinate network 13 %). As the
  protocol allows, each gets ONE longer run: same selected configuration, same data, same
  everything, final budget 180 min instead of 60 (no re-tuning, no change picked from development
  results). The grid arm gets the same if its 60-min final is flagged. Both the 60-min and the
  180-min checkpoints are kept, both are listed (hashes) before the single held-out job, and both
  are reported on development and held-out; neither is dropped.
- 2026-11-11 04:58 observation recorded before any fix: on development the node arm's lift from
  the fixed read-out has save-to-save jitter (about 0.02 in C_L), which makes the ROM lane's
  cycle detector (cycles.py, lift minimum to lift minimum) count 15 "cycles" instead of 4, so the
  St and C_L,max targets fail under the fixed read-out. This is reported as measured. A
  supplementary read-out (same forces, cycle statistics taken after a centred 9-save moving
  average of C_D and C_L; chosen now, before looking at its outcome) is reported alongside,
  labelled post hoc, for every arm; it does not replace the fixed one.
- 2026-11-11 05:00 (after checking the 04:58 supplementary read-out offline on the saved 60-min
  development series): with the 9-save average the node arm still shows one double lift minimum
  (5 "cycles"), so the supplementary read-out also fails St / C_L,max for it; reported as such.
  Added a third, explicitly diagnostic read-out for every arm (smoothed signals, lift minima closer
  than half a reference period to a lower one are discarded) to state what the underlying shedding
  frequency and amplitude are. It was chosen after seeing the outcome and is never counted
  towards the six targets.
- 2026-11-11 08:25: all arms frozen; checkpoint hashes (sha256 of final.pt) listed before the single
  held-out job final01:
  node60  069f5b8560e3aa39214adfd32931401f0cf07f4a799157fcec87548222518229  node01/results/node/final.pt
  node180 2a9b5143121f78a1f7af6c7b63dd9b07786ed3663e5c373794086f9f377f2eae  node02/results/node/final.pt
  grid60  a2af5440073637702257e68e6f6fbc7b8381cab08123428d23adc9ce20fbcc95  grid01/results/grid/final.pt
  grid180 1a07c6ad90d416beb1df66c9d55893b8c86bf1ef6bb6041b52fd91b1480e309a  grid02/results/grid/final.pt
  don60   a1f161d8c4b253592ab9f02fd5ff0075f35212794015f6bda66b823d5263c191  don01/results/don/final.pt
  don180  1aa0d510ab8487097bc760c538be8736b5f741a23006bf5ef36f34389781519b  don02/results/don/final.pt
