# Rebuttal push protocol: neural bank on the unsteady cylinder (fixed 2026-11-11, before any new outcome)

Scope: continue the G5 ROM lane. Reference, export, quadrature, ROM formulation (rom_core.py), time
step dt0 = 1/3200 and the six targets are unchanged from the finished campaign (report 2026-11-06).

## Cases
- Train Re 90/100/110 (POD teachers only), development Re 95, held-out Re 105.
- Re 105 is the only held-out trajectory. It was opened once in the earlier campaign for the then
  frozen arms (POD r8/16/24, bank r16, frozen bank r4). No arm, FE configuration or setting in this
  push is selected, tuned or inspected on Re 105.

## Metric and gates (unchanged, rom_r4.TARGETS)
e_u RMS <= 1e-2, e_u max <= 3e-2, e_p RMS <= 2e-2, St rel <= 1e-2, CD_max rel <= 2e-2,
CL_max abs <= 2e-2, phase-sensitive over the 1.7 s recorded window, rollout from the projected
first state. Divergence (0.1*||div u||/sqrt|Omega|) is reported, not gated.

## Step 1: diagnosis (development data only)
For the existing r16 bank and POD at the same rank: per-mode regression errors (value, gradient,
divergence), sigma^2-weighted subspace miss against the 24 POD modes, development projection floors,
and the share of regression error and of rollout error by region
(cylinder: within 0.05 of the surface; walls: within 0.03 of a channel wall; wake: x > 0.25
otherwise; upstream: the rest).

## Step 2: training arms (each its own job directory; all recorded, none dropped)
Common: coordinate network with the same boundary envelope; fixed seed 3112026; teacher = POD modes.
- A1 r16-long: original architecture/features (3x128), Adam 20k + L-BFGS to its deadline.
- A2 r16-cap: 4x256, extra streamwise Fourier features (16, 32), long training.
- A3 r24-cap: as A2 at rank 24.
- A4 r24-vp: A3 warm start, then joint sigma^2-weighted subspace (variable-projection) fine-tune.
- A5 r24-stream: divergence-free construction (stream function, u = curl psi), rank 24.
- A6 r24-div: A3 warm start + divergence penalty.
- A7 r24-wake: A3 warm start + loss weight toward cylinder/near wake (x in [0.12, 1.3], factor 4).
- A8+ r32: needs a new export with 32 modes; best r24 recipe(s) repeated at rank 32.
Further arms may be added only as a recorded amendment with a reason. The saved bank is always put
in PCA order by the fixed linear map to the best approximation of POD modes 1..r.

## Selection (development Re 95 only)
Among arms meeting all six targets on Re 95, freeze the one with the smallest e_u RMS; if several
are within 5 % of each other prefer the smaller rank, then the smaller divergence. If no arm meets
all six, freeze the arm with the most targets met, then the smallest e_u RMS, and report the miss.

## Step 3: held-out evaluation (once)
One job: frozen bank arm at full rank and its PCA prefixes, POD at the same ranks, on Re 105:
all six targets, divergence, paired timings (warm-up excluded, repeated, with spread, one allocation).

## Step 4: accuracy-matched FE comparator
FE ladder with the reference solver (same mesh and scheme): order P4/P3, P3/P2, P2/P1 and step
1/1600 ... 1/200 (frozen lag kept at 1/80 s), from the reference state at the start of the Re 95
window (L2-projected to the lower order), scored on the same six targets against the reference.
Comparator rule: the cheapest (wall time per simulated second, 16 threads, timing mode) ladder
entry meeting all six on Re 95 is the accuracy-matched FE. It is then run once on Re 105 (targets
checked, timed in the same allocation as the ROM). Not covered, to be stated: coarser meshes (no
admitted coarser mesh exists) and other time integrators.

## Amendments

- A1 (2026-11-11 00:30, before any arm result): every arm uses the same budget so arms are time-matched:
  stage 1 = Adam 20k steps + L-BFGS for 1800 s; stage 2 = 1200 s (plain continuation for A1/A2/A3/A5, the
  arm's own variant for A4/A6/A7). The extra streamwise Fourier frequencies are 12,16,24,32,48 (the text
  above said 16, 32). A6 divergence weight is 10; A7 box weight 4.
- A2 (2026-11-11 00:45, before any ladder result): the FE ladder keeps the frozen-convection lag at 1/80 s
  where the step divides it (1/400: K=5, 1/800: K=10, 1/1600: K=20); at step 1/200 it uses K=2 (lag 1/100 s)
  because 1/80 s is not a multiple of the step. Ladder entries run with the ROM's output contract (reaction
  forces and probes every step, state every save; no surface-force integral), so the stage wall time is the
  entry's cost. Lower-order start states are the reference state interpolated into the lower-order space.
- A3 (2026-11-11 01:10, after the first three ladder entries: P2/P1 at step 1/400 with lag 1/80 s meets all
  six on Re 95 in 302 s; step 1/200 misses e_u RMS and CL_max): to give the FE comparator its own choice of
  solver setting, the frozen-convection lag is also laddered at the cheapest passing step (refresh every
  10, 20, 40 steps and never after the start-up). The accuracy-matched FE is the cheapest entry of the
  enlarged ladder that meets all six on Re 95. This can only make the FE cheaper, i.e. the reported
  speed-up smaller.
- A4 (2026-11-11 01:45, before any ROM result at a step other than 1/3200 and before any FE entry with lag
  above 40 steps): the accuracy-matched FE chose its own step and lag on Re 95, so the ROM gets the same
  choice. On Re 95 the frozen bank arm and POD at the same ranks are rolled out at steps 1/1600, 1/800 and
  1/400 as well as 1/3200; the ROM's cost-comparison step is the largest step at which the arm still meets
  all six targets on Re 95. Held-out Re 105: the primary accuracy table stays at step 1/3200 (the step used
  so far); the dev-chosen step is reported next to it and is the one used for the accuracy-matched speed-up.
  Both sets of numbers are reported whatever they show. The FE lag ladder is extended by refresh every 80
  and 160 steps at step 1/400 (same rule: cheapest entry meeting all six on Re 95).
- Note (2026-11-11 01:45): job rb-fe1 was cancelled by hand after four entries (P2/P1 at 1/200, 1/400, 1/800
  and P3/P2 at 1/200) because P3/P2 cost about three times P2/P1 for the same errors; P3/P2 at 1/400 and
  1/800, P2/P1 at 1/1600 and all P4/P3 entries of the planned ladder were NOT run. The reported
  accuracy-matched FE is therefore the cheapest of the entries actually run, not of the full planned grid.
- Freeze (2026-11-11 02:45, development Re 95 only, before any held-out number of this push): frozen arm = rb-a8b
  (job 7118315; rank 32, 4x256 + streamwise Fourier features, time-matched budget), bank_r32: 6/6 on Re 95, e_u RMS
  1.912e-3 (POD r32 1.867e-3). sha256 trained_bank_eval.npz b5915d667372345adfedfb15817c8071e346ea1dc49543204d4d50e10f820fd4,
  trained_bank_params.npz ad9a6bd3ba38e57e71a2a5fad47b259c7a22e3127cd70efe08494f93439b1f3d. Arms finished at the freeze:
  a1 (r16) 9.556e-3, a2 (r16) 9.523e-3, a3 (r24) 5.632e-3, a8b (r32) 1.912e-3. Arms still running or queued at the
  freeze (a4, a5, a6, a7) are all rank 24, whose rollout error cannot fall below the r24 projection floor on Re 95
  (POD r24 floor 2.35e-3 > 1.912e-3), so by the selection rule they cannot displace a8b; they are run to the end and
  recorded as arms, not as candidates. The held-out job also reports the frozen bank's PCA prefixes r24 and r16.
- A5 (2026-11-11 02:45): rb-dev2 (step ladder for the a3 bank) was cancelled before it started and replaced by
  rb-dev2b, the same step ladder for the frozen a8b bank (r32 and prefixes r24, r16) and POD r16/r24/r32.
- Records (2026-11-11, read after the freeze; development Re 95, step 1/3200, values from artifacts/rb-a{4,5,6,7b}/results/r4/R4-re95.json):
  a4 (r24 variable projection) 6/6, e_u RMS 5.821e-3, divergence 9.33e-3; a5 (r24 stream function) 6/6, 6.153e-3, divergence
  1.93e-4 (below POD's 2.44e-4); a6 (r24 divergence penalty) 6/6, 5.114e-3, divergence 7.51e-3; a7b (r24 wake weight) 6/6,
  5.726e-3, divergence 9.13e-3. None displaces the frozen a8b (1.912e-3). a7 (7120265) was replaced by a7b before it produced
  a result; a8 (7117595) failed its GPU preflight and was rerun unchanged as a8b.
- A6 (2026-11-11 10:10 EST, after the dev step ladder rb-dev2b was read and BEFORE the held-out case is opened in this push;
  everything below is decided from development Re 95 only).
  (a) What the timings contain (read from the code). `rollout_seconds` in rom_r4.py and `warm` in rom_time*.py are the wall
  time of `rom_core.Arm.rollout` only: NumPy on the CPU (16 threads; the allocation holds a GPU that the rollout does not
  use), BDF2/Newton steps of the reduced system, with the reduced velocity and pressure coefficients, the reaction force
  and the two probe pressures at every step. NOT included: the operator/tensor build and start projection (per basis,
  offline), bank training, any reconstruction or output of a full field, and the error evaluation. The FE time is the
  stage wall time of the stepper (assembly, LU refreshes, solves, forces and probes every step, the full state kept in
  memory at every 1/400 save); its setup and construction are excluded. Because the FE time contains the full state at
  every save and the ROM time does not, the ROM field output is measured separately (rom_time2.py: velocity and pressure
  at every 1/400 save on every 21st quadrature point, 49,357 points = 148,071 values per save, more than the P2/P1
  comparator's 96,781 unknowns; and once on all 1,036,497 quadrature points) and the speed-up is reported both without
  and with it. The build time is reported next to them.
  (b) ROM step for the cost comparison (rule of A4 applied to rb-dev2b): the frozen arm bank_r32 meets 6/6 on Re 95 at the
  largest laddered step 1/400 (e_u RMS 6.465e-3), so its cost-comparison step is 1/400. The same rule per prefix:
  bank_p24 -> 1/400 (6/6, 4.660e-3); bank_p16 -> 1/800 (6/6 at 1/800, 5/6 at 1/400). POD at each rank is compared at the
  same step as the bank of that rank.
  (c) Arms for the cost comparison. Headline: the frozen arm bank_r32 at step 1/400. Declared secondary rows: bank_p24 at
  1/400 (the cheapest bank configuration meeting 6/6 on Re 95: 0.63 s rollout) and bank_p16 at 1/800. On the held-out case
  every arm (bank r32, p24, p16; POD r32, r24, r16) is rolled out at all four steps (1/3200 primary accuracy table, 1/1600,
  1/800, 1/400) and every row is reported whatever it shows; a configuration that fails a target on Re 105 is reported as
  failing and is not replaced by another step or rank chosen on Re 105.
  (d) Speed-up = FE stage wall time (median of 5) / ROM time (median of 5), both measured in one allocation in the order
  ROM, FE, ROM (16 threads). Reported twice: ROM rollout only, and ROM rollout + field output of (a).
  (e) Change to step 4 (FE on the held-out case). Step 4 said the dev-chosen FE entry is run once on Re 105. Instead the FE
  gets its own ladder on Re 105 and the accuracy-matched FE there is the cheapest entry meeting all six on Re 105 (same
  rule as on dev, select_fe.py). Reason: the comparator must be allowed its cheapest passing setting on the case it is
  scored on; the dev choice (P2/P1, 1/400, refresh 40) may not pass, or a cheaper one may pass. This lets the comparator,
  and only the comparator, be chosen on the held-out case; it can only lower the reported speed-up. Ladder (P2/P1, step
  inverse, refresh): (200,2) (200,20) (400,20) (400,40) (400,80) (800,40) (800,80); the dev choice is in it and is reported
  whatever is chosen. The same list is rerun on Re 95 in one allocation with the ROM timings (rb-devt); it contains entries
  not run on dev before ((200,20), (800,40), (800,80)); if one of them is a cheaper 6/6 entry it becomes the dev
  accuracy-matched FE (rule unchanged).
  (f) rom_time.py rebuilt the operators twice per repetition (about a minute per build at rank 32), which would take hours
  for six arms at four steps; rom_time2.py builds once untimed, once timed, then repeats the rollouts. tasks/rb_ho_common.sh
  (template, never run) is superseded by tasks/rb_ho1.sh + tasks/rb_cmp_common.sh.
