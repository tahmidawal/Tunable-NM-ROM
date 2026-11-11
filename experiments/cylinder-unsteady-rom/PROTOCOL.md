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
- Held-out opened (2026-11-11, rb-ho1 job 7133112, one job, no rerun, nothing changed afterwards). Re 105, frozen a8b bank,
  values in artifacts/rb-ho1 and artifacts/HEADLINE-A6.json: every arm (bank r32/p24/p16, POD r32/r24/r16) meets 6/6 at all
  four steps. bank_r32 at 1/3200: e_u RMS 1.892e-3 (POD r32 1.860e-3). Accuracy-matched FE chosen on Re 105 by the rule of
  A6(e): P2/P1, 1/400, refresh 40 (122.86 s, median of 5); (200,20) and (400,80) unstable, (200,2) 4/6. Headline (A6 b-d):
  bank_r32 at 1/400, rollout 1.716 s -> 71.6x; with field output 3.98 s -> 30.9x. Dev pair (rb-devt 7133114): 61.5x / 25.0x.
  Not claimed: bank_p16 at 1/400 also meets 6/6 on Re 105 (0.40 s) but failed on dev, so by A6(b) its step is 1/800.

## Second rebuttal push (2026-11-11 afternoon): stronger FE comparator, stream-function bank, extrapolation

Amendments A7-A9 were written at 2026-11-11 14:00 EST, before any job of this push was submitted and before any outcome
of it existed. State at that time: frozen bank a8b, held-out Re 105 opened once (rb-ho1), FE comparator P2/P1 on the
admitted mesh at step 1/400 with refresh 40. Nothing below changes those records.

- A7 (FE comparator: coarser meshes, explicit-convection integrator, intermediate steps).
  (a) Meshes. The admitted mesh m0 was generated by netgen with maxh 0.01 and obstacle_h 0.0025, curve order 4. Coarser
  meshes of the same geometry, same generator, curve order 4: c1 (0.014, 0.0035), c2 (0.02, 0.005), c3 (0.028, 0.007),
  c4 (0.04, 0.01) (cell size x sqrt2, 2, 2 sqrt2, 4). Element counts are recorded when generated (fe_mesh.py).
  (b) Entry = (mesh, Taylor-Hood order k in {2,3}, step 1/N, scheme, refresh K). Schemes are those of the vendored
  reference stepper: `frozen_oseen` (the reference integrator; refresh every K steps), `sbdf2` (BDF2 with extrapolated
  explicit convection: one factorisation for the whole window), `oseen_bdf2` (linearly implicit, matrix reassembled every
  step, stale-LU GMRES). Screening ladder on development Re 95 (one run each): every mesh x k x {frozen (400,40),
  (300,30), (200,20); sbdf2 400, 800, 200}, plus oseen_bdf2 (200, K 20) for k=2 on m0 and c1. Step 1/300 is scored on the
  saves it shares with the reference (every 0.01 s); all others on all 1/400 saves.
  (c) Start state: the reference state at the first save, L2-projected onto the entry's space (least squares on the
  reference quadrature Q), Dirichlet values re-imposed. Scoring: the same six targets; e_u and e_p evaluated on the
  quadrature Q of the admitted mesh (order 12, 1,036,497 points - the points and weights the reduced models are scored
  on), the entry's field evaluated pointwise there (points outside a coarser mesh, if any, take the value 0 and are
  counted); St, CD_max, CL_max from the entry's own per-step reaction force. The m0 entry (k=2, 400, frozen 40) repeats
  the previous comparator and is the cross-check of the new scoring against the mass-matrix scoring of fe_ladder.py.
  (d) Rule (unchanged in spirit): the accuracy-matched FE on a case is the cheapest entry (median stage wall time)
  meeting all six targets on that case. Because screening runs are single and spread over allocations, the paired job
  re-times the five cheapest passing entries of the screening (5 repetitions each) in the same allocation as the ROM
  (order ROM, FE, ROM; 16 threads) and the cheapest of those by median is the comparator. On Re 105 the FE gets its own
  ladder as in A6(e): the eight cheapest dev-passing entries plus the previous comparator are run on Re 105, the five
  cheapest passing there are re-timed, the cheapest is the Re 105 comparator. Speed-ups are reported as in A6(d), for
  rollout only and with field output, whatever they turn out to be; the previous numbers stay in the record.
  (e) Not covered (stated in the report): elements other than Taylor-Hood (stabilised P1/P1, MINI), iterative or
  multigrid solvers, adaptive meshes, GPU finite elements, time integrators other than the three above.
- A8 (stream-function bank at rank 32). Same trainer flag as arm a5 (`--stream`), on the 32-mode export rb-export32.
  Arms: s1 = the a8b recipe plus `--stream` (same training budget as the frozen ordinary bank); s2 = s1 with the L-BFGS
  and fine-tune budgets doubled (3600 s / 2400 s). Selection on development Re 95 only, at step 1/3200: among stream arms
  meeting 6/6 at rank 32, the smallest velocity error; within 5 % of each other, the smaller divergence. If none meets
  6/6 the result is reported as a failure and the better one is still carried through. The selected stream bank is
  frozen by hash; its prefixes 24 and 16 come from the same bank. Then the development step ladder (1/3200 ... 1/400) fixes
  its cost-comparison steps by the rule of A4/A6(b). Reported against POD and the ordinary bank a8b at the same ranks:
  six targets, divergence, rollout time. Re 105 was opened once for the ordinary bank; the frozen stream bank is run on
  Re 105 once, in one job, with nothing selected there (this is a second use of that case and is reported as such); the
  untouched confirmation cases for it are those of A9.
- A9 (extrapolation in Reynolds number and a longer window). No training of any kind: the ordinary bank a8b, the frozen
  stream bank of A8, POD (ranks 32/24/16; modes of rb-export32 by hash) and the frozen neural operators of the operator
  lane are evaluated as they are.
  (a) New references (same mesh, order, integrator, steps and stage sequence as the existing ones): start from the
  recorded end state of the nearest training case (Re 90 or Re 110), settle at 1/800 with refresh 10 until the periodicity
  test passes or 6.0 s, 0.7 s at 1/3200, then the 1.7 s record at 1/3200 with saves every 1/400. A reference that has not
  met the periodicity test at the cap is still used and reported as such.
  (b) Roles, fixed now. DEVELOPMENT: Re 85 and Re 115 (5 outside the training band 90-110), and the long window on Re 95.
  FINAL CONFIRMATION: Re 80 and Re 120 (10 outside the band), and the long window on Re 105. Confirmation cases are
  opened once, in one job per method family, after every setting used on them has been fixed from development outputs
  and written here; nothing is changed afterwards.
  (c) Long window: the Re 95 and Re 105 records are continued from their saved end states for another 3.4 s (same step,
  refresh and saves), giving 5.1 s (about 15 shedding periods) from the same start. The continuation restarts the BDF2
  history with the stepper's start-up substeps at the junction; the two force rows without a consistent time derivative
  there are filled by linear interpolation in the merged history (merge_runs.py). Every method starts from the start of
  the original window and runs the whole 5.1 s.
  (d) Reported per case and method: the six targets, divergence, the phase lag of the lift (rom_r4 phase_aligned), force
  errors; for the long window also the error over each third of the window. ROM steps: 1/3200 and the cost-comparison
  step of each arm. Operator-lane details (which checkpoints, inputs, rollout form) are added as A9b before any operator
  is run on a new case.

**A7 addendum (2026-11-11, before any ladder outcome).** (i) The coarse meshes were generated once by `fe_mesh.py`
and are read by hash from `rb-meshes/` (elements; unknowns at k = 2 / 3): c1 11,050 (50,875 / 123,390), c2 5,444
(25,303 / 61,172), c3 2,882 (13,549 / 32,630), c4 1,480 (7,055 / 16,912); hashes are echoed into every ladder job.
(ii) A 24-step smoke test of `fe_ladder2.py` on c4 (0.06 s of the Re 95 window; four entries) was run to check the code
path (evaluation matrices agree with NGSolve point evaluation to 1e-15); it is not a ladder outcome and selected
nothing. (iii) The development screening and the paired dev timing run in ONE allocation (`rb-fa`): screening with one
run per entry, then ROM, the five cheapest passing entries with five repetitions each, ROM. (iv) All seven first
submissions of this push failed at the GPU preflight on node pax063 before any task code ran and were resubmitted with
that node excluded; the four extra references are packed two per allocation (16 threads each).

**A9b (2026-11-11, before any new reference exists and before any model is run on a new case).**
(i) Operators. All six frozen checkpoints of the operator lane are evaluated as they are, by hash: mesh-node
transformer 60 / 180 min (069f5b85..., 2a9b5143...), Fourier operator on the 1024 x 192 grid 60 / 180 min (a2af5440...,
1a07c6ad...), DeepONet 60 / 180 min (a1f161d8..., 1aa0d510...). No retraining, no fine-tuning, no change of
normalisation: the Reynolds input is the lane's (Re - 100)/10 feature, so Re 80-120 are outside the values seen in
training (90, 100, 110). Rollout exactly as in the lane's final evaluation: the reference state at the first save in the
arm's own representation, then the arm's autoregressive rollout over all saves (681, or 2041 for the long window); the
DeepONet is queried at tau = j/400, so the long window asks it for tau beyond the 1.7 s it was trained on and is
reported as extrapolation in time as well. Scoring: the lane's own scorer (`ops/ops_common.py`, byte-identical copy;
`ops/ops_models.py`, `ops/ops_train.py` likewise, hashes in `ops/VENDORED.txt`): fields mapped to the quadrature Q, the
same six targets, force read-out by the lane's volume functional, end-of-window lift lag, the lane's supplementary
smoothed read-out reported next to the fixed one. Data: `ops/ops_prep2.py`, the lane's prep.py with only bookkeeping
changes (any number of saves; reference on Q stored for every case; geometry compared with the lane's geom.npz, which
is the one the evaluation loads). Inference timing as in the lane (warm-up + 5 repeats, same GPU as the scoring).
Regression check: the development job also re-scores Re 95 (short window) from the lane's own data; its numbers must
reproduce the lane's final evaluation, otherwise the operator results are not reported.
(ii) Reduced models. On every new case: POD 16/24/32, ordinary bank a8b 32/p24/p16, frozen stream bank 32/p24/p16, at
all four steps of the development ladder (1/3200, 1/1600, 1/800, 1/400); reported at 1/3200 and at each arm's
cost-comparison step. Modes and banks by hash; the new trajectories are projected onto them only for the start
coefficients and the error evaluation (`rom_export.py --pod-from`, `rom_project.py`). The reference-uncertainty file
bound into `rom_r4.py` belongs to the Re 95 reference; its "resolved" labels are not used for the new cases.
(iii) Phase drift for every method, from the saved lift series (`drift.py`): the window is cut into consecutive
supports of one reference period; on each the lag minimising the RMS lift difference is searched within a quarter
period of the previous support's lag, so drifts beyond half a period are followed. Reported: lag at the end (in
periods), largest lag, and for the long window the errors over each third of the window. The end-of-window lag of
`rom_r4.py` / the lane's scorer (limited to half a period) is reported as well.
(iv) Jobs. Because the account's GPU cap is shared, each role (development, confirmation) is ONE allocation that runs
merge -> reduced models -> operators (`tasks/rb_xe_common.sh`); each family's block touches a confirmation case once.
If a block fails for a technical reason before producing any score, only that block is re-run and this is reported.
(v) What development is used for: checking the pipeline end to end, the regression check of (i), and fixing which
step each reduced arm is reported at; no model, rank or checkpoint is selected on any A9 case. The confirmation job is
submitted only after the development job has completed and this file records that nothing else changed.

### A8 selection record (2026-11-11 17:25 EST; development Re 95 only, step 1/3200; no Re 105 number of any stream bank exists)
Both stream-function arms completed (rb-s1b job 7136459, rb-s2b job 7136462) and both meet all six targets at rank 32
and at the prefixes 24 and 16. Rank-32 velocity error: s1 0.002425, s2 0.002249 (difference 7.8 %, beyond the 5 % tie
band), so the rule selects **s2**: `rb-s2b/results/train/trained_bank_eval.npz`, SHA-256
`39942eda6f0b0b39155bd8803a4aa17d6ac6964c8048e542f2e61e2f857b52e7` (`tasks/stream_frozen.env`). It is frozen from here
on. For the record, on the same data POD-Galerkin at rank 32 has velocity error 0.001867 (lower than either stream arm);
median divergence is 1.927e-4 for both stream arms at every rank against 2.42e-4 to 2.44e-4 for POD.
FE ladder, development (rb-fa job 7136516): 64 screen entries; the five cheapest passing ones were timed (5 reps). The
cheapest passing entry on Re 95 is mesh c2, P2/P1 (25,303 unknowns), step 1/300, refresh 30: median 35.99 s.
Next: `rb-sd` (stream development step ladder + paired development timing) and `rb-fb` (Re 105: stream bank once at
the four steps; FE candidates = 8 cheapest passing development entries plus the previous m0 comparator; paired timing).
