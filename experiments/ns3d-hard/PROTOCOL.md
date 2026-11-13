# PROTOCOL — ns3d-hard (written 2026-11-11, before any outcome was seen)

Question. Does the paper's ROM (centred spatial bank x small nonlinear head with linear
skip, LSPG online solve) beat an accuracy-matched pseudo-spectral FOM that is free to pick
its own mesh and step count, on a 3D periodic Navier-Stokes family that is less smooth than
the paper's?

## 0. Common definitions
- Solver: experiments/ns3d/ns3d_fom.py (2/3-dealiased pseudo-spectral, Leray projection),
  unchanged, f64. Schemes: CNAB2 and RK4.
- Families (hard_fom.FAMILIES): the paper's parametrisation (centre U(0,1)^3, separation,
  strength U(0.6,1.4), log-uniform viscosity; two obliquely separated potentials
  exp(beta*sum(cos 2pi(x-c)-1))), generalised by the sharpness beta. Blob width scales as
  s=sqrt(2/beta); amplitude 0.2*s keeps the peak velocity; separation range (0.12,0.24)*s
  keeps the geometry.
  A (paper): beta 2, nu [2e-3,1e-2], T 0.2.  B: beta 8, nu [5e-4,2.5e-3], T 0.2.
  C: beta 16, nu [2.5e-4,1.25e-3], T 0.2.    D: beta 8, nu [2.5e-4,1.25e-3], T 0.4.
  E: beta 8, nu [1e-4,5e-4], T 0.4.
  In blob units B/C/D/E have 2 / 2.8 / 4 / 10 times the paper's Reynolds number and
  2 / 2.8 / 4 / 4 times its horizon. The candidate list is fixed here; no family is added
  after outcomes are seen (if one is, it is an amendment).
- Metric: per time, ||pred - truth||_2 / ||u0||_2 (full field); per case the maximum over
  the 5 evolved output times ("evolved worst"; t=0 reported separately); over cases the
  median and the worst. Six output frames at equal spacing over [0,T].
- Seeds: POD-screen train 2026111101, probe 2026111102 (characterisation only);
  main experiment train 2026111111 (first 128 cases build the bank, 512 train the head,
  every 8th is validation -> 448/64), validation-rollout cohort 2026111112 (64 cases),
  sealed test 2026111113 (32 cases, opened once, after everything is frozen).

## 1. Characterisation (Task 1)
- Reference: 192^3, CNAB2 dt 2.5e-4. Self-check on 2 probe cases against 256^3 dt 1.25e-4
  and against RK4 at 192^3; the reference is accepted if both differ by <= 2e-4 (a tenth
  of the tightest target); otherwise the family is flagged and the tightest target that is
  ten times the reference uncertainty is used.
- FOM ladder on 6 probe cases: meshes {16,24,32,48,64,96,128} x steps
  {5,...,640} x {CNAB2, RK4}; the FOM is started from the Fourier truncation of the
  reference initial field (better than sampling); its output is compared on the reference
  mesh as a trigonometric interpolant (error = resolved-mode difference + unresolved
  reference tail). Wall time: best of 3 warm calls per case, median over cases, includes
  the transforms in and out.
- "Cheapest FOM reaching x": smallest median time among configurations whose worst probe
  case is <= x, for x in {0.2 %, 1 %, 3 %}.
- POD screen: centred (energy-centroid co-moving frame) POD of 128 train cases x 6 frames
  at 96^3 (dt 5e-4), projection floor on 16 held-out probe cases at ranks 4..128.
- Gates for choosing the family for Tasks 2-4:
  (a) the cheapest FOM reaching 1 % (worst case) uses a mesh >= 64^3 (equivalently no mesh
      below 64 reaches 1 % at any step count);
  (b) held-out centred-POD floor at rank <= 64 is <= ~1 % (worst case, all six frames;
      "~" means up to 1.2 %).
  Choose the hardest family (order A<B<C<D<E) that passes both. If none passes both,
  report the trade-off curve (required FOM mesh vs POD rank needed) and stop.

## 2. Main experiment (Tasks 2-3), chosen family only
- Truth: the cheapest solver configuration that agrees with the 192^3 reference to
  <= 2e-4 on all probe cases (fixed from Section 1 results before any ROM is run),
  Fourier-truncated (|k|<n/2) to each evaluation mesh n. Evaluation meshes: 64, 96 and,
  if memory allows, 128.
- ROM (paper App. A.5 recipe, per mesh): centred rank-64 POD bank from 128 train cases x 6
  frames; importance rotation; span widths {64,48,32,16,8}; head = auto-decoder MLP
  (2x256 SiLU + linear skip), k in {8,16,32}, 100k full-batch Adam steps, 512 trajectories
  x 21 frames, every 8th case validation; implicit midpoint, Gauss-Newton sweeps, shift
  unknown. Allowed choices, all made on validation rollouts only: k, number of weak test
  modes M (default 292; larger allowed because sharper fields need more tests), step count
  and sweep count. The span arms at width R' (same weak form, linear) are the
  POD-LSPG comparators at latent dimension R'; a POD-Galerkin arm (tests = bank) at the
  same widths is run if the code path is available.
- FOM comparator: CNAB2 or RK4 at ANY mesh (16..128) and step count, started from the
  truncated exact initial field, output transferred spectrally to the evaluation mesh
  (time reported with and without the transfer; the headline uses the time WITHOUT the
  transfer, i.e. the comparator is not charged for it). Accuracy-matched means: the
  cheapest configuration whose worst-case error on the same cohort is <= the ROM arm's
  worst-case error. Configuration chosen on validation, then timed and scored on test.
- Timing: same job, same GPU, f64, warm-up excluded, single trajectory per call,
  interleaved repetitions (>= 7 rounds), median and min-max reported. ROM time includes
  the initial-code solve and reconstruction of the six output fields on the mesh.
- Headline statement: speed ratio = (accuracy-matched free FOM time)/(ROM time) at the
  ROM's test worst-case error; a ratio <= 1 is reported as a loss.
- Optional: FNO trained on the same 448 trajectories until validation plateaus.

## Amendments
A1 (2026-11-11, before any ROM or main-experiment outcome; characterisation jobs still running).
Implementation details of Section 2 fixed in pipeline.py:
- Quadratic tensor: built on a zero-padded mesh so products of bank fields (content up to
  |k|<n/2 after truncation of the reference) are alias-free at the test wavenumbers. Reason:
  the lane's stock builder validates against the 2/3-masked on-mesh product, which is not the
  weak form of the truncated truth; the padded build is the exact weak form.
- Test-mode ladder M in {292,584,1168,2336}; steps in {10,20}; sweeps 3; head k in {8,16,32}.
- Selection rule on the 64-case validation-rollout cohort (seed 2026111112), per mesh and per
  arm: take the smallest evolved worst-case error; among configurations within 5 % of it choose
  the cheapest by (steps, M, k). Frozen in frozen.json before the test cohort is generated
  into any ROM job.
- FOM ladder: meshes {32,48,64,80,96,112,128} x CNAB2 steps {5..640} and RK4 steps {5..160};
  scored against the truth truncated to each evaluation mesh (the part of the truth a coarse
  FOM cannot represent counts as its error, same as for the ROM). Two matched comparators are
  reported: chosen on validation (protocol headline) and chosen on the test cohort itself
  (most favourable to the FOM); both timed in the test job, interleaved with the ROM arms,
  3 timing cases x 9 rounds.
- POD-Galerkin arm: not available in the reused code path; POD-LSPG span arms only.

A2 (2026-11-11, after characterisation of A, B, C; D and E still running; before any ROM run).
Family choice by the Section 1 gates, numbers from results/char_*/characterise.json:
A fails gate (a) (1 % worst-case reached at 32^3). B passes both: cheapest 1 % FOM is
CNAB2 96^3 x 80 steps; held-out centred-POD floor at rank 64 is 1.09 % worst / 0.31 % median
(inside the pre-declared "up to 1.2 %" band, but note it is at the edge). C fails gate (b)
(rank-64 floor 2.3 % worst) and its 192^3 reference missed the 2e-4 self-check. D and E are
harder than C in Reynolds number and their references already missed the self-check on the
first probe cases, so they are excluded whatever their POD floor is (their numbers are
still reported in the trade-off table). Chosen family: B.
Truth for B: the 192^3, dt 2.5e-4 CNAB2 reference itself (self-check <= 4e-5 against 256^3 and
RK4) rather than a cheaper configuration; reason: saves a search, costs only generation time.

A3 (2026-11-11, 11:00 EST, lane restarted after an interruption; written before any ROM
outcome). Known at this point: the characterisation tables, the bank operator checks
(results/bankB/bank.json) and the development FOM ladder fomdevB (job 7118716; its cost/error
front at 64/96/128 evaluation: 3.4 % at 28 ms, 1.6-1.8 % at 75 ms, 0.6-0.9 % at 207 ms).
NOT known: any head fit, floor on the main cohorts, or ROM rollout (romdev was never run).
1. Error target. There is no absolute target: each ROM arm's achieved worst-case error on
   the evaluated cohort is the level its comparators must match (Section 2). The rank-64
   POD floor is 1.09 % worst on the probe cases, so no arm can reach 1 % and none is claimed.
   Usefulness band, declared now: worst error <= 3 % "useful" (3 % is the loosest level of the
   characterisation), 3-5 % "marginal", > 5 % "not useful". Ratios are reported for every arm;
   a win is claimed only for an arm that is useful on the test cohort.
2. ROM step ladder {10,20,40} (40 added to A1's {10,20}). Reason: family B needs about 80
   CNAB2 steps for 1 %, so the second-order implicit-midpoint ROM may be time-limited at 20.
3. Two head arms are frozen per mesh by the A1 rule: `head` (slack 1.05, as A1) and
   `head_fast` (same rule with slack 1.25: the cheapest (steps, M, k) within 25 % of the
   smallest worst error). Reason: the comparison is a cost/accuracy trade-off and the most
   accurate setting alone hides the cheaper end. Both are reported whatever happens. All span
   widths {64,48,32,16,8} are frozen and reported as before.
4. Controls. In this lane the bank IS the per-mesh centred POD, so the span arms are the
   linear POD-LSPG controls at latent width R' (span64 = the full POD bank) and the head-vs-span
   difference isolates what the nonlinear head adds. Added, as in the sibling lane
   (ns3d-pod-match): the interpolated-POD control at 96^3 and 128^3 = the 64^3 bank
   Fourier-resampled to mesh n (exactly orthonormal after the factor (64/n)^{3/2}), its own
   weak operators, the 64^3 rotation, the 64^3 frozen settings and the 64^3 heads zero-shot
   with output scale (n/64)^{3/2}. No retraining, no selection at n.
5. Comparator ladder, refined in the comparator's favour (the fomdevB front is coarse):
   meshes {32,40,48,56,64,72,80,88,96,104,112,128}; CNAB2 steps
   {5,10,15,20,30,40,50,60,70,80,100,120,160,240}; RK4 steps {5,10,15,20,25,30,40,60,80}.
   Dropped relative to A1: CNAB2 320/640 and RK4 120/160 (never the cheapest at any error
   >= 0.2 % in fomdevB). On the development cohort the candidate set is the union of this
   ladder and fomdevB; on the test cohort it is this ladder, run inside the test job.
6. Comparator rule (as the sibling lane, replaces the two-comparator wording of A1). For each
   arm, eligible = stable ladder configurations whose worst error on the SAME evaluated cohort
   (truth truncated to the evaluation mesh) is <= the arm's worst error; the comparator may
   therefore pick its setting on the test cohort itself (this favours the comparator).
   - any-mesh comparator: the fastest eligible configuration; same-mesh comparator: the
     fastest eligible configuration with m = n (CNAB2 or RK4).
   - Two times per configuration: solve-only (initial field already on mesh m, output left on
     mesh m; PRIMARY, as in Section 2 "not charged for the transfer") and end-to-end (storage
     mesh initial field -> truncate to m -> solve -> spectral transfer to n; slices and
     concatenation, no scatter). Ratio = comparator median / ROM median; <= 1 is a loss.
   - Shortlist for the timing block: per arm the 4 fastest eligible configurations by the
     ladder's own timing (any mesh), the 2 fastest with m = n, and on the test cohort also the
     configuration chosen on the development cohort (reported as `dev_chosen`, with its test
     error). The comparator is the fastest of the shortlist by the in-block median.
   - Timing: one job per cohort, A100 80 GB, f64, matmul highest; for each of 3 timing cases
     one randomised interleaved block (9 rounds after 2 untimed calls) holding every ROM arm
     and every shortlisted solver configuration; medians pooled over the 3 cases, per-case
     ratios and min/max also stored. Timing gate as the sibling lane: sentinel arms (head,
     span64) timed alone before and after the block must agree with their in-block median
     within a factor 1.10; a failed block is repeated inside the same job (at most 3 attempts,
     every attempt stored). Timed outputs must reproduce the accuracy-pass errors (ROM 1e-9,
     solver 1e-7 against the ladder's per-case worst error).
7. Verdict rule. Per evaluation mesh, the ROM "beats the solver" only if the frozen `head`
   or `head_fast` arm is useful on the test cohort AND its any-mesh solve-only ratio is > 1
   there; the end-to-end ratio, the same-mesh ratio and the span/interpolated controls are
   reported beside it. If the best span arm does as well as the head arms, the claim is
   "a POD-LSPG ROM beats the solver", not "the nonlinear head does".
8. Sealed test cohort (seed 2026111113, 32 cases): opened once, by one job that runs the
   ladder and all three meshes with the frozen files, writes its JSON incrementally and has a
   wall-time request several times its expected duration. A re-run is allowed only for a
   technical failure (crash, node fault, failed gate) with byte-identical code and inputs.
   No family, arm or setting is changed after the test job is submitted.

A4 (2026-11-11, 11:45 EST; after the technical rehearsal job 7133721 = the new compare stage on
the development cohort at 64^3/96^3 with the OLD ladder fomdevB; before the development
comparison that is reported and before the sealed test). The rehearsal's outputs agreed
(ROM 0, solver 1e-13) but the A3 timing gate failed in all 3 attempts at 64^3: the ROM
sentinels were reproducible alone (before/after within 1 %) yet 5-17 % SLOWER inside the
interleaved block (head 14.5 ms alone, 15.2-16.9 ms in block; span64 17.3 vs 18.1-19.3 ms).
This is an interleaving cost on an idle GPU, not contention, and it is charged to the ROM:
all reported ROM times and ratios remain the in-block medians (the slower number).
Change to the gate only: (i) sentinel alone-before vs alone-after within 1.10 (unchanged;
detects drift/contention); (ii) in-block / alone-before within [1/1.10, 1.25] (upper bound
relaxed from 1.10); (iii) two solver sentinels are added (the cheapest and the dearest
shortlisted solve-only configuration of the block) under the same rule, so that a solver that
is slowed by interleaving more than the ROM is visible. Nothing else changes; the rehearsal
ratios (13x and 23x at 64^3 against the old ladder) are not reported as results.

FREEZE (2026-11-11, 12:08 EST). Development comparison cmpdevB (job 7134656, commit 2155a978b,
status final, no failures) is in results/cmpdevB/compare.json. Frozen for the sealed test: the
code at this commit, results/romdevB_{64,96,128}/frozen.json with their head and rotation
files on the cluster, the interpolated bank interpB (job 7133684), the A3 ladder, the A3/A4
rules. The sealed-test job is submitted from this commit with the same command as cmpdevB
except: cohort genB_test, ladder run in-job, --devcompare cmpdevB.

TEST OPENED (2026-11-11, 12:14-12:57 EST). Sealed cohort opened once by job 7134832 (commit
9d2a8bba8, the freeze commit): status final, no failures, gate passed at the first attempt at
every mesh. Results: results/cmptestB/{compare.json,ladder.json}, results/table_test.md.
No rerun, no change of any setting afterwards.

=====================================================================================
LANE 2026-11-11-ns3d-hard-rank (branch exp/2026-11-11-ns3d-hard-rank, forked from the
TEST OPENED commit 4560156b2; own cluster namespace rebuttal-ns3d-hard-rank). Everything
above is the parent lane and is not changed. The parent's sealed test cohort (seed
2026111113) is NOT read by any job of this lane.

A5 (2026-11-11, 13:55 EST; written before any outcome of this lane). Known: all parent
results above (dev and sealed test at rank 64; dev floors 1.35-1.39 % at rank 64) and the
probe-case projection floors of results/char_B (1.09 / 0.50 / 0.38 % worst at ranks
64 / 96 / 128). NOT known: any rank-96/128 bank check, head fit, floor on the dev cohort,
rollout, or any number on the new cohorts.
Purpose: (A) lower the error of the family-B result with larger banks, (B) three
independent replicates, (C) one confirmation on a fresh cohort that nothing has touched.
1. Arms at rank r in {96, 128}. Per mesh a rank-128 centred POD bank is built from the same
   128 bank trajectories with the unchanged bank stage; the rank-r bank is its first r
   columns (the bank is the QR of the POD prefix, so prefix spans are the POD spans) and the
   operators are the corresponding slices of A, T, Dd. M_max = 2336 as before.
   - head arm: the unchanged head (2x256 SiLU + linear skip, 100k full-batch Adam steps,
     every 8th of the 512 head trajectories for validation), k in {8, 16}, trained on the
     first r coefficients of the head trajectories.
   - control: the rank-matched linear span arm (POD-LSPG on all r bank columns; no rotation,
     since the full-width span does not depend on it), same settings ladder.
   - reference: the parent's frozen rank-64 arms (head, head_fast, span64), files untouched.
   - contingency, fixed now: if at some rank the dev head floor (worst) is above 1.15x the
     dev span floor of that rank for BOTH k, heads of width 512 with 200k steps (k 8 and 16)
     are trained for that rank and join the selection pool. No other architecture change.
2. Order of work: bank operator checks, then dev projection floors and head floors, then dev
   rollouts. Dev ladder: M in {584,1168,2336}; steps in {10,20,40,80} with 3 sweeps, and
   5 sweeps at steps {20,40}. Selection on the 64-case dev cohort (seed 2026111112) per mesh,
   per rank and per arm kind by the A1/A3 rule: smallest evolved worst error; among settings
   within the slack choose the cheapest by (steps x sweeps, M, k, width). `head`: slack 1.05;
   `head_fast`: slack 1.25; span control: slack 1.05. A further dev-only refinement of the
   ladder is allowed only through a dated amendment written before it is run.
3. Goal (not a gate): worst rollout error clearly below 0.7 % with any-mesh solve-only ratio
   above 1. Both ranks are carried to the fresh cohort and reported whatever the dev outcome;
   no arm is dropped or picked after the fresh cohort is opened. Bands and verdict rule as
   A3.1/A3.7, applied per arm.
4. Comparator ladder, extended only (superset of A3 so that the solver can be matched at
   lower errors): meshes A3 + {100,108,120}; CNAB2 steps A3 + {90,140,200,320}; RK4 steps
   A3 + {50,100,120}. Dev cohort: union of fomdevB, fomdevB2 (parent jobs, read-only) and a
   new job with the added configurations. Fresh cohort: the whole extended ladder on all its
   cases (sharded jobs, per-case errors merged; shortlist timings from the first shard). The
   solver ladder contains no ROM quantity and no selection reads it, so the fresh-cohort
   ladder jobs may run before the freeze. Comparator rule A3.6 and gates A4 unchanged, except
   timing cases: 8 per cohort (dev: 0,8,...,56; fresh: 0,16,...,112) instead of 3; ROM
   sentinels r64_head and r128_head.
5. Replicates (B). Replicate 0 = the cohorts above (train seed 2026111111, head seeds
   20260923+k); its rank-64 arms are the parent's frozen files exactly. Replicates 1 and 2
   are fully independent: train seeds 2026111131 and 2026111141 (first 128 trajectories ->
   rank-128 bank, 512 -> head data), head seeds 20260923+1000s+k; rank 64/96/128 banks are
   prefixes of their bank; heads are trained only at the frozen (rank, k, width) of each
   frozen arm, with the frozen (M, steps, sweeps). No selection of any kind on replicates
   1, 2; their dev numbers are reported. All settings are selected on replicate 0 only, so
   replicate 0 is the headline and the three are reported side by side (every seed, min /
   median / max; no best-seed choice).
6. Fresh cohort (C): 128 new trajectories, family B unchanged, same truth solver, seed
   2026111151. No ROM job reads it before the freeze entry below is committed. It is opened
   once, by one compare job per replicate submitted together from the freeze commit (same
   code, same ladder files). Rerun only for a technical failure with identical code/inputs.
7. Reported whatever happens: dev and fresh tables per mesh and arm, replicate spread,
   failures, job ids, offline cost from the scheduler's accounting.

A6 (2026-11-11, 19:45 EST; written before any rank-192 bank, head, floor or rollout exists).
Known (replicate 0, dev cohort only; results/rdev_s0_{64,96,128}/romdev.json, jobs 7141854,
7141858, 7141859): dev projection floors (worst) 0.79-0.81 % at rank 96 and 0.61-0.62 % at
rank 128; head floors within 1.08x of them (contingency of A5.1 not triggered at any mesh or
rank); selected head rollouts 0.85-0.87 % (rank 96) and 0.66-0.71 % (rank 128), i.e. at the
bank floor. The A5.3 goal "clearly below 0.7 %" therefore cannot be met by any setting at
rank <= 128: the limit is the bank, not the head, the step or the test count. The linear
span control does not roll out stably at these ranks (best dev worst 10.6-18.5 %, growing
with the step count); it stays in the tables as the pre-registered control. No comparison
against the solver ladder has been run yet; nothing on the fresh cohort has been read.
The A5 selections for ranks 64/96/128 (frozen_rank/frozen_{64,96,128}.json as produced by
freeze_rank.py from the three files above) stand and are not revisited.
Added arm, same rules: rank 192.
1. Per replicate a rank-192 centred POD bank from the same 128 bank trajectories (unchanged
   bank stage, M_max = 2336), head data against it, heads k in {8, 16} (width 256, 100k
   steps), rank-matched span control, the A5.1 contingency, the A5.2 dev grid and selection
   rule, all unchanged. Arms r192_head, r192_head_fast (if different), r192_span are added
   to the frozen files; ranks 96/128 keep using the rank-128 bank and their trained heads.
2. Replicates 1, 2: rank-192 bank and heads at the frozen (k, width, M, steps, sweeps), no
   selection, as A5.5. Fresh cohort: r192 arms join the single compare job of each
   replicate (A5.6); sentinels unchanged.
3. Rank 192 is reported whatever its outcome (including a failure of the bank stage, an
   unstable rollout, or a solve-only ratio below 1). No further rank is added after this.
4. A dev comparison of the A5 arms against the dev ladder may run before the rank-192
   selection (it performs no selection); the dev comparison handed to the fresh-cohort jobs
   as the A4 timing reference is the one that contains all frozen arms.

FREEZE (2026-11-11, 22:05 EST; rank lane, A5/A6). Written before any ROM job has read the
fresh cohort (seed 2026111151; only the solver-ladder jobs 7137078 and 7137196 have run on
it, as A5.4 allows). Frozen, from replicate 0 on the dev cohort only:
- arms and settings: frozen_rank/frozen_{64,96,128}.json of this commit, produced by
  freeze_rank.py from results/rdev_s0_*/romdev.json (ranks 96, 128; jobs 7141854, 7141858,
  7141859) and results/rdev192_s0_*/romdev.json (rank 192; jobs 7143621, 7143623, 7143624)
  and the parent's frozen rank-64 files. Adding rank 192 left every rank-64/96/128 entry
  unchanged. All selected heads are k = 8, width 256; contingency not triggered anywhere.
  Dev worst error of the selected heads (meshes 64/96/128): rank 64 1.40/1.45/1.45 %,
  rank 96 0.85/0.85/0.87 %, rank 128 0.67/0.66/0.71 %, rank 192 0.50/0.56/0.59 %.
- banks/heads: replicate 0 bank128_s0b + rdev_s0_*, bank192_s0 + rdev192_s0_*, parent bankB
  + romdevB_* for rank 64; replicates 1, 2 bank128_s{1,2} + rrep_s{1,2}_* (ranks 64-128),
  bank192_s{1,2} + heads trained after this entry at the frozen settings (no selection).
- comparator: merged extended ladder fomF_0 + fomF_64 (450 configurations, 128 cases),
  rule A3.6, gates A4, timing cases 0,16,...,112, 9 repetitions, sentinels r64_head and
  r128_head, dev reference = the all-arm dev comparison of replicate 0 run from this commit.
- code: rank.py / pipeline.py of this commit; no code change between this commit and the
  three fresh-cohort compare jobs, which are submitted together.
Known at freeze (dev only): devcmp1 (job 7142556) any-mesh solve-only ratios 3.2-9.6x for
the rank-64/96/128 heads; replicate 1, 2 dev errors for ranks 64-128. Nothing else.

NOTE (2026-11-12, 01:36 EST; rank lane; fresh cohort opened, timing reruns). The three
fresh-cohort compare jobs were submitted together from commit 07ca046b8 and read the cohort
once each: replicate 2 (job 7144218, node pax105) status final; replicate 0 (job 7144215,
pax106) failed-gates at 64^3 only; replicate 1 (job 7144216, pax106) failed-gates at all
three meshes. All errors are final (128 cases; timed ROM outputs agree with the evaluated
ones to 0, solver to 3e-12). Every gate violation, in every attempt, is the same sentinel:
the cheapest shortlisted solver (rk4 32^3 x5, 2.2-2.4 ms alone) ran 1.25-1.39x slower
inside the interleaved block (limit 1.25); every ROM sentinel (max in-block ratio 1.095,
before/after within 1.10) and the dearest solver sentinel (max 1.013) passed everywhere.
That solver is matched only to the unstable linear-span controls (ratios 0.04-0.15x), and
a slower in-block solver would raise, not lower, a ROM ratio. Decision, taken before any
rerun: this is a timing-measurement failure, so A5.6 allows one rerun per failed job with
identical code and inputs (no code file changed since 07ca046b8; same arguments; new job
directories fresh_s0b, fresh_s1b; node pax106 excluded in the scheduler request only). The
gate is not changed. Errors are deterministic and must reproduce; both the failed-gate and
the rerun timings are reported, and if a rerun fails the gate again it is reported as
failed with no further rerun.

OUTCOME of the timing reruns (2026-11-12, 04:15 EST). fresh_s0b (job 7145062, pax105):
status final, all gates passed at the first attempt; errors equal the first run to 8e-9.
fresh_s1b (job 7145063, pax049): failed-gates again at 64^3 and 96^3 (3 attempts each),
passed at 128^3 on attempt 2; errors equal the first run to 3e-9. The violator is again
only the rk4 32^3 x5 solver sentinel (in-block 1.34-1.39x; ROM sentinels <= 1.085). No
further rerun. Replicate 1 timings at 64^3/96^3 are reported as gate-failed measurements.

=====================================================================================
LANE 2026-11-13-ns3d-flat-timing (branch exp/2026-11-13-ns3d-flat-timing, forked from the
rank lane at 64ed589c9; own cluster namespace rebuttal-ns3d-flat-timing). Everything above
is the source lane and is not changed. Source-lane cluster directories are read only.

T1 (2026-11-13, 00:55 EST; written before the job below is submitted). ADDITIONAL
TIMING-ONLY RERUN of the replicate-1 fresh-cohort comparison, BEYOND the one rerun that
A5.6 and the 2026-11-12 NOTE permit. Disclosed as such wherever its numbers are used.
Reason: fresh_s1 (job 7144216) and its permitted rerun fresh_s1b (job 7145063) both failed
the A4 timing gate at 64^3/96^3 on shared nodes, and in every attempt the sole violator was
the cheapest solver sentinel (rk4 32^3 x5, ~2.3 ms) being slowed inside the interleaved
block, i.e. a node-contention symptom. This rerun asks the scheduler for an exclusive node.
- Code and inputs identical: python files byte-identical to the freeze commit 07ca046b8
  (verified by git diff); same arguments as fresh_s1b (cohort genF_0+genF_1, bank128_s1,
  rrep_s1_*, bank192_s1, rrep192_s1_*, ladders fomF_0+fomF_64, devcompare devcmp2, reps 9,
  timing cases 0,16,...,112, shortlist 4, gate 1.10, gate-block 1.25, sentinels r64_head,
  r128_head); frozen_rank/ files identical. Only the job directory (new namespace) and the
  scheduler request (--exclusive, A100 80 GB) differ.
- The gate is not changed. Errors must reproduce fresh_s1/fresh_s1b (deterministic).
- Reported whatever happens, beside fresh_s1 and fresh_s1b; a gate failure is reported as
  such and there is no further rerun. Replicate 1 errors are unaffected and were never in
  question.

T2 (2026-11-13, 01:45 EST; written before any split-timing job is submitted). SPLIT TIMING of
the frozen replicate-0 head arms r64_head, r96_head, r128_head, r128_head_fast, r192_head,
r192_head_fast (frozen_rank/ files and heads exactly as fresh_s0b, job 7145062). Code:
split_timing.py (new; imports pipeline.py / rank.py unchanged).
1. Stages of one query, each a separately jitted function, inputs precomputed on device:
   ENC (energy centroid, centring shift, projection G^T u0; grid sized), FIT (nearest stored
   code + code fit; reduced), SOLVE (nu-dependent operator preparation and every time step in
   reduced coordinates; nothing grid sized), DEC (head evaluation, G a and shift for the six
   output frames, frame by frame as in the source query), and FULL (the source query
   unchanged = what Table G1 reports). DECB = DEC with one batched matmul over the six frames
   (an implementation alternative; reported separately, never substituted into FULL).
   Check: ENC->FIT->SOLVE->DEC (and DECB) reproduce FULL to <= 1e-9 relative.
2. Meshes 64, 96, 128 (native banks, per-mesh frozen settings) and 192, 256. Above 128 no
   bank, head or setting exists. Construction there (the A3.4 interpolated-POD rule of the
   source protocol, with source mesh 128): each arm's 128^3 bank Fourier-resampled to n and
   scaled by (128/n)^{3/2} (exactly orthonormal); reduced operators assembled with the lane's
   build_ops (padded weak form, dense-test validation) in job(s) `asm`; the 128^3 heads with
   output scale (n/128)^{3/2}; the 128^3 frozen (M, steps, sweeps). No training or selection
   at 192/256. The asm job also reports the difference between these operators and the 128^3
   ones (A, Dd, lam equal; T equal after the factor (n/128)^{3/2} if the reduced system is
   mesh invariant).
3. Solver references: per arm and mesh, the any-mesh solve-only and same-mesh comparators that
   fresh_s0b selected at that mesh (for 192/256: the 128^3 any-mesh configuration, solve-only
   and end-to-end to n; same-mesh = the 128^3 same-mesh scheme and step count run at mesh n,
   accuracy not selected at n, its error on the timing cases is reported). No new selection.
4. Timing protocol as the source (A3.6/A4/A5.4): fresh cohort (seed 2026111151) timing cases
   0,16,...,112; per case one randomised interleaved block (2 untimed calls, 9 rounds) holding
   every stage of every arm and every reference solver; medians pooled over the 8 cases; A4
   gate with ROM sentinels FULL r64_head / FULL r128_head and the cheapest/dearest reference
   solver (ordered by one warm call, not by the ladder); up to 3 attempts, all stored. If the
   arms' bases do not fit in 60 % of device memory at a mesh, one block per bank (r64; r96 and
   r128 arms; r192 arms) is timed instead, each with its own references and gate (recorded).
5. GPUs: job split_a100 on A100 80 GB (the fresh_s0b GPU type) at 64/96/128/192; job
   split_h200 on H200 at 64/96/128/192/256 (256^3 x rank 192 does not fit an A100 in f64).
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 in these jobs only (memory, not speed).
6. Errors: only a sanity check on the 8 timing cases (FULL query vs the source per-case
   errors at 64/96/128, which must agree to ~1e-9; at 192/256 vs the 128^3 errors).
7. Reported: per arm and mesh ENC, FIT, SOLVE, DEC, FULL, the sum, the references; flatness
   of SOLVE as max/min over meshes (frozen settings differ by mesh for r64_head, r128_head,
   r192_head; the other three arms have identical settings at every mesh).
T2 addendum (2026-11-13, 02:05 EST; before any split job ran). The cluster user cap (12 GPUs)
is full with other lanes' jobs and every A100 80 GB GPU is allocated, so the separate asm jobs
were cancelled before starting (7161726, 7161729). Assembly now runs inside each split job
(at 192^3, all three banks; failures recorded, not fatal). At 256^3 the operators are NOT
assembled: they are the 128^3 operators with T multiplied by (128/256)^{3/2}, used only if the
in-job 192^3 assembly confirms the invariance (A, Dd, lam equal and T equal after the factor
to <= 1e-10). Reason: assembling the rank-192 tensor at 256^3 needs ~270 GB host and ~100 GB
device memory, which no free node offers. Requests use 3 CPUs (the free H200 slot has 3).
