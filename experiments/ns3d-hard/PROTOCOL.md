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
