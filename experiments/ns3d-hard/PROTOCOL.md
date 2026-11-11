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
(none yet)
