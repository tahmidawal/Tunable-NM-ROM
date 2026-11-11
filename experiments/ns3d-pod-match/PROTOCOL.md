# Confirmation protocol (rebuttal push, written 2026-11-11 BEFORE any outcome below was seen)

Known when this was written: the development-cohort numbers in
`reports/2026-11-03-ns3d-pod-match.md` (seed 202609202, 16 cases) and nothing else. No
coarse-mesh solver run, no interpolated-POD run and no test-cohort run of the coordinate
bank existed.

## Frozen items (no change after this commit; any change is an amendment below)
- Coordinate bank `banks/a2_mu001/bank_selected.npz` (sha256 a1af98e2...), full rank, frozen
  rotation, Lowdin at each mesh; 32^3 uses the orthonormality gate 1.0 (DESIGN A3).
- Coordinate-bank heads k=8: the three heads trained in the development jobs dev32b, dev64,
  dev96, copied to `frozen/` with sha256 in `frozen/frozen_settings.json`.
- POD reference heads: the paper's per-mesh frozen heads (ns3d-operators/frozen/h32, h64,
  ns3d-shift-head/frozen).
- Online solver: dt 0.02 (10 steps), 3 sweeps, damping 1e-6, 12 initial-condition sweeps,
  292 test modes. Span ladder 8,16,32,48,64,128,192,256 (all reported).
- Headline arms: `coordnet_head_k8` (accurate), `coordnet_span32` (most accurate span on
  dev at 64^3/96^3) and `coordnet_span8` (fastest). The rest of the ladder is reported too.

## Cases, truth, metric
- Development cohort: seed 202609202, 16 cases. Test cohort: seed 202609221, 32 cases,
  checked disjoint from every other cohort by the driver. The test cohort was opened before
  for the paper's POD head; for the coordinate bank and for everything new here it is opened
  ONCE, one job per mesh (32^3, 64^3, 96^3), after the development runs of the same driver
  have passed and this file is unchanged. A re-run is allowed only for a technical failure
  (crash, node fault, timing gate) with byte-identical code, configs and frozen inputs.
- Truth at mesh n: CNAB2 pseudo-spectral at mesh n, dt 0.001, horizon 0.2, 6 frames. This
  is the reference the ROM has always been scored against; every comparator is scored
  against the same arrays.
- Error: per frame, L2 norm of the difference over the L2 norm of the case's initial field;
  "worst" = max over cases and evolved frames, "median" = median over the same.

## Task 1: coarse-mesh solver control (accuracy-matched, mesh-free comparator)
- Same interface as the ROM query: input the initial field on the n-mesh, output 6 frames
  on the n-mesh. Query = Fourier truncation of the input to mesh m (coefficients with
  |k_i| <= m/2 - 1 copied, Nyquist planes dropped), the UNCHANGED solver `ns3d_fom.make_solver`
  on mesh m (its own 2/3 mask), zero-padding of the output spectra to mesh n, inverse FFT.
  The whole query is one jitted function.
- Ladder: m in {12,16,20,24,32,48,64} with m < n; schemes CNAB2 with steps
  {5,10,15,20,25,30,40,50,60,80,100,200} and RK4 (already in the solver file) with steps
  {5,10,15,20,30,40}. Unstable = nonfinite or worst > 100 %.
- Timing: same job, same GPU (A100 80GB), same precision (x64, matmul highest), in the
  SAME randomised interleaved block as the ROM arms and the same-mesh CNAB2 ladder, after
  burn-in, 7 repetitions, min/median/max reported. Two times per configuration: end-to-end
  (truncate + solve + pad; the headline, since it returns what the ROM returns) and
  solve-only (coarse input, coarse output; reported, even more favourable to the solver).
  Only configurations with worst <= 5 % are timed.
- Rule: for each ROM arm the comparator is the fastest (median, end-to-end) stable
  configuration whose worst error on the evaluated cohort is <= the arm's worst error.
  Reported ratio = comparator median / arm median (>1 means the ROM is faster). The same
  ratio with the solve-only time and the paper's same-mesh CNAB2 ratio are reported beside it.
  The comparator is selected on the evaluated cohort itself (it may use the test cohort to
  pick its setting; this favours the comparator, not the ROM).
- Whatever comes out is reported, including ratio < 1.

## Task 3: interpolated-POD control
- The paper's rank-64 centred POD built at 64^3 (parent recipe, gated against the stored
  64^3 probe), each mode Fourier-resampled to mesh n, Leray/2-3 masked, Lowdin
  orthonormalised. Reported at n = 32 and 96 (128): oracle-shift floor, span rollouts
  R' in {8,16,32,64} and the paper's 64^3 POD head used zero-shot on the resampled bank.
  No retraining (the point of the control is one bank + one head for all meshes).

## Task 4 (exploratory, not confirmatory)
- 128^3 with the frozen bank, the 96^3 coordinate head zero-shot, the resampled 64^3 POD
  bank + 64^3 head, CNAB2 at 128^3 and the coarse ladder, on the DEVELOPMENT cohort.

## Gates carried over
jax_backend=gpu in every log; POD rebuild gap <= 1e-8; timing sentinel gate 1.10; timed
outputs equal the accuracy-pass outputs (1e-9); the independent NumPy audit
(verify_coordnet.py) for the main arms at 32/64/96.

## Amendments
- A1 (2026-11-11, before any cluster job; seen in a 2-case local smoke at 32^3 only): a head
  used zero-shot at another mesh has its output multiplied by (n / n_trained)^{3/2}. Reason:
  bank columns have unit discrete-l2 norm, so coefficients of the same physical field scale
  as n^{3/2}; without the factor the zero-shot interpolated-POD head failed trivially (59 %)
  from normalisation alone, which would have been an unfair control. Applies to the
  interpolated-POD head (64^3 -> n) and to the 96^3 coordinate head at 128^3.
- A2 (2026-11-11, after the first development job cdev32 = 7117079, before any test job):
  (a) BUG FIX in the comparator's favour: the end-to-end coarse query used a scatter for the
  spectral zero-padding which cost ~220 ms regardless of the solve (solve-only: 2-20 ms).
  Replaced by slices + concatenation; identical output. The cdev32 end-to-end coarse
  timings and the ratios derived from them are void; its errors and solve-only timings stand.
  The other first-round development jobs (7117085, 7117092, 7117100) were cancelled.
  (b) Ladder widened in the comparator's favour after cdev32 showed a spatial floor
  (m=24 vs 32^3 truth: 2.1 %): coarse meshes are now {12,16,20,24,28,32,40,48,56,64,72,80}
  (m < n), and {24,...,96} at 128^3.
