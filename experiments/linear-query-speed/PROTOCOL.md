# Linear-problem query speed: protocol

Fixed on Sat 14 Nov 2026, before any timing or error of this lane exists.
Later changes are appended under "Amendments" with the date and the reason.

## Question

On the linear problems (Poisson 2D first; then Poisson 3D, heat 2D, heat 3D if time allows) how fast can
the online query of the trained reduced model be made by engineering the online path only, with the
trained model, the weak operator and the reduced solve unchanged and the error unchanged to the reported
digits?  The comparison is with a sparse direct solver that reuses its factorisation (cuDSS) and with
algebraic multigrid (AmgX), whose numbers were measured in the lane `sparse-solvers`
(`worktrees/2026-11-13-sparse-solvers/experiments/sparse-solvers/RESULTS-{h200,a100}.md`) and are quoted
unchanged.  No outcome is assumed.  If cuDSS is still faster at some mesh, that is the result.

## What is known before this protocol (no outcome of this lane)

- Paper Poisson 2D numbers (Table 1): jobs 4319401 (256^2), 4319399 (1024^2), 4319396 (2048^2),
  4319394 (4096^2), all on NVIDIA A100 80GB PCIe, lane `poisson2d-test` @ c72b1c51.  Stored values are
  extracted to `paper_reference/p2t*-stored.json` (per-case test errors, field hashes, median times).
  accurate = `R512_linear`, fast = `R128_linear`; GPU-query = `fused_device_seconds` (device source ->
  device float64 nodal field), median over 2 x 5 x 32 invocations.
- Stored stage profile of the paper path: the dense reconstruction u = G'[:, :R'] a dominates at every
  mesh above 256^2 (4096^2 accurate: projection + solve 0.66 ms, reconstruction 41.6 ms).
- cuDSS solve with a reused factorisation, Poisson 2D, ms at 256^2/512^2/1024^2/2048^2/4096^2:
  H200 0.448/1.0/2.96/9.36/36.5; A100 80GB 0.623/1.52/4.96/16.5/66.8.
- One local CPU inspection of the frozen bank only (no source, no solve, no error, no time): at n = 256 the
  rotated bank, viewed as an array (x, y, feature), has numerical rank about 85 (relative Frobenius tail
  1e-6) to 130 (1e-10) in each mesh direction, for 512 features.  This motivated variant T below.

## Frozen inputs

`paper_code/` is a byte-identical copy of the staged code of the paper jobs
(`poisson2d-test/runs/p2t256/code`, identical across the four runs except the config), including the
checkpoint `primary_K32.pkl`, `prep.npz` (rotation) and the four configs; hashes in `paper_code.sha256`.
Nothing in `paper_code/` is edited.  Cohorts are the paper's:
development = `dev12` (seeds 7090703 x 6 and 7090732 x 6), test = seed 2026092501 x 32.

## Step 1: reproduction on the timing GPU (gate before any timing is trusted)

The driver `lqs_p2d.py` builds the bank with the paper's `p2t_core.build_banks_host` (same config values;
`keep_orig` false, which only skips keeping the unrotated chunks) and the two Table-1 arms with the paper's
`pbk_core.make_trunc_linear` / `trunc_linear_query` and the paper's stage profile `make_linear_stages`,
all imported unmodified.  It is not the whole paper driver: the CG reference solver and the six other arms
are not run.  Gate `repro`: for every test case and both arms the same-grid error equals the stored value
to 1e-9 absolute.  The mesh 512^2 has no stored value (the paper did not run it) and is marked so.
Paper-path times on the new GPU are reported next to the stored A100 times.

## Step 2: stage profile

Per mesh and arm, separately jitted and synced: input projection + reduced solve; reconstruction; the fused
query; host -> device and device -> host copies; and for the new variants the three reconstruction pieces.

## Step 3: variants (trained model unchanged)

All variants keep the paper's contract: device float64 nodal source in, device float64 nodal field of shape
(n+1, n+1) out, plus the reduced coefficients a.  Same sine tests, same weak operator, same thin-QR
least-squares solution.

- P (paper): the paper kernel, unmodified.
- T (compressed reconstruction): offline, the mesh-evaluated rotated bank G'[(x, y), r] is compressed to
  G' ~ sum_ab U[x, a] V[y, b] C[a, b, r] (orthonormal U, V from a randomised range finder with 224 Gaussian
  probes, then a truncated higher-order SVD of the core to a relative Frobenius tolerance tau per arm).
  Online: u = U (sum_r a_r C[:, :, r]) V^T.  The reduced operator P = R^-1 Q^T diag(W) is precomputed, so
  a = P (S^T f S)[I, J], and the whole query is one jitted function with zero rows folded into U and V so
  no padding copy is made.  tau candidates: 1e-4, 1e-6, 1e-8.
- precision of the reconstruction: float64, or float32 factors and products with the result cast to
  float64 (projection and reduced solve stay float64).
- input: `full` reads every node of the source, or `sub-m` reads the source only on the nested sub-lattice
  with m intervals per side (the same sine-sum quadrature on a coarser lattice), m candidates 256, 512,
  1024 (only m < n).
- solve only: projection + reduced solve returning a, no field.  Reported in its own column, never as the
  query time.

Accuracy-affecting choices are tau, the reconstruction precision and the input lattice.

## Selection (development cases only)

Run on `dev12` only, at 256^2 and 4096^2 on an H200.  A variant passes if at both meshes and for both arms
the largest relative deviation of its field from the paper-path field of the same job,
||u_var - u_P|| / ||u_P||, over the 12 development cases is at most 1e-5 (so the relative error against
the reference changes by at most 1e-5, i.e. 0.001 percentage points, below the reported digits).
Two configurations are frozen, one per input class:
- F (full input): among passing variants with `full` input, the smallest median GPU-query time of the
  accurate arm at 4096^2; times within 2 % are ties, broken by float64 over float32, then smaller tau.
- S (sampled input): the same rule among passing `sub-m` variants.
The frozen pair is written here as an amendment before the test cohort is touched by any variant.

## Test evaluation (once)

One job per GPU type (H200; A100 80GB if a slot allows), meshes 256^2, 512^2, 1024^2, 2048^2, 4096^2, each in
its own process: reproduction gate, then in the same process A-B-A timing as in the paper (phase A1: paper
arms; phase B: frozen F and S and the float64/float32, tau = 1e-8 `full` ablation; phase A2: paper arms),
5 repetitions x 32 test cases per phase, random order within a case, 0.1 s burn before every invocation,
GPU-query = median `fused_device_seconds`.  The development gate (deviation <= 1e-5 on dev12) is
re-checked at every mesh before the test cases; a mesh where it fails is reported as a failure of that
variant, not retuned.  Reported per mesh and arm: worst and mean test error before and after, the largest
field deviation, median time before and after, stage profile, drift between A1 and A2.

Headline statement: "full-field output, full-source input" (configuration F) against the cuDSS solve on
the same GPU type, both float64 device-in/device-out.  S and solve-only are separate, labelled columns.

## What will be reported regardless of outcome

Every mesh where cuDSS or AmgX is faster; failed gates; variants that did not pass selection; and the
standing caveats: the comparators solve an arbitrary right-hand side to solver tolerance while the reduced
model is accurate to its model error on the trained source family; `sub-m` input relies on the source being
smooth at the sub-lattice scale; offline cost of the compression is reported.

## Resources

Cluster namespace `/cluster/tufts/paralab/tawal01/linear-query-speed-20261114/`, partition `gpu`, at most
two jobs queued or running, in-job preflight `jax_backend=gpu`.  GPU-hours are tallied in `LEDGER.md`.

## Amendments

### A1, Sat 14 Nov 2026 15:58 EST (before any outcome; no job of this lane has started)

Reason: the GPU queue estimated a five-hour wait per job, so selection and test cannot be separate jobs
within the time available.  The two first submissions (7220222, 7220223) were cancelled while pending and
produced nothing.

- Selection and test run in ONE job per GPU type.  The job runs `--mode select` (dev12 only), then
  `lqs_select.py` applies the selection rule above mechanically to those development-only outputs and
  writes `results/frozen.json` (its SHA256 is printed in the job log), and only then runs `--mode test`
  with that file.  No person sees a test number before the freeze, and nothing is chosen after it.
- Selection is therefore done per GPU type, by the same rule, on the GPU that is being timed.  The frozen
  pair may differ between the H200 and the A100 jobs; both are reported.
- Selection meshes: H200 256^2 and 4096^2 as fixed above; A100 80GB 256^2 and 2048^2 (the 4096^2 bank
  alone is 68.7 GB of the 80 GB device, so the 24-variant selection run is not attempted there; the timing
  mesh of the rule is then 2048^2).  The A100 test still includes 4096^2; if it does not fit, that is
  reported.
- If no `sub-m` variant passes, S is reported as "none passed" and the S column repeats F.
- On the H200 job, after the tests, the paper's whole unmodified driver (`paper_code/p2t_solve.py`, all
  arms and the CG reference) is also run at 256^2 and 1024^2 as a second, stricter reproduction.

Note, 16:02 EST: the heading of A1 first carried the time 16:10 by mistake; the amendment was committed
at 15:58 EST (commit fda0acbc1), before the two jobs 7220234 / 7220235 were submitted from that commit.

### A2, Sat 14 Nov 2026 17:25 EST (after the A100 job 7220235 finished; the H200 job 7220234 still running)

What happened: on the A100 80GB the test at 4096^2 failed before any query was run.  The offline
compression ran out of device memory in its second stage (the paper bank alone holds 68.7 GB of the 80 GB).
No 4096^2 number of any kind exists on the A100.  Tests at 256^2 to 2048^2 completed in that job.

Change (offline implementation only; no accuracy-affecting choice is touched): the driver gets the option
`--host-compress`, which does the second stage of the same compression (same probes, same seed, same
algebra) on the host instead of the device.  The frozen pair of job 7220235 is kept as it is
(`frozen/a100-80G.json`, SHA256 012561a1..., a copy of that job's `results/frozen.json`); one further A100
job runs the 4096^2 test only, with that file and `--host-compress`.  The development gate and the
reproduction gate are re-checked in that job as at every mesh.  If it fails again, 4096^2 on the A100 is
reported as not obtained.

Note, Sat 14 Nov 2026 17:43 EST: the heading of A2 carries the time 17:25 by mistake; A2 was committed at
17:12 EST (commit 45d737762), before job 7221131 was submitted (from commit cc7917c2e).

Record of the freezes (written after the test runs; nothing was chosen by hand):
- A100 80GB, job 7220235: F = `T_tau0.0001_f64_full`, S = `T_tau0.0001_f64_sub256`
  (`results/frozen.json` SHA256 012561a11790436b...).  Reused unchanged by job 7221131 (4096^2 only).
- H200, job 7220234: the same pair, F = `T_tau0.0001_f64_full`, S = `T_tau0.0001_f64_sub256`
  (`results/frozen.json` SHA256 9f04a36c32d880aa...).
- Every development gate and every reproduction gate passed at every mesh on both GPU types.
- The raw result JSON files and job logs of the three jobs are committed under `runs/` (no field arrays).

### A3, Sat 14 Nov 2026 17:45 EST: heat 2D (written before any heat number of this lane exists)

Poisson 2D is finished on both GPU types (apart from the paper's whole driver still running as a
cross-check in job 7220234).  The same offline compression is now applied to heat 2D.  Poisson 3D and
heat 3D are not attempted in this lane unless a later amendment says so.

- Paper path: lane `2026-09-23-heat-bank-knob` at cc92aca81, vendored unmodified in `paper_code_heat/`
  (`core.py`, `hbk_core.py`, `configs/h2d.json`, `prep_2d.npz`, `inputs/wide2d/`).  Arms of the paper's
  heat 2D rows: accurate `lin_R128_cn`, fast `lin_R48_cn` (linear rung, moment initialisation, reduced
  Crank-Nicolson composed offline, six output times).  Stored values: `paper_reference/h2d-stored.json`
  (job 4197350, H200, results.json SHA256 39f4cd8f...).
- Timing scope: the paper's heat scope, unchanged: supplied initial field on the device -> all six output
  fields on the device, blocked; no host copies for any subject.  0.1 s burn before every invocation,
  random order within a case, phases paper / variants / paper, 5 repetitions, median over cases x
  repetitions.  "Coefficients only" (the six reduced coefficient vectors, no field) is a separate column.
- New path (model, coefficients and output unchanged up to round-off and the compression tolerance):
  (i) the 256 moments of the initial field from two products with a sine table instead of a full
  transform; (ii) moments -> coefficients at the six times as one precomputed matrix; (iii) the six
  fields from the bank compressed in the two mesh directions (same routine, probes and seed as Poisson).
- Variants: tau in {1e-4, 1e-6, 1e-8} x reconstruction in {f64, f32}.  No sampled-input variant.
- Cohorts as in the paper: development = `validation_791001` (16 cases), test = `heldout_sealed_791099`
  (16 cases).  Development gate: max over cases and the six times of the relative L2 deviation of the
  new field from the paper-path field <= 1e-5, for both arms.
- Selection: on the development cohort only, at 4096^2, by the Poisson rule (fastest passing accurate-arm
  median; ties within 2 % -> f64, then smaller tau), applied by the driver itself, which writes and
  hashes `frozen-h2d.json` before the test cohort is touched.  The frozen variant is then used at 1024^2,
  2048^2 and 4096^2; tau = 1e-8 (f64 and f32) is timed alongside as an ablation, not selectable.
- Reproduction gate before any timing is trusted: the paper path's same-grid errors (six times, all 32
  cases) against the stored values, absolute difference <= 1e-9.
- GPU: H200 only (the paper's heat 2D numbers are from an H200).  Comparators (cuDSS, AmgX; the
  sparse-solvers lane's estimate for 20 Crank-Nicolson solves) are quoted, not rerun.
- The paper's own modal/transform full-order control is not a target.

(Clock times in the A3 heading and in the note above it were first typed from memory as 18:20 and 17:48;
they were corrected to the wall-clock times 17:45 and 17:43 in the next commit, before any heat job.)

### A3a, Sat 14 Nov 2026 17:48 EST (before any heat job has started)

Correction of A3: the paper's fast heat 2D arm is `lin_R32_cn` (stored test medians 0.336 / 0.829 /
3.966 ms, the 0.34 / 0.83 / 3.97 of the paper's table), not `lin_R48_cn` as A3 says (that is the lane
summary's rule-selected arm, not the one in the paper's table).  The arms are therefore accurate
`lin_R128_cn` and fast `lin_R32_cn`.  Job 7221749, submitted with the wrong fast arm, was cancelled while
pending and never started.  Nothing else in A3 changes.
