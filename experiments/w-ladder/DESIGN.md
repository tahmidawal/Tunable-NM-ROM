# w-ladder — on the reflective 2D wave, is the correction ladder's top rung (linear evolution of the full learned bank) both the most accurate and the cheapest operating point?

Pre-registered before any job was submitted. Amendments are appended to
[§10 Amendments](#10-amendments) with date and reason, never edited in place.

Worktree `worktrees/2026-09-17-w-ladder`, branch `exp/2026-09-17-w-ladder`, forked from the
consolidated baseline `exp/2026-09-13-nmrom-consolidated` at `02ff0f1f`. Cluster namespace
`/cluster/tufts/paralab/tawal01/w_ladder_20260917/`. Nothing is merged; nothing is pushed.
Every wave result before the 2026-09-06 evidence reset is untrusted and is not cited here.

---

## 1. The question

The paper's claim under test: *the accuracy–cost trade from correction rank $q$ exists only
where the projected residual is nonlinear in the bank coefficients; on a linear PDE the top
rung ($q = R$, the bank evolved linearly with no head) is itself a linear reduced model and is
also the cheapest point, so the curve is degenerate.* Heat established this
(2026-09-10, job 3511417: linear bank 0.56 ms / 1.68 % vs nonlinear head 12.3 ms / 4.56 % at
$1024^2$). The wave equation is the second linear case, non-dissipative, and reviewer GwrW
asked for waves by name.

What the fresh post-reset wave archive already says, at $64^2$ on two opened cases only
(accel07, job 3563590, and the retained confirmation accel12, job 3565786):

| arm | internal dimension | GPU ms | worst energy-state error % | all-state 5 % |
|---|---:|---:|---:|:---:|
| `chol_guard` (head, $K=32$, $q=0$) | 32 | 178.7 | 6.223 | fail |
| `trained_nested40` (head $+$ 8 fixed linear directions, $q=8$) — **the retained selection** | 40 | 199.5 | 5.041 | fail |
| `linear_bank64` (exact modal evolution of all 64 bank coefficients, no head) | 64 | 1.447 | 3.211 | **pass** |

So the degenerate curve is *already visible* at one mesh on two cases, but it was labelled a
"changed method" and never placed on a ladder, never confirmed at $256^2$/$1024^2$, never
compared with POD at matched rank, and its energy behaviour was never certified. This lane
does that. **The nonlinear head is not tuned to win; it is measured as retained.**

## 2. What is frozen (verbatim from the consolidated baseline)

| item | value |
|---|---|
| bank | fresh learned reflective bank, $R = 64$, `runs/accel12/cluster/in/dirichlet/bank_parameters.npz` + `coordinates.npz`, rebuilt on each mesh by `pilot.rebuild` (mass-QR change of coordinates, parity audits $\le 10^{-9}$) |
| head, $q=0$ | `head32.npz` + `initializer32.npz` (`frozen_mlp32_seed691200`, $K=32$), the `chol_guard` arm of accel12 |
| head, $q=8$ (retained selection) | `trained_nested40.npz` + `initializer_trained_nested40.npz`: $h_{40}(z,y)=h_{32}(z)+B_8y$, $B_8$ = *scaled* training-PCA columns 32:40 |
| integrator (head arms) | RK4, $\Delta t = 0.01$ (240 steps, 960 stage evaluations), guarded Cholesky least-squares (`acceleration.rollout`, variant `chol_guard`), rank guard $10^{-8}$ |
| initial fit | `pilot.cold_fit`: 8 starts (affine, zero, six fixed training codes), 800 trust-region LM iterations, select by objective; tangent LS velocity |
| horizon / outputs | $T = 2.4$, 49 observations every 0.05 |
| truth | same-grid exact semidiscrete DST (`pilot.spectral_propagate`), the retained reference kind |
| metrics | `pilot.metrics`: time-maximum errors; displacement / initial displacement $L^2$ norm; velocity and energy-state / $\sqrt{2E_0}$; current-relative variants retained |
| numerics | float64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, `jax_backend=gpu` asserted, frozen-math SHA256s asserted |
| cohort | **development only**: `opened` seed 690602 indices 0–3 and `fresh_development` seed 691115 indices 0–3 (8 cases). Indices 0–1 of each are the 4 cases of accel12 and are the value-gate subset. The final cohort (seed 690603) stays sealed. |
| meshes | $64^2$, $256^2$, $1024^2$ intervals; **one job per mesh** |
| timing | 3 timed repetitions after a warm-up, 0.3 s GPU burn before every timed call, arm order alternated between repetitions, device-resident inputs to device-resident outputs (`complete_device_query`) with the post-timer host transfer recorded separately (`device_plus_output_transfer`); every repetition retained; medians reported |

## 3. Arms

All ROM arms use the same frozen bank on the same mesh; all are scored against the same
same-grid DST truth in the same job. Same-job controls only; no ratio crosses jobs or GPUs.

### 3.1 The ladder in $q$ (head anchored, linear corrections appended)

The enriched decoder is $a(z,y) = h_{32}(z) + B_q\,y$, $B_q \in \mathbb{R}^{64\times q}$, and the
weak (Galerkin, 64 tests = the bank itself) manifold equation is solved for the joint
acceleration $(\ddot z, \ddot y)$ by the retained least-squares RK4:

$$
\bigl[\,J_{32}(z)\;\;B_q\,\bigr]\begin{pmatrix}\ddot z\\ \ddot y\end{pmatrix}
= -\bigl(c^2 K\,a + h_{32}''(z)[\dot z,\dot z]\bigr),
\qquad a = h_{32}(z)+B_q y .
$$

No analytic elimination is needed as in Poisson: the evolution is explicit and the joint
$64\times(32+q)$ solve costs the same launch-bound stage as the $q=0$ solve. This is exactly the
construction of the retained `trained_nested40` (`nested_head.build`), generalised in $q$.

| arm | $q$ | $B_q$ | $y$-starts |
|---|---:|---|---|
| `head_q0` | 0 | — | retained `chol_guard` |
| `trained_nested40` | 8 | scaled PCA columns 32:40 (retained selection, as-is) | retained initializer |
| `nested_q8` | 8 | **orthonormal** PCA directions 32:40 | affine projection; fixed codes carry $y=0$ |
| `nested_q16` | 16 | orthonormal PCA directions 32:48 | same |
| `nested_q32` | 32 | orthonormal PCA directions 32:64 | same |
| `linear_bank64` | $R$ | all 64 coefficients, **no head** | raw projection $a_0 = G^\top M u_0$, $b_0 = G^\top M v_0$ |

`nested_q8` and `trained_nested40` span the same manifold in different $y$-coordinates (the
same 8 directions, scaled vs orthonormal); they must agree to integration roundoff and that
agreement is recorded (§6 G5a). At $q=32$ the joint tangent $[J_{32}\,|\,B_{32}]$ is $64\times64$;
if it is nonsingular the manifold is a local reparametrisation of the whole bank and the
trajectory must reproduce the linear bank evolution up to the RK4 integration error (§6 G5b).
The PCA directions come from the regenerated 64-case training ladder
(`dynamics.regenerate_ladder`, data hashes gated against `data_manifest.json`).

### 3.2 The top rung, and its integrator/energy certificate

`linear_bank64` is the retained `modal_projection.linear_evolution`: with $K = G^\top M L G$
(SPD, audited minimum eigenvalue $\approx 19.7$ on every mesh), $K = V\Lambda V^\top$,
$\omega = c\sqrt{\Lambda}$, the reduced system $\ddot a = -c^2 K a$ is propagated **exactly**
by $a(t) = V(\cos\omega t\,\hat a_0 + \omega^{-1}\sin\omega t\,\hat b_0)$. Because the bank is
mass-orthonormal and the 64 weak tests *are* the bank, the retained weak least-squares form
and the Galerkin projection coincide (the joint solve above with $J\to I$ is
$\ddot a = -c^2Ka$ exactly); both are therefore covered by one arm, and the identity is
asserted numerically in the local smoke. The reduced energy
$E_r(t) = \tfrac12\bigl(b^\top b + c^2 a^\top K a\bigr)$ is conserved exactly by the modal
propagator; its numerical drift is the certificate for a symplectic-like projection.

Two integrator variants isolate the integrator from the subspace, both at the head's
$\Delta t = 0.01$ on the same 64-dim system: `linear_bank64_cn` (Crank–Nicolson, symmetric,
energy-preserving) and `linear_bank64_rk4` (the head's RK4, slightly dissipative).
If the exact modal arm were unstable (it cannot be with $K\succ0$; this is asserted, not
assumed), the CN variant is the pre-declared fallback.

### 3.3 POD-Galerkin controls, same snapshots, same integrator

$k' \in \{16, 40, 64, 128\}$. Basis from the **same 64 training trajectories the bank was trained
on** (seed 690601, $256^2$, RK4 at CFL 0.12, regenerated in-job and hash-gated), displacement
and velocity snapshots normalised per case by the campaign's scales
($\|u_0\|_M$, $\sqrt{2E_0}$), mass-weighted SVD of the $6272\times 65025$ snapshot matrix.
Transfer to $64^2$ / $1024^2$ is the sine interpolant of each mode evaluated on the query
grid (nodal subsampling on the nested coarse grid; DST-I zero-padding on the fine grid — a
smoke gate checks a pure sine mode transfers to $10^{-12}$), followed by the same mass-QR
re-orthonormalisation the bank receives. Evolution: exact modal, identical code path to
`linear_bank64`. Same arms as for the bank: error, cost, reduced-energy drift, projection floor.

### 3.4 Full-order controls, same job

| arm | what |
|---|---|
| `dst` | direct exact semidiscrete solve (also the truth; its cost is the "direct solver" bar) |
| `rk4_fom` | explicit RK4 at CFL 0.45 (`fresh_fom.integrate`) |
| `cg_1e-06` | implicit-midpoint, counted CG, $\Delta t = 0.0025$, tol $10^{-6}$ (retained tight iterative control) |
| `cgdt_0.005_tol_1e-06`, `cgdt_0.005_tol_0.01` | the retained looser CG settings; the fastest one passing all-state 5 % on all 8 cases at that mesh is "the tolerance-matched FOM" |
| `dst_coarse64` ($N>64$ only) | the coarse-mesh FOM: restrict the supplied fields to $64^2$ (nodal), DST-propagate, prolong every output by DST-I zero-padding; fully charged |

### 3.5 The three-layer decomposition (post-timer diagnostics, every case, all 49 times)

For every head rung ($q\in\{0,8_{\text{scaled}},8,16,32\}$): **bank floor** = mass-$L^2$
projection of the truth onto the bank; **best-found** = the retained 8-start / 800-iteration
fit of the rung's decoder to the truth coefficients at each observation time (velocity by
tangent least squares), the representation error the frozen manifold can reach when handed
the answer; **solved** = the evolved arm's error. For every linear arm (bank and POD): the
projection floor and the solved error. The $t=0$ compression of each arm is reported
separately. Fit stationarity counts are recorded, not gated.

## 4. Pre-registered criterion — the degenerate curve

Primary metric: **worst over cases of the time-maximum initial-normalised energy-state
error** (the all-state target metric the head fails). Secondary: displacement, velocity,
current-relative variants; all reported.

- **D1 (accuracy).** At every mesh, `linear_bank64` has worst energy-state error $\le$ every
  head rung ($q = 0, 8_{\text{scaled}}, 8, 16, 32$).
- **D2 (cost).** At every mesh, the median `complete_device_query` of `linear_bank64` is
  $\le 0.1\times$ that of every head rung.
- **D3 (non-dominated set).** Among the ROM rungs, the non-dominated set on (worst
  energy-state error, median GPU ms) is the singleton $\{q = R\}$ at every mesh.
- **D4 (certificate).** Reduced-energy drift of `linear_bank64` $\le 10^{-10}$ relative over
  the horizon on every case and mesh; `linear_bank64_cn` $\le 10^{-8}$; `linear_bank64_rk4`
  reported.

**Verdict "degenerate curve confirmed for waves"** requires D1–D4 at all three meshes.

**Falsification.** If at any mesh a head rung has strictly lower worst energy-state error
than `linear_bank64`, or `linear_bank64` costs more than $0.1\times$ the cheapest head rung,
the claim is falsified at that mesh and the row is reported as an instructive failure with
the decomposition (§3.5) saying where the head's advantage came from. A D4 failure with
D1–D3 holding is reported as "degenerate but not certified" and the CN variant becomes the
reported top rung.

Secondary, pre-registered but not verdict-bearing:

- **H-mono.** Solved error is non-increasing in $q$ along `head_q0` → `nested_q8` →
  `nested_q16` → `nested_q32`, and `nested_q32` reproduces `linear_bank64_rk4` within the
  parity band of §6 G5b. Any increase is reported as a non-monotonicity (runs are
  deterministic; there is no noise band).
- **H-POD.** The learned bank at $R=64$ and POD-Galerkin at $k'=64$ are compared on error and
  cost; the expectation is "comparable" (the bank is a learned linear subspace), with
  POD-128 better. No verdict hangs on it.
- **H-FOM.** No ROM rung beats direct DST on error; the non-dominated set *including* FOMs is
  reported per mesh and is expected to contain `dst` and possibly `linear_bank64` on cost.

## 5. Jobs

| job | mesh | arms | walltime | GPU |
|---|---:|---|---|---|
| J1 `wl64` | $64^2$ | all of §3 except `dst_coarse64` | 1 h | a100 |
| J2 `wl256` | $256^2$ | all of §3 | 1.5 h | a100 |
| J3 `wl1024` | $1024^2$ | all of §3 | 3 h | a100 |
| J4 (optional, only if J1–J3 are collected with time to spare) | $1024^2$ | head rungs with a fused/lean stage kernel, parity-gated $\le 10^{-12}$ against J3 fields | 1 h | a100 |

`--mem 180G`, `--exclude pax007`, partition `gpu`, resubmit as h100 → h200 → l40s after 3 h
pending, science unchanged. Cap 8; retracted jobs count.

## 6. Gates

- **G0 (local, before J1).** `consolidated/replay.py wave` through the slot helper reproduces its
  recorded JSON (parity $\le 10^{-8}$ on fields against the archived A100 output).
- **G1 (local smoke, before J1).** The new driver at $64^2$, one case, one repetition,
  reproduces accel12's `trained_nested40` and `chol_guard` opened_0 worst energy-state and
  displacement errors at $64^2$, and accel07's `linear_bank64` opened_0 error, each to
  $\le 10^{-9}$ relative; the measured value is recorded whatever it is (cross-machine
  1-ulp transcendental differences are a known landmine — if the local gate reads between
  $10^{-9}$ and $10^{-8}$ it is recorded and the in-job A100 gate G3 decides).
- **G2 (in-job).** `jax_backend=gpu`, x64, highest, frozen-math SHA256s; bank rebuild parity
  audits $\le 10^{-9}$; nested inclusion errors $\le 10^{-10}$ and sampled joint rank
  $> 10^{-8}$ for every $q$; POD transfer sine-mode check $\le 10^{-12}$ and post-QR
  orthogonality $\le 10^{-9}$; $K \succ 0$ for bank and every POD rank; training-data hashes
  match `data_manifest.json`; every timed ROM `completed`; CG true residual $\le 1.01\,$tol;
  repetitions byte-identical (hash) — nondeterminism aborts.
- **G3 (retained-value gates, in-job, A100 class).** On the 4 accel12 cases:
  `head_q0` and `trained_nested40` worst errors equal accel12's `chol_guard` /
  `trained_nested40` at the same mesh to $\le 10^{-9}$ relative; at $64^2$, `linear_bank64` on
  opened_0/1 equals accel07 to $\le 10^{-9}$. Timing is not gated.
- **G4 (audit, local, NumPy only).** Recompute every reported error from saved coefficients
  $\times$ saved bank/POD tables (ROM arms, full grid) and from saved fields (FOM arms: full
  grid on the first case per arm, common $64^2$ grid on the others) against a reference
  regenerated independently with SciPy's DST (`dstn`) from the saved initial fields; recompute
  reduced-energy drifts; recompute medians. Max discrepancy $\le 10^{-10}$ on errors.
- **G5 (consistency, reported not gated).** (a) `nested_q8` vs `trained_nested40` field
  discrepancy; (b) `nested_q32` vs `linear_bank64_rk4` field discrepancy, expected
  $\lesssim 10^{-6}$ (64×64 normal equations over 960 stages); (c) `linear_bank64_rk4` vs
  `linear_bank64` (the RK4 integration error at $\Delta t=0.01$).

## 7. Outputs

`result.json` (every invocation with its timing components, metrics, fit and guard
diagnostics), per mesh: bank table, POD bases (128 columns), per case and arm the
coefficient trajectories (ROM), restricted/full fields (FOM), initial fields and parameters.
Report `experiments/w-ladder/reports/2026-09-1x-w-ladder.md` + `summary.json` generated by
`generate_w_ladder.py`: per-mesh ladder table, non-dominated sets (ROM-only and with FOMs),
decomposition table, energy table, POD table, FOM table, a generated figure, glossary.

## 8. What is deliberately not done

No retraining, no head tuning, no new time step, no new cohort beyond opening validation
indices 2–3 of the two development seeds, no absorbing boundary, no final cohort.

## 9. Glossary

- **Bank / head / rung:** the 64 learned spatial functions / the nonlinear map from $K$
  latent coordinates to bank coefficients / one operating point of the correction ladder.
- **$q$ / $R$ / $K$:** number of appended linear correction directions / bank rank (64) /
  head latent dimension (32).
- **Galerkin / weak least squares:** projecting the PDE onto test functions; here the tests
  are the bank itself, so the two coincide.
- **Modal propagation:** exact solution of a linear oscillator system via its eigenmodes.
- **Energy-state error:** combined displacement-gradient and velocity error, scaled by the
  initial phase energy; the all-state target metric.
- **Non-dominated set:** operating points no other point beats on both error and cost.
- **Bank floor / best-found / solved:** error of the best linear projection / of the best fit
  the frozen manifold can reach when handed the truth / of the evolved ROM.
- **POD:** proper orthogonal decomposition, the classical snapshot-SVD linear basis.
- **DST / CG:** direct discrete-sine-transform solve / iterative conjugate-gradient solve.
- **Development cohort:** cases used for design and screening; the sealed final cohort is
  never opened here.

## 10. Amendments

### A1 (2026-09-17, before any job) — acting on the independent design audit

The Codex quota was exhausted (notice in LANE-PROTOCOL.md), so the pre-job audit was done by
an independent Claude reviewer with the same read-only prompt; its report is
`reports/design-audit-2026-09-17.md`. Findings accepted and what changed:

1. **The §1 table's millisecond values were hand-typed from other jobs' tables** (audit
   finding 38–39). They are withdrawn. The table below is printed by `design_table.py` from the
   two archived JSONs; the head rows are worst over 4 cases (accel12) while the linear-bank row
   is worst over 2 cases (accel07) and the two jobs are different allocations, so **no ratio is
   formed across them here** — J1 measures the in-job ratio.

| accel12 (job 3565786, NVIDIA A100 80GB PCIe, result sha256 4993f0d12cfb), 64 intervals | cases | GPU ms (median, all reps) | worst energy-state % | worst u % |
|---|---:|---:|---:|---:|
| `chol_guard` (dt 0.01) | 4 | 178.9815 | 6.2231 | 1.8109 |
| `trained_nested40` (dt 0.01) | 4 | 199.4380 | 5.0411 | 1.4779 |

| accel07 (job 3563590, NVIDIA A100 80GB PCIe, result sha256 3638e00ac97d), 64 intervals | cases | GPU ms (median, all reps) | worst energy-state % | worst u % |
|---|---:|---:|---:|---:|
| `chol_guard` (dt 0.01) | 2 | 182.2200 | 6.2231 | 1.8109 |
| `linear_bank64` (dt 0.0) | 2 | 1.4473 | 3.2113 | 0.7826 |


2. **Direction scaling (finding 8).** The pre-registered "orthonormal" PCA directions for
   `nested_q*` are replaced by the **scaled** directions `standardized_linear[:, 32:32+q]`,
   exactly `nested_head.build`'s convention, so `nested_q8` and `trained_nested40` differ only
   in the fixed-code $y$-starts. Column norms of the scaled directions fall from
   $8.7\times10^{-4}$ (column 32) to $7.6\times10^{-5}$ (column 63); the guarded-Cholesky
   fast path needs $1/(\|J\|_F\|L^{-1}\|_F) > 10^{-3}$, so **the $q = 16$ and $q = 32$ rungs
   are expected to run the QR + exact-SVD fallback on many or all stages** and cost several
   times `head_q0`. That cost is the genuine cost of that operating point in the retained
   solver and is reported as measured, with `total_guard_fallbacks` per invocation; §3.1's
   sentence "costs the same launch-bound stage as the $q=0$ solve" is withdrawn. The
   trajectory is invariant to the $y$ scaling in exact arithmetic; the guard is not.
3. **D3's arm set (finding 1)** is the five head rungs of §3.1 plus `linear_bank64`; the
   integrator variants and POD arms are reported in a second, non-verdict set.
4. **D1 and the initial state (finding 3).** The primary metric includes $t = 0$, where the
   head returns an 8-start nonlinear fit and the bank a raw projection; this is the single
   largest disclosed confound. The report states, per arm, whether the worst is attained at
   $t = 0$ and gives the worst over evolved times ($t > 0$) beside it; the dimension-matched
   reading `pod_k40` vs `trained_nested40` is reported. D1 itself is unchanged.
5. **D2 matched-integrator reading (finding 2).** Beside D2, the ratio
   `linear_bank64_rk4` / `head_q0` (same RK4, same $\Delta t$, same 240 steps) is reported so
   "closed form beats stepping" and "linear beats nonlinear" are separated. D2 unchanged.
6. **$q = 32$ vs the bank (findings 6–7).** G5b's "$\lesssim 10^{-6}$" is withdrawn: the two
   arms start from different initial states (fit vs projection) and RK4 is not invariant under
   the nonlinear change of variables, so the pair measures both; the RK4-vs-modal pair
   `linear_bank64_rk4` vs `linear_bank64` sets the integration band. All three are reported.
7. **RK4 on the full bank (finding 14).** $\lambda_{\max}(K)$ is now recorded per mesh with
   $\omega_{\max}\Delta t$ at $c = 1.15$ against the RK4 stability bound $2\sqrt2$. If
   `linear_bank64_rk4` fails to complete it is reported as such; it is an integrator control,
   not verdict-bearing.
8. **Gates enforced in code (findings 27–28):** bank $K \succ 0$ asserted; every non-head
   arm must `complete`; every CG arm must meet its true-residual tolerance with no cap exits;
   the retained-value gate G3 raises after saving `result.json`, so a failed gate is a failed
   job (no `ALL-DONE`). Head rungs that do not complete are recorded, not aborted (a
   non-completing rung is a finding). Rank checks now take the minimum over the untransformed
   and mesh-transformed joint Jacobian (finding 10); the initializer identity is checked to
   $10^{-12}$ absolute, not by projector (finding 21).
9. **Smoke (finding 26).** The local smoke uses every 8th training case for the POD
   (`pod_training_case_stride: 8`, hash gate skipped and recorded); the jobs use all 64
   cases with the hash gate. The exact method-of-snapshots POD is not the campaign-config
   randomized SVD and is labelled so (finding 19). Transfer checks now include a
   near-Nyquist mode and the prolong→restrict round trip (finding 18).
10. **Fairness of the POD transfer (finding 17):** the neural bank is re-evaluated
    analytically on the fine grid and carries content beyond the $255^2$ sine band; POD modes
    are band-limited to it. This mildly favours the bank at $1024^2$ and is disclosed with H-POD.
11. **Operations:** a free-space preflight (≥ 30 GB on paralab, exit 43 otherwise) precedes
    any write; walltimes 2 h / 3 h / 6 h for J1–J3 (host-side scoring of 49 fields at $1023^2$
    dominates J3, finding 36); the CN propagator is built inside the timer while the modal
    eigendecomposition is offline (finding 13, disclosed); the 64 training trajectories are
    regenerated twice per job (once for the PCA ladder, once for the POD; ~100 s each, accepted).

Findings rejected: finding 25 (four repetitions) — the protocol fixes three timed repetitions;
order alternation 2:1 is disclosed and all repetitions are retained. Finding 30
(record-not-abort nondeterminism) — a nondeterministic timed output invalidates the hash
deduplication and stays a hard stop.

### A2 (2026-09-17, after the local smoke, before any job) — an integrator tie band for D1, and what the smoke showed

The G1 smoke (`checks/smoke.log`, `checks/smoke-out/result.json`; 64², `opened_0`, one call
per arm, POD from every 8th training case) reproduced all nine retained accel12 / accel07
error values to at most 8.8e-14 relative on the GB10 (gate 1e-9). Its arm table, printed
from the smoke JSON (GB10 single-call timings are not comparable to the A100 job timings):

| arm | GPU ms (GB10, one call) | worst energy-state % | guard fallbacks | reduced-energy drift |
|---|---:|---:|---:|---:|
| `head_q0` | 349.536 | 4.3175 | 0 |  |
| `trained_nested40` | 343.124 | 3.8783 | 0 |  |
| `nested_q8` | 745.433 | 3.1594 | 0 |  |
| `nested_q32` | 14904.444 | 2.8188 | 961 |  |
| `linear_bank64` | 0.850 | 2.8240 |  | 1.1324274851176597e-14 |
| `linear_bank64_cn` | 7.821 | 5.0308 |  | 2.1760371282653068e-14 |
| `linear_bank64_rk4` | 4.483 | 2.8174 |  | 2.1608847809662102e-05 |
| `pod_k16` | 3.759 | 32.4052 |  | 2.220446049250313e-15 |
| `pod_k64` | 0.953 | 1.3888 |  | 6.217248937900877e-15 |
| `dst` | 3.658 | 0.0000 |  |  |
| `rk4_fom` | 7.699 | 0.0298 |  |  |
| `cg_1e-06` | 122.315 | 0.3629 |  |  |
| `cgdt_0.005_tol_0.01` | 54.498 | 1.2127 |  |  |

Bank $\lambda_{\max}(K) = 1288.3$ at 64², so
$\omega_{\max}\Delta t = 0.413$ at $c=1.15$, inside the RK4
bound. Consistency: `nested_q32` vs `linear_bank64_rk4` 2.1e-05,
`linear_bank64_rk4` vs `linear_bank64` 8.0e-05,
`linear_bank64_cn` vs `linear_bank64` 3.1e-02,
`nested_q8` vs `trained_nested40` 1.5e-02.

Three things this changes, all declared before J1:

1. **D1 gets an integrator tie band.** On this case `nested_q32` reads 2.8188 % against
   `linear_bank64` 2.8240 %, and `linear_bank64_rk4` reads 2.8174 %: the
   $q=32$ rung *is* the full bank stepped by RK4 (consistency 2e-5), and RK4's slight
   dissipation happens to lower the metric by 0.005 pp relative to the exact propagator. A
   strict D1 would call that a head win. D1 is therefore evaluated with the tie band
   $\delta = |E(\texttt{linear\_bank64\_rk4}) - E(\texttt{linear\_bank64})|$ measured on the same mesh
   and cases: a head rung beats the top rung only if its worst energy-state error is below
   the top rung's by more than $\delta$. The strict comparison is still printed beside it.
2. **The Crank–Nicolson variant is dispersive at $\Delta t = 0.01$** (5.0308 % with
   reduced-energy drift 2e-14): CN's phase error per step is $(\omega\Delta t)^3/12$, RK4's
   $(\omega\Delta t)^5/120$. D4's CN clause is kept as the energy certificate it is; CN's
   accuracy at this step is reported as a finding, not used as a top rung.
3. **The fallback cost is real**: `nested_q32` took the QR + SVD fallback on all 961 stage
   evaluations (14.9 s); `nested_q8` none. Expected per A1 item 2; reported as measured.

Also observed on this one case and left for the jobs: `pod_k64` (1.3888 %) is more
accurate than the learned bank at the same rank, and the $q=32$ best-found error equals the
bank floor exactly, as the full-rank tangent predicts. H-POD's "comparable" expectation may
be wrong; it is not verdict-bearing.
