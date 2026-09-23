# NS3D: the paper's head and nested corrections inside the co-moving frame

Development design, written before any GPU job of this lane. Branch
`exp/2026-09-23-ns3d-shift-head`, forked from `exp/2026-09-22-ns3d-shift-decoder`
(`experiments/ns3d-shift`, whose `DESIGN.md`, `HANDOFF.md` and
`reports/2026-09-22-ns3d-shift-decoder.md` this builds on). Cluster namespace
`/cluster/tufts/paralab/tawal01/nshead_20260923/<attempt>_<job>/`, one directory per
job, never reused. No number in this file is a measurement.

## Question

The parent lane solved the NS3D translation orbit with a co-moving frame,
$u(x,t) = v(x-c(t),t)$, $v = Ga$ in a fixed centered rank-64 POD bank, the frame
increment $\delta$ solved online from the same weak residual. Its "learned-bank arm F"
(the paper's head inside the frame) was never run. This lane runs it, on the parent's
bank, unchanged:

$$v = G\,\big(h_\theta(z) + C_q\,y\big),\qquad \text{unknowns } (z, y, \delta)\in\mathbb R^{k+q+3}.$$

Does the correction count $q$ give a monotone error/cost ladder from one frozen head,
with the accurate end at or below about 1 % at $96^3$ and the fast end faster than the
FOM at $96^3$? And does the head beat the parent's linear-bank arm (the $q=R-k$ span
endpoint) on either axis?

## Model

- **Bank $G$** ($3n^3\times R$, $R=64$): the parent's centered POD, rebuilt in each job
  by the parent's exact recipe (training seed 202609201, 128 cases, six frames, truth
  CNAB2 $\Delta t=0.001$, snapshots centred on their own energy centroid, Gram POD,
  QR prefix). Its oracle-shift floor and the linear arm must reproduce the parent's
  `ladder64`/`ladder96` numbers (reproduction gate, value-level).
- **Head $h_\theta$** (paper architecture): two hidden SiLU layers of width 256 plus a
  linear skip $W_s^{\mathsf T}z$, $R$ outputs. Auto-decoder: one stored code per
  training state. Training data are the centred bank coefficients
  $a_i = G^{\mathsf T}\,\mathrm{centre}(u_i)$ of **training-seed states only**: 512
  training trajectories (a prefix-preserving extension of the bank's 128) at 21 frames
  each (every 0.01). Cases with index $\equiv 7 \pmod 8$ are held out of head training
  as a validation split. Loss: $\operatorname{mean}_i w_i\lVert h(z_i)-a_i\rVert^2
  + 10^{-6}\lVert z\rVert^2$ with $w_i\propto 1/\lVert u_0\rVert^2$ of the state's
  trajectory (the error metric's normalisation). Initialisation: weighted-PCA affine
  skip with the nonlinear output layer zero (exactly the rank-$k$ PCA at step 0).
  Full-batch Adam, warm-up + cosine $10^{-3}\to10^{-5}$, 100000 steps, gradient clip 1.
  Then best-found codes (Gauss–Newton refit with the head frozen).
- **Corrections $C_q$**: leading $q$ weighted principal directions of the training
  misses $a_i - h(z_i^*)$, nested in $q$.
- **Encoder (inside the timed query)**: centre $u_0$ on its own centroid, project,
  nearest stored code in coefficient space, 12 variable-projection Gauss–Newton
  sweeps for $\min_{z,y}\lVert h(z)+C_qy-a_0\rVert$.
- **Step**: the parent's co-moving residual at $a=h(z)+C_qy$; Jacobian = parent's
  analytic $(a,\delta)$ Jacobian chained with $[\partial h/\partial z,\ C_q]$; the
  parent's fixed-sweep damped Gauss–Newton (Marquardt $10^{-6}$, Cholesky), warm start
  by extrapolation. $y$ stays among the unknowns (advection is nonlinear).

## Arms, per mesh $n\in\{32,64,96\}$, one allocation each

| arm | grid |
|---|---|
| head | $k\in\{8,16,32\}$ × $q\in\{0,8,16,32,R-k\}$ × $\Delta t\in\{0.04,0.02,0.01\}$ × sweeps $\{2,3,4\}$ |
| linear bank (parent) | $\Delta t\in\{0.04,0.02,0.01\}$ × sweeps $\{2,3,4\}$ |
| CNAB2 (spectral) | steps $\{200,100,80,70,60,50,40,20,10\}$ over $T=0.2$ |
| FD-CG (iterative) | mesh $n$: steps $\{20,40,50,100\}$ × CG rtol $\{10^{-4},10^{-6},10^{-8}\}$; mesh $2n$: steps $\{40,50\}$, rtol $10^{-6}$ |

$q=R-k$ spans the whole bank, so its trajectory should coincide with the linear arm up
to solver differences; it is reported as a consistency check, and the linear arm is the
endpoint row.

**FD-CG FOM.** The parent's CNAB2 is Fourier pseudo-spectral: its implicit viscous
step is a pointwise division in Fourier space and the pressure is removed by the
Leray projector, also pointwise in Fourier space — it performs no linear solve. The
project does not feature spectral/direct solvers as the paper comparator, so this lane
adds a conventional iterative FOM for the same problem: second-order central finite
differences on the same periodic grid, rotational advection with AB2 (Euler first
step), Crank–Nicolson viscous step solved by CG, exact discrete projection with the
central-difference Laplacian solved by CG (warm-started). Its error includes its own
spatial discretisation error against the spectral truth; that is the honest price of a
grid FOM, so a finer FD mesh ($2n$, injected onto the $n$ grid, whose points coincide)
is also in the grid.

## Pre-registered selection and reporting

- **$k$** at each mesh: smallest development evolved worst at $q=0$ on the ladder
  setting ($\Delta t=0.02$, 3 sweeps); within 1 % relative, the smaller $k$.
- **Ladder setting**: $\Delta t=0.02$, 3 sweeps, fixed now for every $q$ and for the
  linear endpoint (the parent's headline region). The full frontier is reported too.
- **Comparator**: for each arm and each FOM family separately, the fastest tested
  *stable* setting (all fields finite, evolved worst $\le 100\,\%$) whose evolved worst
  is no larger than the arm's. Also reported: every arm against the fastest stable FOM
  setting at least as accurate as the **most accurate** arm at that mesh. Every speedup
  divides two medians from the same job.
- **Success** (from the task): monotone error/cost ladder in $q$ from one frozen head;
  accurate end $\lesssim 1\,\%$ at $96^3$; fast end faster than the FOM at $96^3$.
  "Monotone" means error non-increasing and cost non-decreasing in $q$ on the ladder
  setting; each is stated separately.

## Controls and gates (each checked to fail where it should)

- Local gate `test_head_solver.py`: analytic head Jacobian and chained step Jacobian
  against `jacfwd`; a linear head with $k=R$ reproduces the parent's driver; planted
  code recovery; translation equivariance; fixed-sweep driver against a generic LM
  reference; FD-CG divergence and CG convergence. The parent's `test_fast_solver.py`
  is re-run unchanged.
- Frame frozen at $\delta\equiv0$ (selected $k$, $q=0$) must fail the 5 % target.
- Random orthonormal correction directions (same $q$) must floor worse than the PCA
  directions.
- Driver parity: fixed-sweep head solve vs generic LM (jacfwd, adaptive damping) on
  four development cases; reported per sweep count.
- In-job independent NumPy audit (`verify_head.py`) from saved fields, with a
  must-fail perturbed copy that the audit has to reject.
- Timing: GPU burn-in; sentinels (linear arm, head $q=0$, head $q=R-k$) timed alone
  first; every fast arm in one randomised, interleaved, synchronised block with two
  burn-in calls and 7 repetitions; FD-CG arms in their own phase (5 repetitions);
  then each sentinel timed immediately after the job's longest arm. Gate: each
  sentinel's median after a long neighbour, and in the big block, within 10 % of its
  solo median. Timed outputs must reproduce the accuracy pass's errors.

## Held-out evaluation

Seed **202609221**, 32 cases, never drawn. Opened once, only after every setting
($k$, $q$ ladder, $\Delta t$, sweeps, damping, encoder, head checkpoint, FOM grids,
mesh) is written to `frozen/frozen_settings.json` and committed. The job refuses to
start without it, checks rounded-row disjointness against training 202609201 (512),
development 202609202 (16) and the closed 202609203 (32) / 202609211 (32), rebuilds
the bank and gates it against stored reference coefficients (sign-aligned, $10^{-8}$),
and loads the frozen head rather than retraining. Seeds 202609203 and 202609211 are
never read.

## Glossary

- **bank $G$** — the fixed spatial basis (centered POD, rank $R$).
- **head $h_\theta$** — small network mapping a $k$-dimensional code to bank
  coefficients.
- **corrections $C_q$, $y$** — $q$ fixed linear directions in coefficient space that
  capture what the head misses, with amplitudes $y$ solved per step.
- **co-moving frame, $\delta$** — the structure's translation, solved each step.
- **floor** — the error of the best reconstruction available to a trial set with the
  true (oracle) frame, before any time stepping.
- **evolved worst** — per case, the worst error over output times after $t=0$; then the
  worst over cases. Errors are relative $L^2$ against $\lVert u_0\rVert$.
- **CNAB2** — the spectral FOM (Crank–Nicolson viscous, Adams–Bashforth 2 advection).
- **FD-CG** — the finite-difference FOM whose viscous and pressure systems are solved
  by conjugate gradients.
- **sweeps** — fixed Gauss–Newton iterations per time step.

## Amendment 1 (2026-09-23, before any job) — corrections removed; importance-ordered bank

Direction change from the user, relayed by the coordinator: the paper removes the
correction directions $C_q$. The method becomes a frozen bank **ordered by importance**
with two solves. This amendment supersedes the $q$ ladder above; everything else
(bank, head recipe, FOM grids, controls, timing, held-out protocol) is unchanged.

- **Ordering** (offline, training data only): SVD of $G\,\Sigma^{1/2}$ with
  $\Sigma=\operatorname{mean}_i a_ia_i^{\mathsf T}$ the (uncentred) second moment of the
  head-training-split coefficients. $G$ is orthonormal, so the ordered bank is
  $G_r = GV$ with $V$ the eigenvectors of $\Sigma$ by decreasing eigenvalue. Rotated
  operators $A V$, $V^{\mathsf T}\mathsf T_m V$, $D_dV$ are gated against a fresh
  operator build of $GV$ ($10^{-10}$).
- **Arm (a), head only:** $v = G\,h_\theta(z)$, unknowns $(z,\delta)$, $k\in\{8,16,32\}$,
  $k$ selected by the rule above ($q=0$ throughout).
- **Arm (b), bank span:** $v = G_r[:, :R']\,c$, unknowns $(c,\delta)$, with the knob
  $R'\in\{64,48,32,16,8\}$ ($R, 3R/4, R/2, R/4, R/8$). $R'=R$ is the parent's
  linear arm up to a rotation of coordinates. Its oracle-shift floor per $R'$ is
  reported.
- The parent's unrotated linear arm is kept at every $\Delta t$ × sweeps as the
  reproduction gate against `ladder64`/`ladder96`.
- The **tunability ladder** reported per mesh is $R'$ on the ladder setting
  ($\Delta t=0.02$, 3 sweeps), with the head arm beside it; "monotone" is checked on
  $R'$. The head is also compared with the span at matched unknown count ($k=R'$).
- Timing sentinels become span $R'=64$, span $R'=8$ and the selected head.
- The corrections code path stays in `head_rom.py` (and $C$ is still computed and
  saved) but no $q>0$ arm is run; the random-direction control is dropped with it.

## Amendment 2 (2026-09-23, after `a1_h64`) — timing protocol v2

`a1_h64` (job 4197294) failed the pre-registered neighbour gate: inside the randomised
fast block every sentinel stayed within 2.3 % of its solo median, but timed
immediately after the job's longest arm (FD-CG at $128^3$, seconds long) the sentinels
ran 17–23 % slower, and the slowdown decayed over the next few calls (end-of-job solo
block: first repetition high, then back to baseline). The accuracy results of that job
are unaffected; its FD-CG speedups are not usable as they stood, because the ROM was
timed in the light state and FD-CG in the heavy one. The 96^3 job (4197368) was
cancelled before its timing phase, and every mesh is rerun under v2:

- fast block unchanged (all reduced arms + CNAB2 randomised and interleaved); CNAB2
  speedups use it, and its gate (sentinel in-block median ≤ 1.10 × solo) is unchanged;
- FD-CG arms are timed after a 1 s idle cool-down each, in their own phase;
- every ladder arm is additionally timed (a) immediately after the heaviest FD-CG arm
  on the job's own mesh and (b) after that neighbour plus the cool-down. Gate: (b) ≤
  1.10 × fast-block median. (a) is reported, not gated: it is the measured cost of the
  effect, and **FD-CG speedups are reported against both the fast-block median and the
  after-heavy median, the latter as the conservative figure**;
- GPU clocks / temperature / throttle reasons are logged at phase boundaries.

## Amendment 3 (2026-09-23, after all development jobs, before the held-out job) — freeze

Frozen by `make_frozen.py a3_h96` from the 96^3 development job (4198101) into
`frozen/frozen_settings.json` (+ head checkpoint, rotation, bank probe), committed before
the held-out job is submitted. Mesh 96^3; head k=8 (selected by the pre-registered rule
at every mesh); span ladder R' ∈ {64,48,32,16,8}; dt 0.02, 3 sweeps, damping 1e-6,
encoder 12 sweeps; CNAB2 and FD-CG grids identical to development. Held-out seed
202609221, 32 cases. Two plumbing changes since the development jobs, neither touching
the model, solver or error code: (i) `stream_audit` — each saved field set is audited
by `verify_head.py --only` in a separate process as soon as it exists and then deleted,
because a 32-case cohort at 96^3 would otherwise put ~75 GB on a share that is 95 % full;
(ii) nothing else. Known weakness carried into the held-out grid unchanged: the FD-CG
3n=288^3 setting uses 40 steps, which is past its explicit stability margin (43 %
development error); it is never a comparator and is kept only because the grid is frozen.
