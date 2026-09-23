# spectral-fom — design and pre-registration

Written 2026-09-23 before any cluster result of this lane.

**Purpose.** The paper compares NM-ROM only with named iterative FOMs (CG, CN–CG, Newton–BiCGStab). The
spectral / fast-transform FOMs are kept out of the paper for now. This lane measures and stores them, so the
authors can decide later. Nothing from here goes into `paper/`.

## Solvers

All solvers solve the paper's own discrete system on the paper's own grid.

- **Poisson square (2D) and Poisson cube (3D).** Zero Dirichlet boundaries, with the 5-/7-point Laplacian scaled by
  $n^2$. The solver is $u = S\,(S f S / \Lambda)\,S$, where $S$ is the orthonormal DST-I and $\Lambda$ holds the discrete
  eigenvalues $\sum_a 4n^2\sin^2(\pi k_a/2n)$. This is an exact solve of the same discrete system. Two
  implementations are timed:
  - `dst_fft`: an odd extension followed by a real FFT of length $2(N+1)$;
  - `dst_mm`: a dense sine matrix applied by GEMM.

  The spectral FOM is the faster of the two at each mesh. This is disclosed as a best-of-two choice.
- **Heat 2D/3D.** Diagonalised by the same DST, with two arms:
  - `modal_cn`: Crank–Nicolson applied exactly in modal space at the paper's $\Delta t$. The per-mode factor is
    $g^{s}$ with $g = (1-\tfrac12\Delta t\nu\lambda)/(1+\tfrac12\Delta t\nu\lambda)$. It is the exact CN solution,
    that is, CN–CG with the linear solve done to round-off.
  - `modal_exp`: exact propagation $e^{-\nu t\lambda}$. This is the heat lane's same-grid truth, so its error is
    round-off by construction.
- **Burgers 2D.** See section B.
- **L-shape.** No fast transform diagonalises the Dirichlet Laplacian on the non-rectangular domain. There is
  no spectral arm.
- **NS 3D.** The lane's own CNAB2 FOM is already Fourier pseudo-spectral. Its timings are recorded from
  `ns3d-shift-head` and not rerun.

## ROM settings

The accurate and fast settings are taken from the owning lanes' committed selections. Each is re-run here with
that lane's committed code, staged byte-for-byte from the pinned commit and recorded in `PROVENANCE.json`. This
lane never chooses them.

- Poisson 2D: `exp/2026-09-23-poisson-bank-knob` @ 06546331. Accurate is `R512_linear`, fast is
  `R128_linear`, on the 12 development sources.
- Cube: `exp/2026-09-23-poisson-bank-knob-3d` @ d2c775a9, `frozen-N32/64.json`. Accurate is `R128_linear`,
  fast is `R64_linear`, on the final cohort (seed 920499, 64 cases).

## Validation (untimed, in the job, before timing)

- **Spectral vs SciPy truth.** The spectral field is compared with the SciPy `dstn` same-grid truth on every
  case. The limit is $10^{-10}$ relative.
- **The paper's CG converges to the spectral field.** On case 0, CG runs at rtol $10^{-6}$, $10^{-8}$ and
  $10^{-10}$. Every run must converge. Its distance to the spectral field must decrease strictly, and the last
  distance must be $\le 10^{-8}$.
- **Control that must fail.** The same DST solve with the continuum eigenvalues $(\pi k)^2$ must exceed
  $10^{-10}$.

## Timing (`spec_timing.py`)

There is one allocation per mesh, and meshes run sequentially inside one job.

**Phases.** A–B–A, in this order:
1. ROM phase A1;
2. spectral phase B;
3. ROM phase A2, identical to A1.

Each phase runs 10 retained repetitions over every case, in randomised order within each case. Between phases
there is a sync, a 5 s cooldown and a fixed 2 s dummy matmul kernel. Before every invocation there is 0.1 s of
burn-in with the same kernel, `gc.collect()`, and the GPU UUID guard. The garbage collector is disabled during
the timed call.

**What is timed.** GPU time is `fused_device_seconds`: from after the synchronised host-to-device input copy
until `block_until_ready`. This is the owning lane's scope.

**Timing statistics.**
- ROM: the median over A1 ∪ A2.
- Spectral: the median over B.

**Gates** (limit 1.10):
- **Drift:** the ROM median in A2 divided by the median in A1 must lie in $[1/1.1,\ 1.1]$.
- **Within-phase neighbour:** the median after a long predecessor, divided by the median after a short one, must
  be $\le 1.10$.
  - With two subjects in a phase, long and short are the slower and faster of {the other subject, itself}.
  - With three or more subjects (the Burgers spectral ladder), long is the top third of the other subjects by
    phase median, and short is the bottom third.
  - Each side needs at least 3 samples.

Also required: determinism (byte-identical fields on every repetition), and the independent NumPy audit
(`sp_audit_np.py`). The audit uses a dense sine-matrix truth, not an FFT. It must detect two controls: a
swapped-case truth and a perturbed recorded error.

## Reported quantities

**Error** follows each problem's paper convention: relative L2 against the same-grid discrete truth, worst over
the cohort.

**Ratio** = spectral FOM ms / our ms. A ratio below 1 means the spectral solver is faster.

**Choosing the spectral FOM.** For exact solvers (Poisson, heat `modal_exp`), it is simply the faster variant.
For solvers with their own error (heat `modal_cn`, the Burgers ladder), it is the fastest spectral setting whose
worst error is $\le$ our arm's worst error. The tight/exact spectral arm is also reported.

**Timing rows that fail a gate** are reported with the failure beside them and are not called gate-clean.

## B. Burgers 2D (written before any Burgers job)

**Solver (`spec_core.make_burgers`).** Each backward-Euler step is solved by the fixed point
$u \leftarrow u - H^{-1} r(u)$, where:
- $r$ is the paper's own residual, `mr-burgers2d/engines.residual`, imported unchanged;
- $H = I + \Delta t\,\nu A$ is applied exactly by DST.

The fixed point satisfies $r(u) = 0$, so it is the paper's discrete solution. Each step stops at the paper's
Newton rule, $\lVert r\rVert \le \text{ntol}\,\lVert u_{\text{prev}}\rVert$. With exactly one sweep, the scheme is
the semi-implicit IMEX step: explicit upwind advection, implicit diffusion, and the same stencil.

**Why plain Picard and not Anderson acceleration.** Local prototypes at 128² and 512² (dev4 cases 0 and 1)
decided this before any timing:
- Picard needs about 4 sweeps per step at ntol $10^{-6}$ and 1–2 sweeps at loose ntol, independent of mesh.
- Anderson acceleration with memory 3 or 5 never reduced the sweep count, and cost 2–3× more per sweep.

The paper's Newton–BiCGStab already uses $H^{-1}$ as its preconditioner. Pure Picard is the variant in which
the whole solve is done by the fast transform.

**Ladder (11 subjects).**
- Picard with $\Delta t = 0.005$ and ntol $\in \{10^{-6}, 10^{-4}, 10^{-3}, 3\cdot10^{-3}, 10^{-2}\}$.
- Picard with $\Delta t = 0.01$ and ntol $\in \{10^{-4}, 10^{-3}, 10^{-2}\}$.
- IMEX with $\Delta t \in \{0.0025, 0.005, 0.01\}$.

**DST implementation.** The DST inside $H^{-1}$ (fft or mm) is picked per mesh by an untimed micro-benchmark,
and the choice is recorded.

**Reference and error.**
- The reference is the paper's `fft_tight` (Newton–BiCGStab, $\Delta t = 0.005$, ntol $10^{-6}$, ltol $10^{-8}$).
- The error is $\max_{t \ge 0.05} \lVert u - u_{\text{ref}}\rVert / \lVert u_0\rVert$, worst over dev6.

**Validation (case 0).**
- The paper's FOM at ntol $10^{-10}$ / ltol $10^{-12}$ and spectral Picard at ntol $10^{-10}$ must agree to
  $\le 10^{-8}\,\lVert u_0\rVert$.
- The control, Picard on a residual with $1.01\nu$, must not agree.

**Parity gate.** The re-run ROM arms must reproduce the burgers-bank-knob lane's recorded per-case errors to
$10^{-8}$ relative.

**ROM settings.** These are the lane's selection per mesh (`lane-ref/burgers-<L>-selection.json`, taken from the
lane's `checks/bk<L>-summary.json`):

| mesh | accurate | fast |
|---|---|---|
| 256² | R'=384 linear, x1 | R'=128 linear, x1 |
| 512² | R'=384 linear, x1 | R'=512 q=0 |
| 1024² | R'=384 q=256 | R'=128 linear |
| 2048² | R'=384 q=256 | R'=128 linear |

4096² waits for the lane's `selection-4096.json`.

**Spectral FOM for each ROM arm.** It is the fastest ladder subject whose worst error is $\le$ the arm's worst
error. Tight Picard is also reported.

## H. Heat (written before any heat job)

**ROM arms.** The heat lane selects per stepping family (`cn`, `bf`) and pooled
(`lane-ref/heat2d-selection.json`, from `heat-bank-knob/runs/h2d/summary.json` at e45cae7e). All distinct
selected arms are timed:
- pooled accurate is `lin_R128_bf` (role `rom_accurate`);
- pooled fast is `lin_R48_cn` (role `rom_fast`);
- `lin_R128_cn` and `lin_R48_bf` are timed as well.

**Cohort.** The lane's held-out cohort (`heldout_sealed_791099`, 16 cases).

**CN–CG validation tolerance.** It is rtol $10^{-11}$, not the smoke's $10^{-12}$. This was set before any
heat job, after the Poisson runs showed that the paper's unpreconditioned CG cannot certify its true residual
below about $10^{-9}$ at the finest meshes. Agreement limit: $10^{-9}$.

**Spectral FOM for each arm.** The matched spectral FOM is the fastest of {`modal_exp`, `modal_cn`} × {fft, mm}
whose worst error is $\le$ the arm's. `modal_exp` (exact) is always eligible.

## Amendment T1 (2026-09-23, after spC; post hoc, disclosed) — case-normalised neighbour statistic

**What failed.** At 512² and 1024² in spC, the pre-registered within-phase neighbour gate failed for the
Burgers accurate arm in phase A1, with ratios 1.183 and 1.192.

**Why the raw statistic is confounded.** With two ROM subjects, the "after itself" samples occur only across a
case boundary. They are therefore a different mix of cases from the "after the other" samples. The Burgers ROM's
cost depends on the case, because its iteration counts differ between cases.

**The amended statistic.** Each time is divided by the median of the same subject on the same case in the same
phase before the neighbour medians are taken. With that normalisation the ratio is 1.001 and 1.000. For the
Poisson runs, whose cost does not depend on the case, it is 1.001–1.016. The raw gate is not redefined.

**How it is applied.** Every run reports both statistics. A row whose raw gate fails but whose case-normalised
gate passes is labelled as such, and is never called gate-clean without that qualifier. The harness computes
both from spE on. For earlier runs the case-normalised value was recomputed offline with the same harness code
(`T1-case-normalised-neighbour.json` beside each result). That recomputation also confirms the raw gate matches
the recorded one.

**Burgers 2048² rerun.** spC's 2048² pass ran out of memory on an A100-PCIE-40GB while building the bank. It is
rerun unchanged as spE on an 80 GB card.

## Amendment L1 (2026-09-23, after spC/spE; written before any L1 job) — a stronger Burgers spectral ladder

**What prompted it.** The original ladder has a gap in error between ntol $10^{-4}$ (about 0.05 %) and
$10^{-3}$ (about 0.9 %) at $\Delta t = 0.005$. Every accurate arm, at 0.17–0.53 %, was therefore matched to a
spectral setting about ten times more accurate than needed. At 2048² (spE) the fast arm, at 1.87 %, came out
faster than its matched spectral setting (ratio 2.12). The accurate arm's ratio was 0.87.

**What changes.** The brief asks for the strongest reasonable spectral solver, so the spectral side gets:
- a denser ntol ladder:
  - $\Delta t = 0.005$: $\{10^{-6}, 10^{-4}, 2\cdot10^{-4}, 5\cdot10^{-4}, 10^{-3}, 3\cdot10^{-3}, 10^{-2}\}$;
  - $\Delta t = 0.01$: $\{10^{-4}, 10^{-3}, 2\cdot10^{-3}, 3\cdot10^{-3}, 5\cdot10^{-3}, 10^{-2}\}$;
- a predictor variant of each (`ppic_*`), whose first iterate per step is $2u_n - u_{n-1}$. The ROM arms use a
  quadratic predictor. The stopping rule is unchanged, so every step still satisfies
  $\lVert r\rVert \le \text{ntol}\,\lVert u_n\rVert$.

**Prototype evidence.** At 256² on the local GB10, dev6, the predictor cut the total sweeps by about 40 % at
equal ntol. Examples of worst error:
- $\Delta t = 0.005$, ntol $5\cdot10^{-4}$: 0.28 % with the predictor, 0.54 % without;
- $\Delta t = 0.01$, ntol $5\cdot10^{-3}$: 1.82 % with the predictor.

**What stays the same.** The 29-subject ladder contains the original 11 settings under the same names.
Everything else is unchanged: ROM arms, reference, gates, audit, and the T1 reporting.

**Which runs are reported.**
- The L1 runs (spF: 256²/512²/1024², any A100; spG: 2048², 80 GB) are the Burgers result.
- spC and spE stay in the report as the original-ladder record.

## Amendment L2 (2026-09-23; written before any L2 job) — a half-length DST-I

**The variant.** `dst_half` computes the DST-I from a real FFT of length $N = n$ instead of $2n$, using the
classical pre-twiddle and running-sum post-step (Numerical Recipes `sinft`). It agrees with SciPy to about
$10^{-14}$ in 2D and 3D. On the local GB10 it was 10–20 % faster than `dst_fft` for the batched heat transform at
1024² and 2048².

**Where it is used.**
- Heat: it is added as a third variant, and heat 2D is re-timed at all three meshes in one job (spI, 80 GB).
- Burgers: it joins the per-mesh micro-benchmark candidates (`dst_candidates`) for later jobs.

**What happens to earlier runs.** The heat 4096² job spH (4209368) was cancelled by me. The squeue check showed it had just started running
(correction: an earlier draft of this line said "before it started"). None of its output was pulled or used, and
its directory was removed. spD (1024²/2048², two variants) is kept as a record.

## Note K (2026-09-23) — Burgers 4096² row blocking

spK (4210663) ran out of memory on an A100 80GB while building the bank. The job had four row blocks of 17 GB,
and the rotated copy was transient alongside the 51.5 GB kept prefix. It is rerun as spL with `bank_blocks = 16`,
which affects memory only. Rows are partitioned more finely, and every arm's arithmetic is otherwise unchanged. Any
round-off difference this introduces is covered by the lane-error parity gate, which requires agreement to 1e-8
relative. No output of spK was used, and its directory was removed.

## Note M/N (2026-09-23) — cube 128³/256³ and heat 3D

**Cube 128³ and 256³ (spM).** The lane ran these meshes on its development cohort (seed 920411, 16 cases). Its
summary row (`lane-ref/cube-selection.json`) selects accurate `R128_linear` and fast `R64_linear`. They are re-timed
here on the same cohort, with the ROM code pinned at d2c775a9. The only later lane change is to `pbk3_core`, and it
touches the order-gate code only.

**Heat 3D (spN).** Meshes 32³, 64³ and 128³, from the h3d summary at 134f30a2
(`lane-ref/heat3d-selection.json`):
- pooled accurate is `lin_R320_bf`;
- pooled fast is `lin_R128_cn`;
- `lin_R320_cn` and `lin_R128_bf` are timed as well.

The cohort is held-out sealed 921099 (64 cases). Because the cohort has 64 cases, the job uses 5 retained
repetitions, which meets the brief's ≥5. 256³ follows once the lane's h3d256b selection lands.
