# DESIGN — burgers-eq-tol-knobs: the quadrature and stopping-tolerance cost knobs at the Table-1 Burgers 2D settings

Written before any cluster job (2026-09-24, ~20:20 EDT). Lane `exp/2026-09-24-burgers-eq-tol-knobs` (worktree
`worktrees/2026-09-24-burgers-eq-tol-knobs`, forked from `exp/2026-09-23-burgers2d-speed` @ `fb4a9ff7`, sparse
checkout). Local commits only; **never pushed**. Cluster namespace `/cluster/tufts/paralab/tawal01/b2eqk_20260924/`,
one sub-directory per attempt, never reused, deleted after a checksum-verified pull.

## 1. Question

The paper's §6.3 / Figure 1 / conclusion say that on the nonlinear problems "the quadrature rule and the stopping
tolerance" are deployment-time cost knobs, but no current same-job measurement of either exists for the frozen Burgers
2D model at the Table-1 settings (lab log 2026-09-24: "no current stopping-tolerance or dense-vs-EQ same-job
measurement"). This lane measures, on the current frozen model (checkpoint `dn256b`, rotation
`inputs/rotation_R512.npz` sha256 `51149166…`), at the Table-1 accurate and fast settings, with the Table-1 FOM in the
same allocation:

1. **Empirical quadrature**: dense residual (no quadrature) vs the current rule vs a ladder of node counts
   $N_{\rm eq}\approx 0.25\times, 0.5\times, 1\times, 2\times$ (and $4\times$) the current rule.
2. **Stopping tolerance**: the LM stationarity tolerance $g_{\rm tol}\in\{10^{-1},10^{-2},10^{-3},10^{-4}\}$
   ($10^{-3}$ is the current value at $1024^2$ and $4096^2$).

## 2. Which rule the Table-1 Burgers arms use (determined from the code, not assumed)

| mesh | Table-1 arm (job) | rule field | code path |
|---|---|---|---|
| $1024^2$ accurate | `R384_lin_M1536_lat64_g0p001_fast_chol_clip_lamcarry_pred2__eng` (b1024, 4241031) | `lat64` | `config-1024.json` `rules.lat64 = {"lattice": 64}` → `b2speed.rule()` → `hops.lattice_rule(L, 64)` |
| $1024^2$ fast | `R96_lin_M384_lat64_g0p001_…__eng_graphs` (b1024, 4241031) | `lat64` | same |
| $4096^2$ accurate | `R384_lin_M1536_lat64_g0p001_…` (bk4096b, 4197473) | `lat64` | `burgers-bank-knob/config-4096.json` `rules.lat64 = {"lattice": 64}` → `hops.lattice_rule` |
| $4096^2$ fast | `R64_lin_M256_lat64_g0p001_…` (bk4096b, 4197473) | `lat64` | same |

`hops.lattice_rule(L, s)` returns the uniform interior sub-lattice with $s$ intervals per axis, nodes
$(aL/s,\,bL/s)$, $a,b=1..s-1$, and **equal weights $(L/s)^2$** — $63\times63=3969$ nodes, **no fit, no NNLS, no
training states**. It is the fine-grid composite sum restricted to every $(L/s)$-th line and maps to any mesh with
$L\bmod s=0$ at the same physical points. It is certified empirically (held-out $\rho$ on reached states, bar 0.116),
never by a fit residual. (NNLS rules exist in the repository — e.g. `q0scaled`, the file rule of the older $q=0$ head —
but no Table-1 Burgers 2D arm uses one.) The rule being a fixed lattice, the ladder varies the lattice spacing; it is
not a refit.

**Ladder construction.** $L=1024$ and $4096$ admit only power-of-two $s$, so square lattices give 961 / 3969 / 16129
nodes (0.24× / 1× / 4.06×). The 0.5× and 2× rungs are **anisotropic tensor lattices** with the same construction on each
axis (weights $(L/s_x)(L/s_y)$):

| rule | lattice | $N_{\rm eq}$ | × current |
|---|---|---|---|
| `lat32` | $31\times31$ | 961 | 0.242 |
| `lat32x64` | $31\times63$ | 1953 | 0.492 |
| **`lat64`** (current) | $63\times63$ | 3969 | 1 |
| `lat64x128` | $63\times127$ | 8001 | 2.016 |
| `lat128` | $127\times127$ | 16129 | 4.064 |

The anisotropic rungs break the $x\leftrightarrow y$ symmetry of the stencil sampling (the PDE's advection
$u(u_x+u_y)$ is symmetric); disclosed, not corrected (one orientation only).

**Dense arm.** The residual evaluated on every interior node (no quadrature): the existing exact-step path of
`b2fast.make_linear_query` (E2 analytic Jacobian, E3 distinct-wave-number dense projection) run for **all 50** backward
Euler steps (`exact_steps = 50`). Same weak residual, test space, LM, predictor, initial fit, decoder. It has no $\rho$
(it is exact). $1024^2$ only: at $4096^2$ the $(L-1)^2\times R'$ dense bank prefix the analytic Jacobian needs
(51 GB at $R'=384$) does not fit beside the 69 GB rotated bank, and the parent text's `jax.linearize` path costs
$R'$ dense projections per Jacobian — well beyond the ~2 h budget; skipped by the coordinator's rule.

## 3. Arms

Code: `eqtol.py` = `burgers2d-speed/b2speed.py` @ `fb4a9ff7` plus C1 (anisotropic lattices), C2 (E1 separable tables and
the separable projection generalised from $n\times n$ to $n_x\times n_y$ node arrays; for $n_x=n_y$ the emitted
operations are unchanged) — nothing else. Every arm: linear rung, $M=4R'$, the Table-1 variant (Cholesky, clipped step,
damping carried, quadratic predictor), no LM cap, no exact first step (except the dense arm), step budget 600.

| mesh | code | settings $R'$ | knobs per $R'$ | compile modes |
|---|---|---|---|---|
| $1024^2$ (`e1024`) | `eng` (E1–E3, burgers2d-speed) | 384 (accurate), 96 (fast) | current; 4 other rules at $g_{\rm tol}=10^{-3}$; 3 other $g_{\rm tol}$ on `lat64`; dense | both (default, `graphs`), faster reported |
| $4096^2$ (`e4096`) | `parent` (= the burgers-bank-knob query text, `bkfast.make_linear_query`) | 384 (accurate), 64 (fast) | current; 4 other rules; 3 other $g_{\rm tol}$ | default only (as bk4096b) |

18 knobs × 2 modes = 36 ROM arms at $1024^2$; 16 at $4096^2$. The full 15-setting Newton–BiCGStab grid runs in the same
allocation (both compile modes at $1024^2$, default at $4096^2$).

## 4. Per-arm measurements

- **error**: worst and median over the 6 dev cases of the max over the five evolved output times of
  $\lVert u-u_{\rm ref}\rVert/\lVert u_0\rVert$, reference `fft_tight` (Newton, ntol $10^{-6}$) at the same mesh — the
  lane's metric.
- **time**: median GPU ms over A1 ∪ A2 (A–B–A protocol of burgers2d-speed §5: warm-up, ROM A1 5 reps × 6 cases
  randomised, FOM B 3 reps, ROM A2 5 reps; 5 s cool-down + 2 s dummy kernel between phases; 0.1 s burn before every
  invocation — 0.25 s at $4096^2$ as bk4096b; `block_until_ready` both sides; scope = dense input on GPU → six dense
  output fields). At $1024^2$ a knob's time is its faster compile mode (both modes must give identical fields).
- **iterations**: total LM iterations per case (50 steps), max per step, stalled/budget exits.
- **quadrature certificate**: $\rho$ on the states the arm itself reaches on the eqcert populations
  (`params_draw(20260921,56)`, 5 × 8 certification + 16 confirmation), exact side = full-grid separable projection,
  bar 0.116, primary states $k\ge\max(j,1)$ (burgers2d-speed §4); status confirmed / not confirmed / fails; $\rho_{\max}$
  reported. Dense arms: not applicable.
- **speedup**: (i) **Table-1 FOM** = `lean_nt3e-3_l3e-3_dt005` (the Table-1 Burgers comparator at both meshes), faster
  mode, median GPU ms in the same job ÷ arm ms; (ii) beside it, the fastest converged FOM (setting, mode) at least as
  accurate as the arm (the lane's own-comparator rule).
- **ratios reported**: dense / current ms and error; each rule / current; each $g_{\rm tol}$ / current. Every ratio is
  recomputed by the audit from raw invocation times.

## 5. Gates (a mesh's numbers are accepted only if all pass; a failure is reported, never hidden)

| gate | bar |
|---|---|
| backend / x64 / highest | log `jax_backend=gpu`; asserted in the job |
| cohort | dev6 sha256 `108f12dc…` |
| rotation, directions files | sha256 equal to the committed files |
| **parity with Table 1** | the current-setting arms (`lat64`, $g_{\rm tol}=10^{-3}$) reproduce the recorded Table-1 worst evolved error to $\le10^{-6}$ relative: $1024^2$ R'=384 0.2108198098 %, R'=96 2.7870339393 % (b1024 4241031); $4096^2$ R'=384 0.2239874024 %, R'=64 4.8304355547 % (bk4096b 4197473) |
| E1 separable lattice projection vs dense $P_q$ | $\le10^{-12}$ relative for every (rule, $M$) — covers the new anisotropic tables |
| $\Phi$-free operator / E3 | as burgers2d-speed ($L\le512$ only; not evaluated here) |
| compile-mode parity ($1024^2$) | `graphs` vs default for every knob: fields and all 51 internal states $\le10^{-10}$, identical iterations/exit reasons |
| FOM compile-mode parity ($1024^2$) | as burgers2d-speed |
| repetitions | every timed output SHA-identical to the quick run; ROM ≥ 10, FOM ≥ 3 invocations per (subject, case) |
| A–B–A drift | A2/A1 median in $[1/1.10,1.10]$ per ROM arm |
| neighbour | case-controlled after-long / after-short ≤ 1.10 |
| independent NumPy audit | `audit_eqtol.py` (from `audit_b2speed.py`): restricted-grid error recompute, full-grid at $1024^2$ for the audit case, certificate status from saved $\rho$, $\rho$ recomputed in NumPy on spot states, coefficient map, drift/neighbour recomputed, every median and ratio recomputed from raw invocations |

## 6. Stopping rule for the lane

Two jobs (`e1024`, `e4096`), submitted together. An infrastructure failure (preflight exit 42, node fault, disk full,
OOM) may be resubmitted once as a new attempt directory; a code bug found after submission is fixed, committed and
resubmitted as a new attempt, recorded in the log. No arm, rule, cohort or gate is changed after seeing numbers. The
lane ends at the coordinator's hard stop (2026-09-25 05:00 EDT) whatever state it is in: anything unfinished is
reported as not done. $4096^2$ is secondary: if its job has not finished by ~04:00 EDT it is reported not done.

## 7. Jobs

| attempt | mesh | GPU | notes |
|---|---|---|---|
| `e1024` | $1024^2$ dev6 | A100 80 GB | 36 ROM arms + 30 FOM subjects |
| `e4096` | $4096^2$ dev6 | H200 (the 69 GB rotated bank; bk4096b ran on H200) | 16 ROM arms + 15 FOM settings |

## A1 — 2026-09-24 ~21:00 EDT, before any 4096² result: `e4096` moved from H200 to A100-80G

`e4096` (job 4303889, H200) never started: the scheduler's start estimate was 2026-09-26 (a long H200 queue from another
user), past the hard stop. It is cancelled and replaced by `e4096b` on an **A100 80 GB**. To fit, the driver gains C4
(`bank_columns = 384`): the offline rotation keeps only the first 384 rotated columns, $(GT)_{[:, :384]} = G\,T_{[:, :384]}$
(51.5 GB instead of 68.7 GB). No arm reads a column beyond $R'=384$, so every arm, rule and number is defined exactly as
before; round-off may differ at the GEMM level (different matrix shape), which the Table-1 parity gate ($10^{-6}$ on
the worst error) covers. Checked locally at $128^2$ (`bank_columns=64`): the $R'=32$ arm's worst error is unchanged
(3.0634 %). Absolute ms at $4096^2$ are on an A100, not the H200 of bk4096b; speedups are same-job. The e1024 job is
unaffected (staged from the earlier commit, no `bank_columns`).
