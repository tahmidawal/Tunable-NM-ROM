# DESIGN — burgers2d-speed: a faster 2D Burgers NM-ROM at $256^2$, $512^2$, $1024^2$ without retraining

Written before the first cluster job (2026-09-23, ~16:30 EDT). Lane `exp/2026-09-23-burgers2d-speed`, forked from
`exp/2026-09-23-burgers-bank-knob` @ `b5c843ab` (the lane that produced Table 1's Burgers rows). Cluster namespace
`/cluster/tufts/paralab/tawal01/b2speed_20260923/`, one sub-directory per job, never reused. Local commits only;
**never pushed**.

## 1. Question

Table 1's Burgers rows are slower than Newton–BiCGStab at the small meshes (dev6, same-job FOM):

| mesh | accurate (parent pick) | fast (parent pick) |
|---|---|---|
| $256^2$ | $R'=384$ linear, x1: 0.166 %, 123.7 ms, 0.15× (FOM 17.9 ms) | $R'=128$ linear, x1: 1.60 %, 31.8 ms, 0.56× |
| $512^2$ | $R'=384$ linear, x1: 0.195 %, 229.3 ms, 0.12× (FOM 28.6 ms) | $R'=512$ $q=0$: 2.14 %, 35.5 ms, 0.81× |
| $1024^2$ | $R'=384$ linear: 0.211 %, 83.1 ms, 0.89× (FOM 73.8 ms) | $R'=128$ linear: 1.83 %, 32.1 ms, 2.30× |

With the **same frozen model** (checkpoint `dn256b`, rotation `inputs/rotation_R512.npz` sha256 `51149166…`) and the
**same settings rule**, can deployment knobs and iterate-preserving engineering make the fast setting ≥1× at
$256^2$/$512^2$ and the accurate setting ≥1× at $1024^2$?

Why there is room: the ROM's cost is mesh-independent except for the initial fit, the decode and the exact first
step, and at these meshes it is dominated by (i) the dense $(M\times m)(m\times R')$ test-matrix product in every
Levenberg–Marquardt (LM) iteration, (ii) the exact first step's Jacobian, built by pushing $R'$ forward-mode tangents
through the dense residual ($R'$ dense sine projections per Jacobian), and (iii) per-iteration launch/sync overhead
of the LM while loop.

## 2. Engineering (iterate-preserving; `b2fast.py`)

| id | change | why it is the same arithmetic |
|---|---|---|
| E1 | The lattice rule's $P_q^\top X$ (tensor $63\times63$ sub-lattice, equal weights, sine test functions) is applied separably through the distinct wave numbers: $P_q^\top X = c\,[S_{x,u}^\top (X S_{y,u})]_{(i_x,i_y)}$ | the same sum, reassociated |
| E2 | The exact first step's Jacobian is assembled analytically, $J = A + \Delta t\,S\,\Phi^\top(D\odot \mathrm{stencil}(G))$ with $D$ the pointwise derivative of the upwind flux (the same `jax.grad` of the same flux), instead of `jax.linearize` with $R'$ tangents | same derivative, other association |
| E3 | Every dense $\Phi^\top v$ uses the distinct-wave-number factorisation | same sum, reassociated |
| E4 | Compile mode `graphs`: XLA command buffers including WHILE and CONDITIONAL (`xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE,CONDITIONAL`, `xla_gpu_graph_min_graph_size=1`), per-jit compiler options | same HLO, launched as CUDA graphs |

The LM (`b2fast.make_fused_lm`) is the text of `hfast.make_fused_lm` with the loop driver factored out; with no cap
it is the same program. Engineering is accepted only through the **parity gate** (§5).

**Compile modes are applied symmetrically.** Every ROM candidate and every one of the 15 Newton–BiCGStab settings is
compiled and timed in BOTH modes (default and `graphs`). A ROM knob's time is the faster mode's median; the FOM
comparator is the fastest (setting, mode) meeting the accuracy condition. Both modes of a knob must give identical
fields and integers (parity gate). The parent's text is also timed in the same allocation (default compile) at the
three Table-1 knob settings, so the same-job before/after factor is measured directly.

## 3. Deployment knobs and the pre-registered candidate set (`make_configs.candidates`)

Knobs only: span width $R'$, test count $M$ (kept at $4R'$), quadrature rule (`lat64`, $63\times63$ lattice, for the
linear rung; `q0scaled` file rule for the $q=0$ head), exact first step on/off, LM iteration cap. No retraining,
no new rules, no gtol change (the parent's accurate variant: gtol $10^{-2}$ at $\le512^2$, $10^{-3}$ at $1024^2$).

At every mesh, all **engineered** (E1–E3), each in both compile modes:

| family | settings |
|---|---|
| linear rung | $R' \in \{384, 256, 192, 128, 96\}$, $M=4R'$, `lat64`, the parent's accurate variant (Cholesky, clipped step, damping carried, quadratic predictor) × exact first step {on (x1), off} at $\le 512^2$ (only off at $1024^2$, as in the parent) × LM iteration cap {none, 1} |
| $q=0$ head | $R'=512$, $M=64$, `q0scaled`, the parent's fast variant (LU, clip, carry, quadratic predictor, gtol $10^{-3}$) × cap {none, 1} |

22 knobs at $256^2$/$512^2$, 12 at $1024^2$. The `cap 1` knob runs at most one damped Gauss–Newton step per time
step (straight-line code, no while loop); it changes iterates, so it is a separate knob whose error is measured.
New $R'$ values (96, 192) are prefixes of the same fixed rotation; the nested column ladder is
$[32,64,96,128,192,256,384,512]$.

Non-candidates in the same job: the parent's text at the three Table-1 knobs (default compile; parity twins, "before"
times); at $1024^2$ also the **general solver path** on the Table-1 fast knob ($R'=128$ linear; `b2fast.
make_linear_query_general`: `varpro.make_block_lm` verbatim — jacfwd Jacobian, LU, over-long step rejected,
residual then residual+Jacobian under `lax.cond`, damping restarted every step — with the two-way linear predictor),
for the paper's §6.3 statement "the current solver path reaches the same error as the path it replaced X×
faster": $X$ = general-path median / fast-path median (parent text, default compile), same cases, same A–B–A
protocol; errors and iteration counts of both reported.

## 4. Quadrature certification (per knob, at the target mesh)

As the parent lane: $\rho = \lVert P_q^\top a_{\rm EQ} - \Phi^\top a\rVert/\lVert\Phi^\top a\rVert$ on the states the
knob itself reaches on the eqcert held-out populations (`params_draw(20260922,56)` at $256^2$,
`params_draw(20260921,56)` otherwise; 5 certification draws × 8 + 1 confirmation draw × 16), exact side by the
full-grid separable projection, bar **0.116**; **confirmed** iff every certification draw and the confirmation draw
pass. **States: $k \ge \max(j,1)$ for an $x_j$ knob (primary)** — the paper's disclosed check, which excludes the
initial state $k=0$ that backward Euler never evaluates, and equals the parent's pre-registered $k\ge j$ for x1
knobs. $k\ge j+1$ (the states the EQ advection actually touches) is reported beside it. Only the `graphs`-mode arm of
each knob is certified. Never the NNLS fit residual.

## 5. Gates (all must pass for a mesh's result to be accepted)

| gate | bar |
|---|---|
| backend / x64 / highest | log `jax_backend=gpu`; asserted in the job |
| cohort | dev6 sha256 `108f12dc…` |
| rotation, directions, rule files | sha256 equal to the committed files |
| $\Phi$-free operator, E3 dense projection vs explicit $\Phi$ | $\le 10^{-12}$ relative ($L\le512$) |
| E1 separable lattice projection vs dense $P_q$ | $\le 10^{-12}$ relative, every (rule, $M$) used |
| **parity** | engineered arm vs parent twin (Table-1 knobs, both modes) and `graphs` vs default for every knob: fields $\le 10^{-10}$ relative on every case, identical per-step iteration counts and exit reasons |
| fast bar reproduces | the parent twin $R'=512$ $q=0$ reproduces the parent job's worst evolved error at this mesh ($\le 10^{-6}$ relative) |
| repetitions | every timed invocation's output SHA-identical to the quick run; ROM ≥ 10 invocations per (arm, case) (5 in A1 + 5 in A2), FOM ≥ 3 |
| **A–B–A drift** | per ROM arm: A2 median / A1 median in $[1/1.10, 1.10]$ |
| **neighbour** | per subject and phase: median after a long predecessor (phase median in the top third) / after a short one (bottom third) $\le 1.10$ |
| independent NumPy audit | `audit_b2speed.py` recomputes every reported error from the saved restricted fields (full grid for the audit case), recomputes $\rho$ for spot states in NumPy, re-runs the selection from raw invocations, and must detect injected controls (a swapped case, a perturbed error, a perturbed time) |

Timing protocol (as the parent and the Poisson A–B–A amendment A5): warm-up call of every subject; phases A1 (every
ROM arm, 5 reps × 6 cases, randomised within a case), B (every FOM setting in both modes, 3 reps × 6), A2 (= A1);
between phases a device sync, 5 s cool-down and a 2 s dummy kernel; 0.1 s burn before every invocation;
`block_until_ready` on both sides; GPU time = supplied dense input resident on the GPU → six dense output fields.
A ROM arm's time = median over A1 ∪ A2 (all reps kept); a FOM's = median over B.

## 6. Pre-registered setting rule (fixed now; the parent's rule on the enlarged candidate set)

Per mesh, over the candidate knobs of §3 (parent twins and the general path excluded):

- **error** of a knob = worst over the 6 dev cases of max over the five evolved output times of
  $\lVert u-u_{\rm ref}\rVert/\lVert u_0\rVert$, reference `fft_tight` at the same mesh; identical in both modes.
- **time** of a knob = the faster compile mode's median GPU ms.
- **accurate** = the confirmed knob with the smallest error. A more accurate unconfirmed knob is reported beside, marked.
- **fast** = the cheapest confirmed knob whose error ≤ the parent's fast bar at this mesh (the $R'=512$ $q=0$
  setting's error: 1.8892 % / 2.1375 % / 2.2884 % at $256^2$/$512^2$/$1024^2$, re-measured in the job). A cheaper
  unconfirmed knob meeting the bar is reported beside, marked. Near-ties (within 5 % in time) listed.
- **Table-1 speedup** = GPU ms of the fastest (FOM setting, mode) whose worst evolved error ≤ the accurate knob's,
  divided by the knob's ms, same job; the same FOM time divides both the accurate and the fast knob. Each knob is
  also reported against the fastest FOM at least as accurate as itself.
- The selection is written by a script to `selection-<L>.json` and committed **before** any held-out job.

**Held-out confirmation** (`hold`, config generated only from the committed selection): at each mesh, the selected
accurate and fast knobs (in their selected compile mode), the parent's Table-1 knobs (parent text), and the full
FOM grid in both modes, on hold64 (`params_draw(20260916, 64)`), one allocation, same A–B–A protocol (ROM 2 reps
per phase, FOM 1). The held-out numbers are reported whatever they are; settings, FOM settings, gates and cohorts
are never changed after seeing them. Note: the parent lane's held-out job `bkh64e` measures the OLD settings at
$4096^2$ only; this lane's held-out jobs measure $256^2$–$1024^2$.

## 7. Jobs

| attempt | mesh | GPU | contents |
|---|---|---|---|
| `b256`, `b512`, `b1024` | dev6 | A100 | §3 candidates + twins (+ general path at $1024^2$), certificates, A–B–A |
| `h256`, `h512`, `h1024` | hold64 | A100 | §6 held-out confirmation, after the selection commit |

Disk: restricted fields ($\le 257^2$ per output time) for every (arm, case), full fields for the audit case only;
remote directories are deleted after a checksum-verified pull.

## A0 — changes after the pre-run Codex design audit (2026-09-23 ~16:40 EDT, before any dev6 job)

The independent read-only audit (Codex; findings kept in the lane's `checks/codex-design-audit.md`) found no algebraic
error in E1–E3 (NumPy checks $<6\times10^{-16}$), confirmed the LM text is AST-identical to `hfast.make_fused_lm`,
and raised five blockers, all addressed before the first real job:

1. **Held-out audit path**: `audit_b2speed.py` now has an explicit held-out mode that consumes the frozen selection
   (`heldout_selection` in the hold config, generated only from a COMMITTED `selection-<L>.json`), reports those picks
   whatever they show, and applies no dev-cohort reproduction checks. The held-out FOM comparator is re-chosen on
   hold64 by the same rule (fastest setting+mode at least as accurate as the frozen accurate pick) — stated now.
   Held-out timed invocations check the full-output SHA on every invocation.
2. **Neighbour gate is case-controlled**: each invocation is divided by its (subject, case, phase) median; the gate
   is mean-after-long / mean-after-short ≤ 1.10; subjects with < 3 observations on either side are listed as not
   evaluable (never silently dropped).
3. **FOM compile-mode parity**: every FOM setting's two modes must give fields ≤ $10^{-10}$ relative and identical
   per-step Newton iteration vectors on every case (`fom_mode_parity` gate).
4. **Parity scope widened**: ROM parity compares the six output fields AND all 51 internal states, per-step
   iterations, exit reasons and rejected-step counts. The cap-1 knob is parity-checked against the parent text with
   an LM budget of 1 (the same semantics as a while loop capped at one iteration), for the linear rung $R'=128$ and
   the $q=0$ head. New widths share the width-generic E1–E3 code whose parity is checked at $R'=128$ and $384$.
5. **Injected controls run through the real predicates**: a swapped case must be rejected by the restricted-error
   predicate, a $10^{-6}$ relative perturbation of a reported error by the full-grid predicate, a ×1.2 A2 time by the
   drift gate.

Should-fix items adopted: the fast bar is the $q=0$ setting's error **re-measured in this job** (gated against the
parent's number); the FOM comparator requires `nonlinear_converged` on every case (inherited from the parent's
audit; disclosed here); the §6.3 general-path factor $X$ is reported with a pre-registered "same error" criterion
$|e_{\rm general}/e_{\rm fast}-1|\le 0.05$ on the worst evolved error, and is described as a reconstruction of the
general solver on the linear rung (it also changes clipping, damping carry-over and predictor, so $X$ is the factor
of the whole solver-policy change, not of any single option). Wording: the certificate covers the stored endpoint
states $k\ge\max(j,1)$, not every trial state the LM or the predictor touches; the eqcert populations are
certification data used in selection, not fresh confirmation (hold64 is the fresh confirmation). Min-of-two-modes
timing is symmetric for ROM and FOM; both mode medians are reported; held-out timing uses the frozen mode.
