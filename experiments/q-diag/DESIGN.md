# q-diag — why the Burgers correction ladder is not monotone in $q$ on evolved times

**Pre-registered before any number was computed.** Lane: read-only diagnosis. No cluster
job, no GPU work beyond (at most) a sub-minute local `jaxrun` smoke; every number in the
deliverable is produced in NumPy on the already-audited saved fields and run JSONs of
`btq101`, `btq102`, `btq201` (b-ladder-top), `cclad01` (cheap-corrections) and `qlad01`
(head-ablation). Worktree `worktrees/2026-09-16-q-diag`, branch `exp/2026-09-16-q-diag`,
forked from `exp/2026-09-16-b-ladder-top` at `b8efd5b4`. Nothing is pushed or merged.

Two sibling lanes (`q-trajdirs`, `q-ridge`) are building *fixes* from the same fork. This
lane builds the *diagnosis they are judged against* and deliberately proposes no fix of its
own beyond naming which one the evidence supports.

## The observation under investigation

In `btq201` (job 3747245), at fixed test count $M = 256$, empirical quadrature and
evolution tolerance $10^{-6}$, the correction ladder

$$ u(z, y) \;=\; G\bigl(h_\theta(z) + C_q\, y\bigr), \qquad w = (z, y) \in \mathbb{R}^{K+q},\; K = 16 $$

is monotone in $q$ on the worst-over-**all**-output-times metric but **not** on the
worst-over-**evolved**-times metric: the $q=16$ rung is worse than $q=0$. The lab log
records this as open item (1) of the b-ladder-top entry.

## Metric definitions used throughout (identical to `audit_top.py`)

For an arm's field stack $f \in \mathbb{R}^{6\times(L+1)\times(L+1)}$ on case $c$, with
$g$ the same job's `fft_tight` converged full-order field stack on the same case and
$\hat u_c$ that case's 4096-interval reference:

$$ \varepsilon_{c}(t_k) \;=\; \frac{\lVert f_k - g_k\rVert_2}{\lVert \hat u_c(t_0)\rVert_2},
\qquad k = 0,\dots,5,\quad t_k \in \{0, .05, .10, .15, .20, .25\}. $$

*all-times* $= \max_c \max_k \varepsilon_c(t_k)$; *evolved* $= \max_c \max_{k\ge 1}
\varepsilon_c(t_k)$; *$t_0$ compression* $= \max_c \varepsilon_c(t_0)$. Everything is
recomputed from the saved fields; agreement with the archived `same_grid_*` values to
$10^{-12}$ relative is a precondition of this lane and is reported as a gate.

## What will be computed, and the decision rule for each candidate cause

Every rule below is stated so that it can come out **against** the cause.

### Gate 0 — reproduction

`recomputes_archived_same_grid`: for every invocation in all five archives, the recomputed
$\varepsilon_c(t_k)$ matches the archived `same_grid_per_time` to $\le 10^{-12}$ relative.
If this fails the lane stops and reports the discrepancy rather than any verdict.

### Cause (1) — directions fitted to static reconstruction residual mislead the time stepping

The directions $C$ are the field-metric POD of the decoder-output residual
$\eta - h_\theta(z^\ast)$ over 1024 **static** training snapshots (`directions.py`,
`_finish`); nothing in their construction sees a trajectory.

Computed:

1. `per_time`: $\varepsilon_c(t_k)$ for every rung, every case, every output time, and the
   paired difference $\delta_c(t_k) = \varepsilon_c^{q=16}(t_k) - \varepsilon_c^{q=0}(t_k)$
   at matched $M$, matched quadrature, matched tolerance.
2. `local_defect`: the one-output-interval defect
   $d_c(t_k) = \lVert f_k - \Phi_{\mathrm{FOM}}(f_{k-1})\rVert_2 / \lVert \hat u_c(t_0)\rVert_2$,
   $k\ge 1$, where $\Phi_{\mathrm{FOM}}$ is ten backward-Euler substeps of the *same*
   $L=256$, $\Delta t = 0.005$ upwind/5-point discretisation (`engines.residual`) solved in
   NumPy to Newton tolerance $10^{-10}$ relative. This separates error *injected* in
   interval $k$ from error *inherited* from $t_{k-1}$ and amplified.
   `local_defect_selfcheck`: applying $\Phi_{\mathrm{FOM}}$ to `fft_tight` itself must give
   $d \le 10^{-6}$; otherwise the NumPy stepper is not the job's operator and the defect
   numbers are withdrawn.
3. `latent_growth`: $\lVert y_n\rVert_2$ over the 51 internal steps from
   `internal_latents`, per rung, per case, to see whether the correction block grows along
   the trajectory.

**Decision rule.** Cause (1) is SUPPORTED iff both hold: (a) $\max_c \delta_c(t_k)$ is
positive and non-decreasing in $k$ over $k = 1..5$ (error *grows with time* at $q=16$
relative to $q=0$); and (b) the median over cases of $d_c^{q=16}(t_k)/d_c^{q=0}(t_k)$
exceeds $1$ for $k\ge2$ — the $q=16$ step map is locally worse even when started from its
own state. It is REFUTED if $\delta_c(t_k)$ is flat or shrinking in $k$, or if the local
defect ratio is $\le 1$ (the extra directions then step at least as accurately and the
regression must be inherited from $t_0$ or from noise).

### Cause (2) — more unknowns overfit the $M$ test equations

Computed:

1. `weak_residual_recorded`: the archived per-step weak-residual norms `residuals` (50 per
   invocation), compared at matched $M$ **and matched quadrature** — the `cclad01`
   `q{0,16,32,64,128}_m256_dense_block` arms, which share $M=256$, dense quadrature and one
   solver. Ratio statistics per step and per case.
2. `heldout_tests`: if and only if the correction directions can be **recovered** from the
   archives (below), the projected residual is recomputed in NumPy at every internal
   reachable state $u_n = G(h_\theta(z_n) + C_q y_n)$, on (i) the arm's own $M = 256$ test
   modes and (ii) a disjoint held-out set — sine modes ranked $257 \dots 1280$ by the
   discrete eigenvalue ordering of `engines.modes`, i.e. the next $4M$ modes.
   `directions_recovered`: $C_q$ is obtained by least squares from the pooled
   $(y, \; u - G h_\theta(z))$ pairs of every archived arm at that $q$ (6 output times
   $\times$ 6 cases $\times$ #arms), using the decoder and bank rebuilt in NumPy from the
   exact checkpoint (SHA256 `18f0266a…`, present locally). It is accepted ONLY if the
   pooled design has numerical rank $q$ and the reconstruction of held-out $(y,u)$ pairs is
   $\le 10^{-8}$ relative. If it is not accepted at a given $q$, part 2 is reported as
   **not determinable from saved data** at that $q$ and the cause is ranked on part 1 alone.

**Decision rule.** Cause (2) is SUPPORTED iff the on-test weak residual at $q=16$ is
systematically *below* $q=0$ (median per-step ratio $< 1$) while the evolved field error is
higher, AND — where `heldout_tests` is available — the held-out-to-on-test residual ratio is
strictly larger at $q=16$ than at $q=0$. It is REFUTED if the on-test residual at $q=16$ is
not below $q=0$, or if the held-out ratio does not worsen with $q$.

### Cause (3) — quadrature

Computed: `dense_vs_eq`, every metric ($t_0$, per-time, evolved, all-times) for the matched
dense/EQ twins at the same $q$ and the same $M$: `cclad01` `q{0,16,32,64,128}_m256_{dense,eq}_block`
and `btq201` `q{0,128}_M256_{dense,eq}_g1em06`.

**Decision rule.** Cause (3) is SUPPORTED iff the $q=16$ evolved regression is present in
the EQ arms and ABSENT ($\delta \le 0$) in their dense twins. It is REFUTED if the dense
twins show the same regression.

### Cause (4) — noise

Computed: `per_case_delta`, the six per-case evolved deltas $\delta_c$ and their sign count;
and `equivalent_arm_spread`, the spread of $\varepsilon$ between arms that are *numerically
the same computation by construction* (at one $q$ and $M$: `block` vs `joint` vs `varpro`
vs `alt` in `cclad01`; `base` vs `pre` vs `damp` vs `predamp` in `btq101`). That spread is
the yardstick for what "within noise" means in this cell.

**Decision rule.** Cause (4) is SUPPORTED iff the $q=16$-minus-$q=0$ evolved delta is
positive in $\le 3$ of the 6 cases, **or** if $|\delta|$ at the worst case is smaller than
the equivalent-arm spread at the same $q$. It is REFUTED if the sign is consistent across
$\ge 5$ of 6 cases and the magnitude is above that spread.

### Question (5) — where else does the non-monotonicity appear

`census`: for every job and both metrics, the full sign table of
$\varepsilon(q_{i}) - \varepsilon(q_{i-1})$ along each job's own ladder — `btq201`
(EQ, $M=256$ fixed, budget 180), `cclad01` (both quadratures, budget 180, the $m256$/$m4$/$m2$
test-count rules), `qlad01` (the joint-solver ladder, dense). Reported as a table, not a
verdict: the question is only whether the regression is at $q=16$ alone, whether it
recurs at $q=32$ or $q=64$, and whether it survives the change of solver, quadrature and
test-count rule.

### Second question — where the $q=512$ EQ rung's 3.65 % lives

$q=512$ is the degenerate endpoint $q=R$: the corrections span the whole bank. Its EQ arm
reports 3.65 % against 0.60 % for its dense twin, with a valid untruncated rule.

Computed:

1. `q512_per_time`, `q512_per_case`: where in time and in which cases the error sits;
   $t_0$ compression separated out.
2. `quadrature_functional`: the EQ rule approximates **only** the advection term
   (`arms.weak_eq`: the mass and Laplacian terms go through $A = \Phi^\top G$ exactly). So
   its error is
   $$ \rho(u) \;=\; \frac{\bigl\lVert \sum_{j=1}^{m} w_j\, \Phi(x_j)\, a(u)(x_j) \;-\; \Phi^\top a(u) \bigr\rVert_2}{\lVert \Phi^\top a(u)\rVert_2}, $$
   with $a(u)$ the upwind advection field of `engines.spatial` — a functional of a **single**
   field, so it is computable on every saved field without any internal state. $\rho$ is
   evaluated for each rung's own rule (nodes and weights are archived per arm in
   `arm_setup.eq_indices` / `eq_weights`) at: the arm's own $t=0$ state, its evolved states
   $t_1..t_5$, and the converged FOM states at the same times.
3. `node_support`: for each output time, the fraction of $\sum_x |a(u)(x)|$ that lies in the
   cells the rule actually samples, and the fraction of the top-1 % $|a(u)|$ cells (the
   advection front) within one grid cell of a rule node.

**Decision rule.** "The rule does not cover the reachable states" is SUPPORTED iff
$\rho$ at the evolved states is $\gg \rho$ at $t=0$ for the $q=512$ rule **and** is larger
than the $q=128$ rule's $\rho$ on the same states, with `node_support` falling as the front
develops. It is REFUTED if $\rho$ for the $q=512$ rule is comparable to the lower rungs' at
the same states, in which case the error must be attributed to the degenerate $q=R$
geometry (the solve, not the rule) and that is what will be reported.

## Deliverable

`experiments/q-diag/reports/2026-09-16-q16-regression-diagnosis.md`, generated end to end by
`experiments/q-diag/reports/generate_q_diag.py` from the run JSONs and saved fields beside
it — no number typed by hand — with a per-time error figure for $q = 0, 16, 32, 64$ in both
quadratures, a one-paragraph ranked verdict, and a glossary defining every column and every
term of art. Intermediate NumPy products are written to
`experiments/q-diag/checks/*.json` and are what the report reads.

## Declared limits of this lane

- No new solve is run: no rung is re-executed, no direction is refitted, no checkpoint is
  retrained. Any claim that needs a new ROM solve is reported as not determinable here.
- `local_defect` runs the full-order operator in NumPy on already-saved fields. It is a
  full-order *propagation of ROM states*, not a ROM solve, and its self-check gate above is
  what licenses it.
- The recovered $C_q$, if accepted, is a numerical reconstruction from saved fields, not the
  job's own array (which is archived nowhere); its acceptance threshold is stated above and
  its measured reconstruction error is reported next to every number that depends on it.

## Amendments

Amendments are appended below, dated, before the affected number is computed.
