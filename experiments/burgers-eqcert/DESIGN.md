# burgers-eqcert — DESIGN (pre-registered 2026-09-21, before any GPU job)

Lane of the 2026-09-20/21 speed-and-accuracy campaign; binding contract
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (on main). Worktree
`worktrees/2026-09-21-burgers-eqcert`, branch `exp/2026-09-21-burgers-eqcert`, forked from
`exp/2026-09-20-hires-burgers` @ `0ab60014`. Cluster namespace `/cluster/tufts/paralab/tawal01/bcert_20260921/`,
one directory per job. Budget: at most 2 running and 8 total GPU jobs for this lane; no submit while the account
already has 6 running.

## 1. Question

The paper's accurate Burgers 2D rows at $256^2$ and $512^2$ use b-eqtop's stored $q=256$ rule
(`rule_q256_m2560_bet101_fs64`, *marginal*: 1 of 5 construction draws pass), and the $1024^2$ row uses the dense
(exact) residual. For the frozen checkpoint `sep_hfit_dense_mid_N256_dense.pkl` ($K=16$, $R=512$) at the accurate
setting $q=256$, $M=1088$:

1. Is there an empirical-quadrature (EQ) rule at $256^2$, $512^2$, $1024^2$ that is **certified**: held-out
   $\rho_{\max}\le 0.116$ on solver-reached states of trajectories disjoint from every fit and every evaluation
   case, **confirmed on every one of five independent held-out draws**?
2. With the certified rule (or, if none, with the exact residual), what are the worst evolved error on the six
   development cases and the speedup against the named Newton–BiCGStab FOM, chosen by the paper's rule, in the
   same allocation?

## 2. What exploration already showed (local GB10, not results; `explore/`)

On the **initial-fit states $w_0$** of 40 trajectories (`params_draw(0,128)` indices 8–47), uniform sub-lattice
rules at $q=256$, $M=1088$ have $\rho_{\max}$ (`explore/w0_lattice_rho_*.json`):

| $L$ | lat32 | lat64 | every 4th / 8th node | every 2nd node |
|---|---|---|---|---|
| 256 | 0.632 | 0.516 | — | 0.456 (lat128) |
| 512 | 0.350 | 0.295 | 0.949 (lat128) | 0.210 (lat256) |
| 1024 | 0.196 | 0.222 | 1.205 (lat128), 0.413 (lat256) | not run |
| 2048 | 0.121 | 0.183 | 1.311 (lat128), 0.494 (lat256) | not run |

The worst trajectories are 18 and 22, which hires-burgers never sampled (its certificate used 8–15, where lat64
gave 0.100 at $2048^2$). **So lat64's certificate at $2048^2$/$4096^2$ is a single-draw result**; this lane does not
re-certify those meshes unless budget remains (§8), but reports the exploration.

On the **evolved** states of the deployed lat64 query at $256^2$ (`explore/evolved_lattice_rho_L256.json`,
trajectories 22, 18, 15, 9): the worst evolved state is always $k=1$ (0.139 on trajectory 22 for lat64, 0.078 for
every-2nd-node), and $k\ge 2$ is $\le 0.033$. The $\rho$ failure is the initial transient, not the evolution.

These trajectories (`params_draw(0,128)`) are **never used** by any certificate in this lane (§4), so the choice of
arms below is not tuned on the certification populations.

## 3. Arms

ROM arms, all on the `hfast` kernel with Cholesky, trust clipping, damping carry-over and the `pred2` predictor
(hires-burgers' kept speed stack; each measured directly, no parity claim), LM stationarity tolerance
$g=10^{-3}$ unless stated:

| rung | rule | exact steps $j$ | note |
|---|---|---|---|
| $q=0$, $M=64$ | `scaled` (b-eqtop $m=1024$, confirmed 3/3 at $256^2$; weights $\times(L/256)^2$) | 0 | fast setting |
| $q=128$, $M=576$ | `scaled` (b-eqtop $m=2319$, single draw) | 0 | the paper's "validated $q=128$" rule |
| $q=128$, $M=576$ | `lat64` | 0, 1 | |
| $q=256$, $M=1088$ | `scaled` (b-eqtop `fs64` $m=2560$, marginal 1/5) | 0 | the paper's current rule at $256^2$/$512^2$ |
| $q=256$, $M=1088$ | `lat64` ($63^2=3969$ nodes, equal weights) | 0, 1, 2 | |
| $q=256$, $M=1088$ | `lathalf` (every 2nd node, $s=L/2$) | 0, 1 | |
| $q=256$, $M=1088$ | `exact` (all-node advection through the separable projection, no rule) | — | fallback; needs no certificate |
| $q=256$, $M=1088$ | `bad0` (q-ridge $m=2048$ rule, original weights) | 0 | **control: must fail** |

Tolerance arms $g=10^{-2}$ (labelled) for `lat64` $j=1$, `lathalf` $j=1$ and `exact`.

**Exact-first-steps arms ($j\ge1$), new in this lane.** The first $j$ backward-Euler steps are solved with the
exact residual (same LM, same stopping rule, same predictor), the remaining $50-j$ with the EQ rule. The rule is
then evaluated only at the per-step states $w_k$, $k\ge j$ (and at LM iterates and predictor candidates between
them, which no certificate in this project samples — unchanged convention). This is a different online method,
labelled as such everywhere, not a relaxation of the bar: the bar is applied to **every** state at which the rule
is evaluated, exactly as for $j=0$, where that set includes $w_0$ (the LM of step 1 starts there).

FOM arms (same allocation, `engines.make_fom` "lean" unless marked audited): `fft_tight` (audited; same-grid
truth), `lean_tight`, `nt1e-4_dt005` and `nt1e-3_dt005` (audited, the paper's earlier comparators),
`lean_nt1e-4_dt005`, `lean_nt1e-3_l1e-3_dt005`, `lean_nt3e-3_l3e-3_dt005`, `lean_nt1e-2_l1e-2_dt005`,
`lean_nt1e-3_l1e-3_dt01`, `lean_nt3e-3_l3e-3_dt01`, `lean_nt1e-2_l1e-2_dt01`, `lean_nt1e-3_l1e-3_dt0125`,
`lean_nt3e-3_l3e-3_dt0125`, `lean_nt1e-3_l1e-3_dt025` (the larger steps give the FOM every chance to be cheap at
an error near the ROM's). Coarse-grid FOM at $L/2$, $L/4$ (nt1e-4, dt 0.005), scored against a refined reference
at $4L$, $\Delta t/16$ (protocol-required control; the paper excludes it by user decision). Dense truth per case
for $q=256$, $M=1088$ (exact residual, $g=10^{-6}$) to separate quadrature from representation error.

## 4. Certificate

$\rho(u)=\lVert P_q^\top a(u)-\Phi^\top a(u)\rVert/\lVert\Phi^\top a(u)\rVert$, primary bar **0.116**, tight 0.06
(inherited, unchanged). Never the NNLS fit residual.

**Populations — fresh, never explored.** Parameters `params_draw(20260921, 40)`, asserted disjoint (no equal row)
from the evaluation cohort and from hold64; split into **five held-out draws** $H_1\ldots H_5$ of 8 trajectories
each (indices $8d \ldots 8d+7$). No rule in this lane is fitted, so there is no fit population; the stored b-eqtop
/ q-ridge rules were fitted on other seeds.

Two populations per draw, both at the target mesh $L$, every per-step state tagged by its step index $k=0\ldots50$:

- *dense-query population*: the audited dense query (`topfix.make_query(..., 'dense', 'base')`, $g=10^{-6}$) at
  the rung's $(q, M=4(K+q))$ — rule-independent;
- *deployed population*: the states the arm's own EQ query visits — per arm.

**Arm status** ($j$ = the arm's exact steps; $\rho$ taken over states with $k\ge j$):

- **confirmed** — $\rho_{\max}\le 0.116$ on all five draws in both populations;
- **marginal ($n$/5)** — some but not all draws pass (both populations per draw);
- **fails** — no draw passes;
- `exact` — no rule, status "exact residual".

$\rho$ over all $k\ge0$ is reported beside every $j\ge1$ arm, and $\rho$ over $k\ge1$ beside every $j=0$ arm, as
diagnostics; neither changes a status.

**Control:** `bad0` must fail at every mesh (controls-must-fail rule). If it certifies, the certificate is not
discriminating at that mesh and the mesh's verdict is reported as untested.

## 5. Selection rule and bar (fixed now)

Per mesh, on the six development cases dev6 (`params_draw(7090702,4)+params_draw(911702,2)`, SHA256 `108f12dc…`,
opened), error $\epsilon=\max_{k\ge1}\lVert u_k-u^{\mathrm{tight}}_k\rVert/\lVert u_0\rVert$ against the same-job
`fft_tight`, worst over cases:

- **Certified accurate row** = the cheapest (median GPU ms) **confirmed** non-control $q=256$ arm with worst evolved
  error $\le 1\,\%$. If none: the row is reported as *no certified rule*, and the `exact` arm (cheaper of its two
  tolerances with error $\le1\,\%$) is reported as the exact-residual row, labelled.
- **FOM** = the fastest tested FOM arm (median GPU ms) whose every time step converged and whose worst evolved error
  is $\le$ the chosen ROM arm's (the paper's rule). Speedup $S=T_{\mathrm{FOM}}/T_{\mathrm{ROM}}$, GPU-query scope,
  medians of 5 repetitions × 6 cases; complete-query (host-inclusive) ratio beside it.
- **Fast row** = the $q=0$ arm, its own FOM by the same rule.
- **Lane bar (protocol):** error $\le 1\,\%$ and $S\ge 5$. At these meshes earlier panels put the accurate ROM at
  $0.004$–$0.07\times$, so a speed miss is expected and is reported, not softened.

The certification verdict (question 1) is separate from the speed verdict and is the lane's primary result.

## 6. Jobs

One job per mesh, each self-contained (cohort, truth, populations, certificates, quick answer, dense truth, timed
panel, refined reference last under a deadline): `bc256`, `bc512`, `bc1024`, A100-80G (H200 if needed), 5
repetitions, 0.25 s burn-in, randomised order, `block_until_ready`, GPU name + UUID recorded, preflight exit 42.
At most 2 running. Spare budget (≤ 5 jobs) is reserved for failures and, only if the three verdicts are in and time
remains, a certificate-only job at $2048^2$ for `lat64` $j\in\{0,1\}$ on the same five draws.

## 7. Integrity

f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, Φ-free operator parity at $L$ (≤1e-12), evaluation-cohort SHA256
asserted, repetition outputs identical (full-field SHA256), every FOM row with converged/stalled step counts,
every ROM row with exit reasons. After each job: checksum-verified pull, independent NumPy re-computation of every
error from saved fields **and of $\rho$ from saved per-state coefficients for the headline rule** (restricted
fields for every arm, full fields for case 0), remote directory deleted, small `summary.json` with source hashes.

## 8. Stop rules

Stop when the three meshes are audited, or 8 jobs are spent, or the 2026-09-24 12:00 EDT hard stop. No arm is
added after seeing a certification result without a dated addendum here written before its job.

## Glossary

**EQ (empirical quadrature)** — the advection term's weak projection $\Phi^\top a(u)$ approximated by a weighted sum
over $m$ grid nodes. **ρ** — relative error of that approximation on one state. **Certified / confirmed** — ρ_max
≤ 0.116 on every held-out draw. **Held-out draw** — 8 trajectories never used to build or choose any rule and
disjoint from the evaluation cases. **w₀ / k** — the initial-fit state / the time-step index of a per-step state.
**Exact steps j** — number of initial time steps solved with the exact (all-node) residual before the rule takes
over. **lat64 / lathalf** — equal-weight uniform sub-lattices with 64 intervals per axis / every second grid node.
**q, M** — number of linear correction directions / number of sine test functions. **dev6** — six opened
development cases. **Evolved error** — worst relative error over output times after t = 0. **FOM** — backward-Euler
upwind finite differences solved by Newton with FFT-preconditioned BiCGStab.

## Addendum A1 (2026-09-21, after the Codex design audit, before any GPU job)

Audit: `reports/codex-design-audit-2026-09-21.md` (gpt-6-astra, read-only, files inlined; 12 findings). Dispositions:

1. **(blocker) What the certificate covers.** Accepted as a wording correction, as in every earlier lane: the
   certificate samples the **per-step (converged) states** $w_k$ only, never LM trial iterates or predictor
   candidates. §3's sentence "the bar is applied to every state at which the rule is evaluated" is withdrawn and
   replaced by: *for an arm with $j$ exact steps the rule is certified on the per-step states $w_k$, $k\ge j$ — the
   per-step states at which the rule is first evaluated — and, like every arm, not on LM iterates or predictor
   candidates.* Reports carry this sentence wherever an $x_j$ arm is quoted.
2. **Exact-step hybrid.** Accepted: the whole hybrid query is timed (initialisation, exact steps, EQ steps, decode);
   damping and predictor history cross the phase boundary (`xfast.py`); new audit gate
   `exact_phase_matches_exact_arm` requires $w_0\ldots w_j$ of every $x_j$ arm to equal the exact arm's at the same
   tolerance to $10^{-10}$ (same initialiser, same exact LM). "Exact" means exact residual evaluation.
3. **Selection on the certification draws.** Accepted. A **confirmation draw** of 16 further trajectories
   (`params_draw(20260921,56)` indices 40–55) is computed in-job for every arm but is **not** used to select: the
   headline arm is chosen on draws 1–5 by §5; it is then reported *certified* only if it also passes the
   confirmation draw (both populations). If it fails, the mesh has no certified rule; the next candidate is **not**
   promoted.
4. **Statistical strength.** Accepted as reporting: zero failures in $n$ trajectories bounds the per-trajectory
   failure probability by $\approx 3/n$ at 95 % (40 → 7.5 %, 56 → 5.4 %); reports say "empirical coverage on $n$
   held-out trajectories", not a guarantee.
5. **Control semantics.** Specified: `bad0` must not be *confirmed* (gate), and its dev6 error is reported beside
   the exact arm's (it regressed 1.04 % vs 0.59 % at $2048^2$); a confirmed `bad0` voids that mesh's certificate.
6. **Error metric.** Named precisely: worst over the five output times $t=0.05\ldots0.25$ of
   $\lVert u_k-u_k^{\mathrm{tight}}\rVert/\lVert u_0\rVert$ (the paper's metric); the worst current-relative error
   is reported beside it.
7. **FOM eligibility.** Cohort-level (worst over cases), as the paper states. Made symmetric: an eligible ROM arm
   must also have zero stalled exits.
8. **Refined reference.** It only serves the coarse-grid control, which the paper excludes; a missing reference
   marks that control *incomplete*, not the verdict. Per the protocol's sizing rule, `bc512` (reference $2048^2$)
   and `bc1024` (reference $4096^2$) request H200 + 240G; `bc256` A100-80G.
9. **Full-grid audit.** Accepted: full fields for **all six cases** of every FOM arm and every decision-relevant
   ROM arm are saved and recomputed (at $256^2$ the restricted grid is the full grid).
10. **Acceptance predicate.** Accepted: the audit's verdict carries `accepted = no failed audit gate`.
11. **Speed stack parity.** `hfast` vs the audited base passed parity at $\le10^{-13}$ with identical integers in
    hires-burgers `hb2k01`; Cholesky/clip/lamcarry/pred2 are algorithmic variants whose error is measured directly
    (their same-job before/after timings are hires-burgers SPEED-LOG rows). Not re-litigated here; the new code
    path is gated by item 2.
12. **Scope.** Accepted: certification, development accuracy and speed are separate verdicts; the exact-residual
    fallback is never called a certified rule; dev6 results are development evidence.
