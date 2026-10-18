# jcp-heads — linear span against the nonlinear head, and error and cost against the head's latent size $K$ (C5; pre-registration)

**Status:** pre-registration, 2026-10-18, written before any lane job. Branch `exp/2026-10-18-jcp-heads` (fork
`21c175a1b`), lane dir `experiments/jcp-heads`, cluster namespace `/cluster/tufts/paralab/tawal01/jcp-heads/`.
Governing documents: `reports/2026-10-17-full-experiment-plan/` (PLAN-v2 C5; v3/v4 amendments win where they differ),
`campaign/briefs/{COMMON-BRIEF,jcp-heads}.md`, `reports/2026-10-08-jcp-lane-agent-rules.md`, governing plan E3c/F5/K5
(`reports/2026-10-06-jcp-offmesh-paper-plan.md:87,98,154`). Accuracy numbers are E0-conditional; $\varepsilon_{\rm ref}$
is a conditional Richardson estimate. Amendments are appended below, dated; nothing above them is rewritten.

## 0. Questions and outcomes

The advisor asked whether cost depends on "k". Priorities 1–2 read k as the bank width $r$. This lane reads it as $K$,
the latent size of the nonlinear head, and answers PLAN-v2 C5: the headline results use the linear span, so the claim
for the head is either measured under pre-registration or the paper drops "nonlinear manifold" (F5).

| id | question | registered outcome labels |
|---|---|---|
| Q-K-err | does the head ROM's E0 error fall as $K$ grows over $\{4,8,16,32\}$? (1024²) | per transition improves / worsens / tie / inconclusive / unqualified; range "monotone" or not |
| Q-K-cost | how does online cost grow with $K$, and is it mesh-independent at fixed $K$? (256², 1024², 4096²) | per $K$: H-cost-N pass / fail / inconclusive; cost-vs-$K$ table and fit (descriptive) |
| Q-LH-dim | at matched online dimension ($r=K$), which ROM dominates, head or linear span? (1024²) | head dominates / linear dominates / neither (trade-off) / unresolved / unqualified |
| Q-LH-cost | at equal online cost, which is more accurate? (1024²; 256² deferrable) | head more accurate / linear more accurate / matches (TOST) / unresolved / unqualified |
| Q-repro | does a matched retrain of the incumbent recipe reproduce the incumbent $K=16$ head? | reproduces / does not reproduce (§2.4) |
| F5 consequence | pre-declared reading for the paper (§5.6) | "nonlinear-manifold claim supported at $K$" / "not supported by validation evidence" |

The sealed-cohort test of K5 is **not** run here (global freeze, §F of the plan). This lane produces validation evidence
only (dev6 + val32).

## 1. The reduced model

**Bank (frozen, shared by every arm).** The dn256b coordinate-network bank, $R=512$ features
(`experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl`, sha256 `18f0266a…`; only its bank
parameters are used), rotated by `vendor/quad2d/inputs/rotation_R512.npz` (sha256 `51149166…`): $\tilde G = G T$, with
the ordered prefixes used by every campaign lane.

**Head ROM ($K$).** $u(z)=\tilde G\,\tilde h_\theta(z)$, $z\in\mathbb R^K$, where $\tilde h_\theta = L\,h_\theta$ is the
head folded into the rotated basis (`bkfast.fold_head`, exact), $h_\theta:\mathbb R^K\to\mathbb R^{512}$ a 512×2 SiLU MLP
plus a linear skip $zW_{\rm lin}$ (`sep_hfit.head_apply` = `sep_common.head`). Linear-correction rank $q=0$ (the deployed
head setting; quadrature-study `qcore.SETTINGS['head']`). Unknowns: $K$.

**Linear-span ROM ($r$).** $u(c)=\tilde G_{[:, :r]}c$, $c\in\mathbb R^r$ (the deployed linear rung, P1/qualify).

**Residual (both).** The two-step LSPG residual of P1/qualify (`p1core.make_evolve`, form LSPG) in the coefficient
$c$, with $c=\tilde h_\theta(z)$ for the head:
$$
\mathcal R_n(c)=A\,(a_0c+a_1c_n+a_2c_{n-1})+\Delta t\,\big(b_0F(c)+b_1F(c_n)\big),\quad F(c)=N(c)+\nu\Lambda Ac,\quad
\min\ \lVert D\,\mathcal R_n\rVert,\ D=(a_0+\Delta t\,b_0\nu\Lambda)^{-1},
$$
$A=\Phi^{\mathsf T}\tilde G$ (exact discrete projection, mesh-dependent), $\Lambda$ the discrete eigenvalues, $M$ sine test
modes, and $N(c)=L\sum_q w_q\psi(x_q)\,u(u_x+u_y)(x_q)$ the off-mesh tested advection (point form, `qcore.tested_value
('point')`) on a Gauss tensor rule. Head Jacobian: $J_z=(A+\Delta t\,b_0 S\,\partial N(c))\,\partial_z\tilde h$, assembled
analytically with the rule's tangent block (`qcore.tested_jac('point')` with $T_m=\partial_z\tilde h$, $K$ tangents);
the in-job certifier uses forward-mode autodiff instead (§3.1).

**Test space.** $M=4K$ for the head and $M=4r$ for the linear span (the deployed ratio; the incumbent head is $K=16$,
$M=64$). Mode list: the frozen $1024^2$ ordering (V4-1, P1/qualify `mode_list('frozen')`), reused at every mesh. For
$M\le1024$ (every head, linear $r\le256$) the frozen and nominal lists coincide (V4-1); the lane records the symmetric
difference for every $(M,\text{mesh})$ it uses (non-empty only for linear $r=384$, $M=1536$, at 256²/4096²).

**Schemes.** LSPG-BE at $\Delta t_0=0.005$ (**primary**: the head's deployed scheme; the only scheme with a matched S
reference) and LSPG-CN-R at $\Delta t_0$ (secondary: the campaign's second-order scheme, two BE start-up steps). GAL is
not defined for a nonlinear head with $M>K$ and is not run.

**Solver (head).** The deployed head query (vendor `hfast.make_query` + quadrature-study `make_head_query`): fused LM,
LU solve, step clipped to the trust radius, damping carried across steps, quadratic predictor (best of three by residual),
budget 600, production tolerances gtol $10^{-3}$, tol $10^{-9}\cdot$scale; trust $=0.01\max_i\lVert z_i-\bar z\rVert$
over the head's own training codes. **Initial fit:** the supplied full-field IC sampled at the $48^2$ Gauss points,
nearest candidate among $\le 8192$ subsampled training codes (stride $\lfloor S/8192\rfloor$, as quadrature-study), then
the stationary LM in $z$ (GJ solve, budget 400, gtol $10^{-6}$). **Solver (linear):** qualify's LSPG driver unchanged
(Cholesky, clip, damping carry, quadratic predictor, closed-form Gauss-48 initial projection, `trust_linear(r)`).

## 2. Matched heads: training

### 2.1 Recipe (identical across $K$ and seeds; only $K$ and the seed change)

The incumbent head was produced by stage C of dn256b (job 2837431): `sep_hfit_run.py` arm `mid` on the frozen bank
(`run_dn256b.sbatch:58-62`). The lane copies `sep_hfit.py` and `sep_hfit_run.py` (provenance-hashed) and runs arm `mid`
with the latent size overridden:

- head 512 × 2, SiLU, linear skip; init `sep_hfit.init_head` (He-normal, skip scale 0.3);
- codes: free auto-decoder variables, init $0.1\,\mathcal N(0,1)$;
- objective: the exact whitened coefficient objective $\lVert L^{\mathsf T}(h(z)-c^*)\rVert^2+f^2$ (identity (*) of
  `sep_hfit.py`), per-snapshot normalisation `snap`;
- Adam, warmup-cosine $0\to10^{-3}\to10^{-5}$ (500 warm-up steps), **200,000 steps**, batch 4096, no weight decay, no EMA;
- RNG `PRNGKey(SEED0 + 11)`; encoder for the oracle diagnostic 12,000 steps, `PRNGKey(SEED0 + 7)`;
- data: `sep_coeff_N256_K16_R512.npz` (sha256 `4a771565…`), the incumbent's own training file: 131,072 states of
  4,608 training trajectories (576 canonical seed 0 + 4,032 seed 1000; all $t\le 5$ states plus a random pick; 256²),
  coefficients $c^*$ of the frozen bank, Gram matrix, norms; 408 fresh pipeline-test states (8 trajectories, test seed 1).

**Matrix:** $K\in\{4,8,16,32\}\times$ SEED0 $\in\{0,1,2\}$ = 12 heads, plus the incumbent (unmatched reference).
Every head records: recon (train), the representation oracle on the 408 pipeline-test states (diagnostic, uses their
truth; never enters a solve), oracle/span floor, code-convergence gain, wall time, GPU, job id, sha256 of the emitted pkl.

### 2.2 Data handling (deviation from "regenerate data from the seed", stated)

The coefficient file is a frozen, hashed training artifact (567 MB). It is uploaded to the namespace and sha256-checked
in the job, not regenerated: regeneration needs 4,608 Burgers solves plus the bank projection (stage B, 915 s plus data
generation) and would itself be a reproduction risk, and "same data" is the point of a matched sweep. Disjointness:
every dev6 / val32 descriptor is asserted absent from `mu_tr` of all 4,608 training trajectories.

### 2.3 Training controls

- **Driver identity (positive).** The copied driver run on the incumbent with zero steps (arm `base`) reproduces the
  incumbent's recorded recon (train mean $7.5072\times10^{-3}$) and oracle (mean $5.2006\times10^{-3}$) from
  `hfit_dn256b_full.json` to $10^{-6}$ relative (data, whitening and metric integrity). The whitening round-trip gate of
  the driver ($<10^{-10}$) must pass.
- **Under-trained head (negative).** $K=16$, SEED0 = 0, 2,000 steps (1 % budget): its oracle must be $\ge1.5\times$ the
  full-budget head's. If not, the training budget does not discriminate and Q-K-err is reported as "training budget
  not shown to matter".

### 2.4 Reproduction of the incumbent (Q-repro)

Retrains are not bitwise reproducible (GPU reductions). $K=16$, SEED0 = 0 is the like-for-like retrain. Label
**reproduces** iff (i) its pipeline oracle mean is within ±15 % of the incumbent's and (ii) its ROM median E0 ST error
(LSPG-BE $\Delta t_0$, 1024², all eligible cases, its own qualified rule) is within a factor 1.25 of the incumbent's
evaluated in this lane under the same contract. Otherwise **does not reproduce**: the K sweep is then "internally
matched only" and the incumbent is reported beside it as an unmatched point. Prior evidence: b-seeds reran the dn256b
scripts at seeds 1–3 and reproduced (best-found worst 2.86/2.54/2.57 % vs 2.54 %); b-head-train's different pipeline
did not (1.56× worse). The seed spread (max/min over SEED0 of the median error, per $K$) is reported for every $K$.

## 3. Qualification (per configuration; common brief §3 / V4-2)

A configuration key: (arm head-$K$-seed or linear-$r$, $M$ and mode-list hash, rule, mesh, scheme, $\Delta t$, tolerance
level, solver, budget, trust, initial fit, head/bank sha256).

### 3.1 Q1 — solver, certified in-job at every step with a fresh Jacobian

Every production rollout records its internal trajectory ($z_0..z_S$ for the head, $c_0..c_S$ for the linear span).
1. every run valid: finite outputs and states, no non-finite step, no damping-saturation exit (reason 3); budget exits
   counted and judged by the certificate;
2. per-step stationarity $\gamma_n=\lVert J^{\mathsf T}r\rVert/(\lVert J\rVert_F\lVert r\rVert+10^{-300})$ rebuilt from the
   saved states (scheme coefficients incl. start-up, history $Ac_n$, $Ac_{n-1}$, $F(c_n)$ with $c=\tilde h(z)$) with a
   **forward-mode autodiff** Jacobian in the unknowns ($z$ or $c$); non-stationary if $\gamma_n>10^{-3}$ or non-finite;
   non-stationary steps ≤ 1 % pooled over all steps of all cases;
3. scoring-lattice relative distance to the adaptive **tight** solve of the same configuration ≤ $2.5\times10^{-4}$ on
   every case. Tight = gtol $10^{-7}$, tol 0, step budget 2000, and for the head also the initial fit tightened
   (gtol $10^{-10}$, budget 2000) from the same start; an invalid tight anchor makes the case unqualified;
4. tight/tighter (gtol $10^{-8}$; head initial fit $10^{-11}$) agreement ≤ $2.5\times10^{-5}$ on dev6;
5. G-cert: certified $\gamma$ equals the solver's recorded exit $\gamma$ to $10^{-7}$ relative $+10^{-12}$ on every step;
   offline NumPy recomputation of every distance agrees with the in-job value ($10^{-9}$ rel. + $10^{-13}$).

The head initial fit's own stationarity ($\gamma_{\rm ic}$, iterations, exit reason) is recorded per case (descriptive).

### 3.2 Q2 — quadrature, on the configuration's own reached states

Rule ladder $\mathcal L$ = Gauss $p^2$, $p\in\{16,24,32,48,64,96,128,192,256\}$; successor $q^+$ = next rung,
$q^{++}$ = the one after. Population sub16 (dev6 + val32[0:10]). Rollout bar: replay with $q^+$, max over the five evolved
outputs of the scoring-lattice relative distance ≤ $2.5\times10^{-4}$ on all 16 cases. ρ bar: on the configuration's own
reached coefficient states $c=\tilde h(z)$ (6 outputs × 16 cases), $\rho=\lVert N_q(c)-N_{q^+}(c)\rVert/\lVert N_{q^+}(c)\rVert$,
$\rho_{\max}\le0.116$ (borrowed, quadrature-study `DESIGN.md:234`). Finer-rule convergence on dev6 ($q^+$ vs $q^{++}$
rollout bar), else "unverified". **Numerically qualified** = Q1 pass and Q2 pass.

**Rule per configuration (frozen before any accuracy job).** Job `rule1024` (§7) runs, at 1024² with LSPG-BE
$\Delta t_0$ on sub16, the ladder for every head ($K$, seed) and every linear rung $r\le64$; the rule of a head size $K$
is the smallest $q\in\mathcal L$ whose Q2 (both bars, $q\to q^+$) passes for **all three seeds** (so the seeds of one $K$
share a rule and a cost); for linear $r\le64$ the smallest passing $q$. Linear $r\ge128$ keep qualify's frozen rules
(R128 `fib4181`, R256 `gauss128`, R384 `gauss128`; FRONT-2D v1) for comparability, re-certified here. The selection
reads only successor agreement (a numerical criterion), never E0 error. If no rung up to Gauss 192² passes for some
$K$, that $K$ is "not quadrature-qualified" and the remaining analysis is reported for it labelled so. Frozen rules are
used unchanged at 256²/4096² and for CN-R, with Q2 re-certified there.

### 3.3 Q3 (descriptive)

Singular values / condition number of the step Jacobian $J_z$ (head) and $J$ (linear) at output indices 1, 3, 5 of each
dev6 trajectory; rank at $10^{-13}$ relative.

## 4. Scoring

dev6 and val32 only (38 cases); no test64, sealed or OOD access; `sealed/` and `custody/` never touched. E0 store via
`refstore` (version pinned in a lane manifest), $e=\max_{k=1..5}\lVert u(t_k)-R(t_k)\rVert/\lVert R(t_0)\rVert$ on the
$257^2$ lattice; eligibility per case-kind from the store; ST primary; S only for LSPG-BE at $\Delta t_0$ (labelled);
ties $|\Delta e|\le2\varepsilon_{\rm ref}$; every record asserts its identity (cohort, case, time tag, kind, reference
sha256, store version, metric id, $\Delta t$). Cases are the statistical unit. Post-hoc labels (E0 A10; combined A10,
A11, A15) carry forward.

## 5. Hypotheses and verdict procedures (frozen)

Statistics reuse P1 `p1stats.py` (Wilcoxon signed-rank exact for $n\le50$ without ties, TOST, case bootstrap 10,000,
seed 20261017). Every procedure is computed separately for BE and CN-R; **BE is primary** for every headline sentence.

### 5.1 Per-case head error for a size $K$

Per case $i$, $e_{K,i}$ = median over the numerically qualified seeds of $K$ of the seed's E0 ST error (needs ≥ 2
qualified seeds, else $K$ is "unqualified" for the pooled analyses); each seed's own analysis is also reported.

### 5.2 Q-K-err (H-acc-K)

Transitions $4\to8$, $8\to16$, $16\to32$ at 1024², P1 AM-7 procedure on cases eligible and qualified at both sizes: a case
is resolved if $|\Delta e_i|>2\varepsilon_{{\rm ref},i}$; **improves** iff resolved coverage ≥ 50 % of eligible cases,
> 50 % of resolved cases improve, and the median paired change < 0; **worsens** symmetrically; **tie** if coverage < 50 %;
otherwise **inconclusive**. "Monotone over 4–32" iff every transition improves. Reported per transition: improve /
worsen / tie / unqualified counts, median and worst paired change; plus each seed's own transitions (sensitivity).

### 5.3 Q-K-cost

**H-cost-N for the head (per $K$, BE, seed 0):** P1 `h_cost_n` on the evolution-only cost $c_{\rm ev}$ (median of the
paired $t_{\rm ev}/\bar t_A$ over dev6 × 3 repeats) at 256²/1024²/4096²: ratio $\max_N/\min_N$, case-bootstrap 95 %
interval; pass iff upper ≤ 1.25, fail iff lower > 1.25, else inconclusive; two-mesh evidence if 4096² is deferred
("registered three-mesh verdict not reached"). **Cost vs $K$ (descriptive):** per mesh, $c$ (fused online query) and
$c_{\rm ev}$ against $K$, per-iteration cost ($t_{\rm ev}$ / total LM iterations) and iterations per step; fit
$t(K)=a+bK^p$ (NLS, case bootstrap); no verdict. The operation model for the head step is
$O(mRK + mMK + MK^2 + K^3)$ plus the MLP and its $K$ tangents ($O(K\cdot 512^2)$); stated, not tested.

### 5.4 The paired comparison $X$ (head) vs $Y$ (linear) on error and cost

On the cases eligible and qualified for both: $d_i=e_{X,i}-e_{Y,i}$, margin $\delta_i=\max(2\varepsilon_{{\rm ref},i},
\,0.1\min(e_{X,i},e_{Y,i}),\,10^{-12})$.
- **Error:** $X$ **more accurate** iff resolved coverage ($|d_i|>2\varepsilon_{\rm ref}$) ≥ 50 %, one-sided Wilcoxon of
  $d$ for "less" has $p<0.05$, and median $d<0$; $Y$ more accurate symmetrically; **matches** iff TOST (two one-sided
  Wilcoxon tests of $d_i\pm\delta_i$, α = 0.05 each) rejects both; else **unresolved**. Worst error over the cases is
  reported beside it.
- **Cost:** from the same 1024² timing job, per dev6 case the paired ratio $c_{X}/c_{Y}$ (fused online query, both
  against the same A); case-bootstrap 95 % interval of the median ratio: $X$ **cheaper** iff upper < $1/1.1$, **costlier**
  iff lower > 1.1, **equal cost** iff the interval lies in $[1/1.1,\,1.1]$, else **unresolved**.
- **Three-way dominance:** $X$ dominates $Y$ iff error ∈ {$X$ more accurate, matches} and cost ∈ {$X$ cheaper, equal
  cost}, not both "matches" and "equal cost"; symmetrically for $Y$; **neither** iff one is more accurate and the other
  cheaper; **unresolved** otherwise (any component unresolved and no dominance decided); **unqualified** if either
  configuration is not numerically qualified (its numbers still shown, labelled).

### 5.5 Q-LH-dim and Q-LH-cost

- **Q-LH-dim (1024²):** head $K$ (§5.1 per-case seed median; and per seed) vs linear $r=K$, procedure §5.4.
- **Q-LH-cost (1024²; 256² deferrable):** linear cost ladder $r\in\{4,8,16,32,64,128,256,384\}$. The equal-cost rung
  $r_{\rm eq}(K)$ = the largest ladder $r$ whose median fused cost ratio $c_{L_r}/c_{H_K}$ (same job, seed-0 head) is
  ≤ 1.1; if every rung is cheaper, $r_{\rm eq}=384$ and the label "ladder ceiling" is attached. Error comparison §5.4
  (error part) between head $K$ and linear $r_{\rm eq}$, plus the next rung above (bracket), reported. Verdict labels as
  in §0.

### 5.6 F5 consequence (pre-declared reading)

"**Nonlinear-manifold claim supported at $K$ (validation evidence)**" iff, under BE, the head at $K$ is numerically
qualified (≥ 2 seeds), its Q-LH-cost verdict is "head more accurate", and Q-LH-dim is not "linear dominates". If no $K$
meets this, the lane reports "**the head is not supported over the linear span on validation evidence**", which, under
the governing plan's F5/K5, means the paper drops "nonlinear manifold" or presents the head as a documented negative
result. The sealed-cohort K5 test stays with the global freeze.

## 6. Cost measurement

qualify's per-mesh timing protocol (`qtime.py`, P1 §7 + A1.10, A3.2, A4.1), one H200 job per mesh: A–B–A triplets with
P1's reference A ($r=384$, 1024², LSPG-CN-R, $\Delta t_0$, frozen list, `gauss128`) on dev6 × 3 repeats, randomised
order; split triplet (A evolve | B init | B evolve | B decode | A evolve; drift band $[1/1.1,1.1]$, ≤ 2 retries) and
fused triplet (A | B fused | A); each timed execution reproduces its scoring run's identity hash. **Primary cost** $c$ =
median fused $t/\bar t_A$ (online query, full-field contract: input sampling at quadrature points, initial fit, evolution,
full-field decode of the six outputs); $c_{\rm ev}$ evolution only; absolute seconds reported; never compared across GPU
types. Subjects per mesh: 12 heads + incumbent and the linear ladder $r\in\{4,8,16,32,64,128,256,384\}$, LSPG-BE
$\Delta t_0$; at 1024² also LSPG-CN-R for heads (seed 0) and $r\in\{4,8,16,32\}$. Each subject's untimed scoring run is
certified in-job (Q1 on dev6, tight/tighter, $q^+$/$q^{++}$ replays), as qualify.

## 7. Controls (shown to behave in the real-size smoke before any verdict)

| control | kind | fixture and predicate | blocks |
|---|---|---|---|
| head reproduction gate | positive | incumbent head, lane head stepper, LSPG-BE $\Delta t_0$, 1024², Gauss 32² and 64², $M=64$, dev6\|0 and dev6\|2: internal $z$ trajectories equal quadrature-study `dv1024` archive (`audit_head_gauss{32,64}_dev6{0,2}.npz`, key `internal`) to ≤ $10^{-10}$ relative (A100-80G, the archive's GPU type) | every head result |
| linear reproduction gate | positive | lane driver, R384 `gauss128` BE and CN-R, 1024², dev6: coefficients equal qualify `acc1024` archive to ≤ $10^{-10}$ | every linear result |
| head Jacobian | positive | analytic $J_z$ vs `jax.jacfwd` of the head residual at 4 fixtures per $K$: ≤ $10^{-10}$ relative (CPU test and in-job) | head Q1 |
| Q1 negative | negative | incumbent head, BE, 1024², step budget 0 (returns the predictor), dev6: pooled non-stationary fraction > 1 % (independent NumPy γ > $10^{-3}$ on > 3 steps first) | Q1 |
| zero-update | negative | $z_{n+1}=z_n$ on dev6\|0 (incumbent, BE): distance to tight > $2.5\times10^{-4}$ | Q1 |
| under-resolved rule | negative | incumbent head, BE, 1024², Gauss 8² → 16²: Q2 fails (rollout > $2.5\times10^{-4}$ on ≥ 1 case and ρ$_{\max}$ > 0.116) | Q2 |
| driver identity / under-trained head | positive / negative | §2.3 | Q-K-err, Q-repro |
| training disjointness | positive | §2.2 assertion | all |
| wrong-mesh table, case / time / kind / Δt swap | negative | inherited (P1/qualify code paths; a 256² table fed to the 1024² query raises; swapped records raise) | all |
| timing | negative | injected 1.5× fault on one subject flagged invalid (`timing.py` self-test + in-job identity fault); doubled work B/A ∈ [1.8, 2.2] | cost |
| dominance / TOST | negative / positive | CPU synthetics: X 10 % more accurate on every case and 10 % costlier → "neither"; X better and cheaper → "X dominates"; X = Y → "matches" + "equal cost" → neither dominates; X unqualified → "unqualified" | §5.4 |
| mode-list identity | positive | every configuration's mode-list hash equals the frozen 1024² list's prefix at every mesh; symmetric difference to the nominal list reported (§1) | all |

## 8. Jobs, resources, budget

| job | GPU | content |
|---|---|---|
| `smk1` | A100-80G | real-size smoke at 1024²: training pipeline (driver identity on the incumbent; $K=4$ and $K=32$ at 2,000 steps, emitted and loaded), head and linear reproduction gates, head Jacobian test, every Q1/Q2 negative control, a short rule ladder for the incumbent and one new head, all-38 cost probe for one head; ≤ 2 GPU-h |
| `smk2` | H200 | timing smoke at 1024² (A + 4 subjects, dev2, 2 reps, doubled work, injected identity fault) and the 4096² memory pilot (full 512-column bank at 4095², head tables, the 1024² bank for A, one head and one linear rollout); peak ≤ 120 GB decimal (else 4096² is reported "not reached (memory)") |
| `train` | A100-80G | 12 heads + the under-trained control + driver identity (sequential; ≈ 7 min per arm with evaluation) |
| `rule1024` | A100-80G | §3.2 ladders at 1024² (sub16, BE) |
| `acc1024` | A100-80G | every configuration, 38 cases, BE and CN-R, Q1/Q2/Q3 |
| `time1024`, `time256`, `time4096` | H200 | §6 |
| `acc256` | A100-80G | heads and linear ladder at 256², BE, 38 cases (for Q-LH-cost at 256² and the head's cross-mesh error, descriptive) |

**Budget.** Cap 35 GPU-h including contingency; at most 2 lane jobs queued/running; account ≤ 8 (`squeue -u tawal01 -h
| wc -l` before each submit); ledger `runs/GPU-LEDGER.json` (P1/qualify `ledger.py`: submit only if spent + Σ time
limits of live jobs + this limit ≤ 35); time limit = 1.5 × a smoke-derived projection (`resources.py`). A-priori
estimate (re-priced by the certificate): smokes 3, train 2, rule1024 1.5, acc1024 3, acc256 2, time × 3 4.5 ≈ 16 GPU-h.
At most one cluster smoke at a time, ≤ 2 GPU-h each. Production waits for the coordinator's release.

**Defer order** (brief; first dropped first): 4096² cost panel → $K=32$ → matched-cost comparison at 256² (`acc256`
and the 256² linear-ladder subjects). **Never dropped:** $K\in\{4,8,16\}$ trained, error vs $K$ and cost vs $K$ at
1024², linear vs head at $r=K$. Deferred items are reported, never silently omitted.

## 9. E3c (head-repair hypotheses): listed as open

The governing plan's E3c hypotheses (initial-fit basin shared across meshes; LM branch restarts; mesh-dependent linear
terms) concern the head's **cross-mesh** spread on test64 cases 36/48/60, which this lane may not select on. They are
not cheap here (they need multi-mesh head runs with shared $z_0$ and an all-continuum arm) and are listed as open. The
lane reports, descriptively, the head's 256² vs 1024² per-case error spread on dev6 + val32 from `acc256` (if run) and
the initial-fit stationarity per case.

## 10. Deliverables

`analyze.py` → `runs/EXPORT.json` (every label, verdict and key number; a `contract` field on every cost number);
`make_report.py` → `report.md` (generated; level-1 title, status line, LaTeX, mermaid, inline provisional labels,
plain-language glossary) + PNG plots: error vs $K$ (per seed and pooled, with linear $r=K$ and the linear ladder), cost vs
$K$ per mesh, the error–cost plane at 1024² (heads and linear ladder, qualified vs not), seed spread, training curves
(oracle vs $K$). `runs/GPU-LEDGER.json`. Codex audits at every gated step in `audits/`.

## 11. Code (copy, then edit in the lane; `vendor/` untouched)

Copied from `exp/2026-10-18-jcp-qualify` @ `4399491eb` (sha256 in `PROVENANCE-copies.json`): `p1core.py`, `p1common.py`,
`p1stats.py`, `p1score.py`, `qscore.py`, `npcheck.py`, `timing.py`, `ledger.py`, `qtime.py` (as the base of `htime.py`),
`qacc.py` (as the base of `hacc.py`), `resources.py`, `cluster/{stage,submit,collect}` (namespace, prefix `jh_`, caps
edited). Copied from `experiments/separable-decoder` (this branch): `sep_hfit.py`, `sep_hfit_run.py` → `htrain.py`
(arm table extended by the $K$ override and SEED0 loop; nothing else changed). New: `hcore.py` (head LMM stepper =
P1 `make_evolve` with $c=\tilde h(z)$, head initial fit, head certifier, head Jacobian), `hcommon.py` (head settings,
tables per $(M,\text{rule})$), `hacc.py`, `htime.py`, `hrule.py`, `hscore.py`, `hstats.py` (§5.4), `test_h.py` (CPU
controls), `analyze.py`, `make_report.py`, `make_configs.py`.

## 12. Integrity rules carried

Never quote a pre-September Burgers speed-up; never mention the heat POD-128 comparison; retractions stated plainly;
speed-ups are always against named comparators with scope (K2 fired in 2D and 3D). Every GPU number from the cluster;
local GB10 only for CPU work (`JAX_PLATFORMS=cpu`) and at most one `jaxrun`. Codex audits: read-only prompts, no large
copies (≤ 2 GB scratch under `/tmp/jcp-heads-audit-scratch/`, deleted after each round), `df -h /home` ≥ 60 GB before
each audit. Waits on PIDs with an overall deadline.
