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

---

## Amendment A1 (2026-10-18, before any lane job; answers `audits/codex-design-1.md`, NEEDS-RESTATEMENT)

Pre-audit evidence (CPU, local, no lane job; not a result): the lane's head stepper (`hcore.py`) on the incumbent, BE,
256², Gauss 32², dev6|0 and dev6|2 reproduces quadrature-study's archived internal $z$ trajectories
(`dv256/archive/output/audit_head_gauss32_dev6{0,2}.npz`, A100) to $1.7\times10^{-12}$ and $1.1\times10^{-13}$ relative;
certified γ agrees with the recorded γ within 0.17 of the G-cert bar on every step.

**A1.1 Estimand (item 2).** What varies between heads is the **recipe at latent size $K$** under the deployed
deployment policy, not $K$ in isolation: with $K$ change the first-layer width, the skip-path output scale (skip weights
$0.3\,\mathcal N$, codes $0.1\,\mathcal N$), the parameter count, the learned code scale, the trust radius, the candidate
codes and the difficulty of the nonlinear initial fit. All are recorded per head (parameter count, code std and max norm,
trust radius, IC iterations / exit reasons / initial-fit residual per case). Claims are worded "the head recipe at $K$
under the $M=4K$ policy". The whitened ($q$-space) initialisation and optimisation of `sep_hfit.fit` are run unchanged
(the copied code), not an equivalent physical-coefficient fit. Equal optimiser steps are not equal convergence or equal
GPU-hours; wall time per head is reported.

**A1.2 Data provenance (item 3).** Source of the coefficient file:
`worktrees/2026-08-25-sepdec-consolidated/experiments/separable-decoder/runs/dn256b/out/sep_coeff_N256_K16_R512.npz`
(sha256 `4a7715659fe24b640c68e3c5f8b44e804383d4705aa62efa7dd6a3aaef73ebbb`), produced by dn256b stage B (job 2837431);
its extraction record `sep_coeff_N256_K16_R512.json` is the copy in this branch. "All $t\le5$ states" means **time
indices 0–5** (physical $t\le0.025$). The driver asserts the file and checkpoint sha256, and that the incumbent's emitted
`hfit_pick` equals the file's `pick` (so its codes `Z_tr` are aligned with the targets). The result is conditional on
this one frozen dataset; it is not a replication over data-generation seeds.

**A1.3 Q1 anchors and the initial fit (item 4).**
- A tight anchor is **verified** iff the run is valid, has no budget exit (reason 0), its certified γ is ≤ $10^{-6}$
  ($10\times$ its gtol $10^{-7}$) at every step, and (head) its initial fit exits stationary (reason 4) at its tightened
  gtol. Tighter: certified γ ≤ $10^{-7}$ at every step and initial-fit reason 4. An unverified anchor makes its case
  "anchor unverified", which counts as **not qualified** for Q1 predicate 3/4; a configuration with any such case is not
  numerically qualified (reported with the count). Tightened initial-fit gtol is $10^{-9}$ (tight) and $10^{-10}$
  (tighter), replacing $10^{-10}/10^{-11}$ of §3.1 (headroom for the unpivoted GJ solve); the smoke reports whether the
  anchors reach them, and a change after the smoke is a dated amendment before any production job.
- **Initial-fit veto (production):** a head case whose production initial fit does not exit stationary (reason 4) or by
  residual (reason 1) is an **invalid run** for Q1 predicate 1.
- **Degeneracy:** at output indices 1, 3, 5 of every dev6 trajectory the singular values of $J_z$ are recorded (Q3);
  a head configuration with $\sigma_{\min}/\sigma_{\max}<10^{-10}$ at any recorded state is flagged "rank-deficient step
  Jacobian" (reported beside its Q1 label). Control: a duplicated-latent fixture (the incumbent with latent 15's
  first-layer row and skip row copied onto latent 14) must be flagged (CPU).
- **"Qualified" means numerically qualified only** (local stationarity, solver and quadrature convergence); it says
  nothing about accuracy or global optimality (a head can be qualified in a bad basin). Accuracy is E0's job.

**A1.4 Q2 ladder and rule selection (item 5), replacing §3.2's selection paragraph.**
- Ladder $\mathcal L$ = Gauss $p^2$, $p\in\{16,24,32,48,64,96,128,160,192,256,320\}$. Selectable main rules: $p\le192$
  (each has a successor and a check rule in $\mathcal L$).
- **Common policy for every arm** (heads and every linear rung $r\in\{4,\dots,384\}$; qualify's frozen R128–R384 rules
  are no longer used, so both arms are selected by one policy): the main rule is the smallest selectable $q$ such that,
  on sub16 at 1024² with LSPG-BE $\Delta t_0$: every $q$, $q^+$ solve is valid; the rollout bar and ρ bar of $q\to q^+$
  pass on all 16 cases; and on dev6 the $q^+\to q^{++}$ rollout bar passes with valid $q^{++}$ solves ("pass", not
  "unverified"). Heads: all three seeds of a $K$ must meet it. If no selectable rung meets it, the arm is "not
  quadrature-qualified (ladder exhausted)" and is reported, not ranked.
- This is a registered **deployment-policy** comparison (each arm at its cheapest rule passing the same numerical test);
  it is not a best-achievable comparison of either arm. Seeds sharing a rule do not share a cost (iterations and initial
  fits differ); costs are per seed.

**A1.5 Fixed-$M$ sensitivity (item 6).** Registered: at 1024², LSPG-BE $\Delta t_0$, all 12 heads and linear
$r\in\{4,8,16,32\}$ also run with **$M=128$** (one explicit list: the first 128 frozen modes), each with its own rule
from the A1.4 policy (run in `rule1024`) and full Q1/Q2, 38 cases; accuracy only (no timing). It is a separate panel
("fixed-$M$ sensitivity"); the headline K claims stay "under the $M=4K$ policy". The §1 sentence predicting a non-empty
symmetric difference is withdrawn: the lane computes and reports the frozen-vs-nominal symmetric difference for every
$(M,\text{mesh})$, predicting nothing.

**A1.6 Statistics (item 7), replacing §5.1, §5.4 (error/cost/dominance labels), §5.5 and §5.6 where they differ.**
- **Seed population.** For a head size $K$: $\mathcal S_K$ = its numerically qualified seeds (at the mesh and scheme);
  pooled analyses need $|\mathcal S_K|\ge2$, else "$K$ unqualified". Per case, error $e_{K,i}$ = median over
  $\mathcal S_K$; cost $c_{K,i}$ = median over $\mathcal S_K$ of the seed's paired fused ratio on case $i$ (dev6). Error
  and cost therefore use the same seeds. Per-seed analyses are reported as sensitivity.
- **Error labels (paired, §5.4 population).** Superiority: one-sided exact Wilcoxon of $d_i$ (p1stats.wilcoxon), with
  resolved coverage ≥ 50 % and median $d<0$ → "$X$ more accurate"; symmetric. Equivalence: the inherited normalised TOST
  (p1stats.tost on $d_i/\delta_i$ against ±1, α = 0.05 each), $\delta_i=\max(2\varepsilon_{{\rm ref},i},0.1\min(e_{X,i},
  e_{Y,i}),10^{-12})$. **Precedence:** superiority first; else equivalence → "matches"; else "unresolved". The
  equivalence flag is reported beside a superiority label ("more accurate, within the 10 % margin"). Empty population
  (no case eligible and qualified for both) → "no population".
- **Cost labels** unchanged (bands $[1/1.1,1.1]$ on the case-bootstrap interval of the median per-case ratio $c_X/c_Y$
  over dev6); exactly 1.1 counts as inside the equal-cost band.
- **Dominance labels:** "$X$ dominates" iff error ∈ {more accurate, matches} and cost ∈ {cheaper, equal cost} and not
  (matches and equal cost); "**equivalent**" iff matches and equal cost; "neither (trade-off)" iff one is more accurate
  and the other cheaper; "unresolved" otherwise; "unqualified" if either side is not numerically qualified (numbers
  shown, labelled). Dominance uses the paired error test only (it is "paired-error dominance", not median-and-worst
  dominance); worst errors are displayed beside it.
- **Q-LH-cost (replaces §5.5's equal-cost rung).** Affordable rungs $\mathcal A_K$ = numerically qualified linear rungs
  whose cost label against the head is "cheaper" or "equal cost" (upper interval end ≤ 1.1). Primary comparator $r^\*$ =
  the rung of $\mathcal A_K$ with the lowest median error over the shared cases ("the most accurate linear rung at no
  more than the head's cost"; a validation-selected comparator, which can only favour the linear arm). Verdict = §5.4
  error label of head vs $r^\*$, read as: "linear at most equal cost is more accurate / matches / head more accurate than
  every affordable rung / unresolved"; "no affordable rung" if $\mathcal A_K=\emptyset$. Secondary: head vs $r_{\rm hi}$,
  the cheapest qualified rung whose cost label is "costlier" or "equal cost", with the full dominance label; all rungs'
  labels are tabulated (no monotonicity of linear error in $r$ is assumed).
- **H-acc-K** (replaces §5.2's labels) uses p1stats.h_acc_r_transition / h_acc_r_range **exactly**: per transition the
  outcome is "improves" (resolved ≥ 50 % of the population, > 50 % of resolved cases improve, median resolved change
  < 0), "does not improve", or "inconclusive" (reasons: empty population; > 25 % of cases lack qualification; resolved
  < 50 %); range "monotone" iff every transition improves, "not monotone" iff any does not improve, else inconclusive.
  Inputs: per-case seed-median errors of the two sizes on cases eligible in E0; a case is **qualified at size $K$** iff
  at least two seeds of $\mathcal S_K$ pass the per-case Q1 predicates on it (valid run, verified anchor, distance
  ≤ bar) — $\mathcal S_K$ already requires configuration-level Q1 and Q2.
- **F5 reading (replaces §5.6).** Positive: "**validation evidence of an accuracy–cost benefit of the head at $K$**
  (one mesh, $M=4K$ policy; repair and sealed confirmation open)" iff under BE: $|\mathcal S_K|\ge2$, Q-LH-cost =
  "head more accurate than every affordable rung" with the superiority p-value significant after **Holm correction
  across the tested $K$** (family-wise α = 0.05), and Q-LH-dim is not "linear dominates". Negative: "**no benefit shown on
  validation evidence**" only if every tested $K$ is qualified and resolved (Q-LH-cost = "linear … more accurate" or
  "matches"). Otherwise "**inconclusive**" (unqualified, unresolved or deferred $K$), which is not evidence of absence.
  Per-$K$ unadjusted labels are reported as exploratory. K5 (sealed) is untouched.

**A1.7 Controls (item 8).**
- Driver identity: the `identity` arm evaluates the incumbent's **own emitted codes** `Z_tr` (aligned via `hfit_pick`),
  not the file's `Z_ck` (which `base` would use). Bars: recon (train) mean equal to the record to $10^{-9}$ relative
  (deterministic arithmetic); oracle mean within $10^{-3}$ relative (it trains a 12k-step encoder).
- The under-trained head is a **sensitivity probe**, not a must-fail control: run in `train` beside its full-budget
  comparator ($K=16$, SEED0 = 0), ratio reported.
- Dominance synthetics corrected: $X$ 10 % more accurate on every case and **25 %** costlier → "neither"; $X$ 10 % more
  accurate and 25 % cheaper → "$X$ dominates"; $X=Y$ → "equivalent"; $X$ 10 % more accurate, cost ratio 1.0 →
  "$X$ dominates"; resolved 5 % better on every case with margin 10 % → "more accurate" with equivalence flag set; an
  unqualified side → "unqualified"; empty population → "no population"; Q-LH-cost with every rung costlier → "no
  affordable rung"; H-acc-K with a size of fewer than two qualified seeds → "$K$ unqualified" (no transition computed) and with > 25 % unqualified cases → "inconclusive"; a ladder with a missing successor → selection
  refuses. All CPU (`test_h.py`).
- H-cost-N must-fail: CPU synthetic (P1 `h_cost_n`): a 1.5× slower mesh in an otherwise flat three-mesh panel → "fail";
  equal → "pass". Q3 rank-deficient fixture: inherited duplicated column (CPU) + the head fixture of A1.3.
- Q2 under-resolved control: Gauss 8² → 16² on the incumbent, with the 16² and 24² solves required valid (else the
  control is "not demonstrable" and Q2 is blocked).

**A1.8 Coverage (item 1).** CN-R: `time1024` times every head and every linear rung under CN-R too, so Q-LH-dim and
Q-LH-cost are computed for both schemes at 1024² (BE primary). Population labels: accuracy-job labels (38 cases Q1, sub16
Q2) at 1024² and 256²; timing-job labels (dev6 Q1, dev6 Q2 replays) at every timed mesh; a configuration's 4096²
statements carry "dev6 timing-job label" only.

**A1.9 Budget reservation and 4096² pilot (item 9).** Ledger rule (qualify A3.1 pattern): a submission needs spent +
Σ time limits of live jobs + this job's limit + Σ projected limits of the **pending non-deferrable** jobs ≤ 35. Non-
deferrable: `train`, `rule1024`, `acc1024`, `time1024`. Deferrable in order: `time4096` → (K = 32 subjects, removed from
every job not yet submitted) → `acc256` + 256² linear-ladder subjects of `time256`. If the core cannot complete, the
outcome is "core incomplete; dependent verdicts unavailable". The 4096² pilot (`smk2`) builds the full 512-column bank,
the 1024² bank for A, **all** main-rule tables of the planned 4096² subjects and the largest concurrently live
successor/check tables, and runs one head and one linear rollout; it reports peak device bytes against 120 GB decimal.

**A1.10 Wording (item 10).** LSPG-CN-R is "the campaign's other LSPG scheme (observed order configuration-dependent,
qualify B4: 1.74–1.96)", never "second order" by name. A positive F5 reading is validation evidence only; the governing
plan's scope decision needs confirmed accuracy **and mesh invariance** on the sealed cohort, which this lane does not
test; nothing here bears on K2 (the FOM comparison).

---

## Amendment A2 (2026-10-18, before any lane job; answers `audits/codex-design-2.md`, NEEDS-RESTATEMENT)

**A2.1 Q-LH-cost restated (blockers 1, 3; replaces A1.6's "Q-LH-cost" bullet).** Population: the E0-eligible (ST)
cases among the 38 — one fixed population for every comparison of a size $K$ (numerically qualified configurations are
qualified on every case, so qualification does not shrink it). Costs at 1024² are measured on the same 38 cases (A2.4).
Rung classes by the cost label of linear rung $r$ against head $K$ (A1.6 bands): **affordable** $\mathcal A_K$ =
"cheaper" or "equal cost"; **equal-cost** $\mathcal E_K$ = "equal cost"; **costlier** = "costlier". Only numerically
qualified rungs enter; unqualified rungs are listed with their labels. Three separately named results:
1. **Equal-cost comparison (the brief's "r giving equal online cost").** For every $r\in\mathcal E_K$ the full §5.4
   dominance label; outcome "**no equal-cost rung on the ladder**" if $\mathcal E_K=\emptyset$.
2. **At-most-cost test (confirmatory, used by F5).** "Head more accurate than every affordable rung" iff
   $\mathcal A_K\ne\emptyset$ and, for **every** $r\in\mathcal A_K$, the superiority predicates hold (resolved coverage
   ≥ 50 %, median $d<0$) and the intersection–union p-value $p_K=\max_{r\in\mathcal A_K}p_{K,r}$ (one-sided signed-rank,
   p1stats.wilcoxon) is < α. "**Linear at no more cost is at least as accurate**" iff some $r\in\mathcal A_K$ has error
   label "linear more accurate" or "matches" (the rung is named). Otherwise "unresolved"; "**no affordable rung**" if
   $\mathcal A_K=\emptyset$. The lowest-median affordable rung is reported descriptively only (no inference attached).
3. **Bracket (descriptive).** $r_{\rm lo}$ = the costliest affordable rung, $r_{\rm hi}$ = the cheapest costlier rung,
   each with the full dominance label; "no rung below" / "no rung above" when absent. Cost-order ties are broken by the
   smaller $r$; no monotonicity of linear error in $r$ is assumed.

**A2.2 F5 made executable (blocker 2; replaces A1.6's F5 bullet).** Family $\mathcal F$ = $\{4,8,16,32\}$ minus sizes
deferred by the budget rule (A1.9; deferral is decided from the ledger before any accuracy result is read), frozen in
`runs/F5-FAMILY.json` when `acc1024` is submitted. Per $K\in\mathcal F$: $p_K$ of A2.1(2) under BE; $p_K:=1$ (a
non-rejection) when $K$ is unqualified ($|\mathcal S_K|<2$), $\mathcal A_K=\emptyset$, the population is empty, or any
component p-value is not computable. Holm across $\mathcal F$ at family-wise α = 0.05. **Positive at $K$** iff Holm
rejects at $K$, every component predicate of A2.1(2) holds, and Q-LH-dim at $K$ is a **resolved label other than
"linear dominates"** (i.e. "head dominates", "equivalent" or "neither"; "unresolved", "unqualified" or "no population"
block a positive). Lane outcome, in precedence order:
1. "**validation evidence of an accuracy–cost benefit of the head at $K\in\{\dots\}$**" if some $K$ is positive (other
   sizes listed with their labels; one mesh, $M=4K$ policy, error on eligible all38 and cost on the same cases at 1024²;
   repair and sealed confirmation open);
2. "**no benefit shown on validation evidence for $K\in\mathcal F$**" if every $K\in\mathcal F$ is qualified and has
   the resolved outcome "linear at no more cost is at least as accurate" (deferred sizes named as deferred);
3. "**inconclusive**" otherwise (not evidence of absence).

**A2.3 Seed summaries (item 4).** Seed-median errors and costs are a **summary over the numerically qualified
retrains** of a size $K$, not the error and cost of one deployable model (the median seed may differ between case,
error and cost). Qualification counts $|\mathcal S_K|$ are reported with every pooled number, and every pooled
comparison is repeated per seed (joint error–cost of each individual head) as a reported sensitivity.

**A2.4 Populations (blocker 4).** `time1024` times **all 38 cases** (dev6 × 3 repeats, val32 × 1 repeat; per-case cost =
median over its repeats) so 1024² cost labels and errors share one population; the per-case cost ratio of $X$ to $Y$ is
the ratio of their A-normalised fused costs on that case, and the cost label's interval is the case bootstrap (10,000) of
the median over the population. Timing-job certification (§6) stays on dev6; val32 timing runs reproduce the identity
hash of an untimed prod run in the same job and are covered by `acc1024`'s 38-case qualification. `time256` and
`time4096` stay dev6 (H-cost-N, as P1); any 256² comparison is scoped "error on eligible all38 (acc256), cost on dev6".

**A2.5 Rule policy for heads (item 7, minor).** A head size's rule is the smallest selectable rung at which **at least
two** of its three seeds pass every A1.4 predicate (one bad seed no longer exhausts the ladder); seeds that fail at the
chosen rung are labelled "not quadrature-qualified at the size's rule" and leave $\mathcal S_K$. Wording: "smallest
passing rule", never "cheapest" or "cost-optimal"; selection under BE, transferred to CN-R with fresh Q2.

**A2.6 Controls and classification details (items 8, 9).**
- The "> 25 % unqualified cases" branch of H-acc-K cannot arise from qualified sizes (configuration-level Q1 covers every
  case); it is a **unit test** of `h_acc_r_transition` only, not an integrated outcome.
- Propagation: an empty population gives error label "no population", and the dominance label is then "no population"
  (precedence: unqualified > no population > the A1.6 labels).
- Degeneracy: an all-zero or non-finite $J_z$ spectrum is flagged "degenerate" (in addition to the
  $\sigma_{\min}/\sigma_{\max}<10^{-10}$ flag).
- Wilcoxon wording: exact null when $n\le50$ and no ties among $|d_i|$, otherwise the inherited normal approximation.
- CPU synthetics are fixed: $n=38$, $\varepsilon_{{\rm ref},i}=10^{-6}$, base errors $e_{Y,i}=0.01(1+i/38)$, all runs
  valid and timed with A-normalised costs drawn as exact constants; the A1.7 expected labels follow deterministically.
  Added: Codex's counterexample (head $H_i=i$; rung A $=H+1$ except cases 20–25 $=H-10$; rung B $=H$; both affordable)
  must give "linear at no more cost is at least as accurate (rung B: matches)" and **not** "head more accurate than
  every affordable rung".

**A2.7 Budget inventory and repricing (blocker 5; amends A1.9).** Non-deferrable (reserved while pending): `train`,
`rule1024`, `acc1024`, `time1024`, and `time256` restricted to the head subjects and the matched-dimension linear rungs
$r\in\{4,8,16,32\}$. Deferrable, dropped in order: `time4096` → $K=32$ (its subjects removed from every job not yet
submitted) → `acc256` and the 256² linear-ladder subjects $r\ge64$. The pending total excludes the attempt being
submitted; a failed or timed-out mandatory job returns to the pending list. A-priori repricing (the smoke certificate
replaces it): smokes 3, train 2.5, rule1024 2.5, acc1024 5, time1024 3, time256 1.5, time4096 2.5, acc256 2.5 ≈ 22.5
GPU-h; time limits 1.5 × projection; cap 35.

---

## Amendment A3 (2026-10-18, before any lane job; answers `audits/codex-design-3.md`, NEEDS-RESTATEMENT)

**A3.1 Simultaneous inference over a predeclared component family (blocker 1; amends A2.1(2) and A2.2).** The
confirmatory family is fixed now, before any data: all component hypotheses
$H_{0,K,r}$: "head $K$ is not more accurate than linear rung $r$" for $K\in\mathcal F$ and every ladder rung
$r\in\{4,8,16,32,64,128,256,384\}$, LSPG-BE, 1024², on the A2.1 population. Component p-value $p_{K,r}$ = one-sided
signed-rank (p1stats.wilcoxon) of $d_i=e_{H,i}-e_{L,i}$; $p_{K,r}:=1$ whenever it is unavailable (either side
unqualified or without complete timing, empty population, not computable). **Holm across the whole family**
($|\mathcal F|\times8$ hypotheses, family-wise α = 0.05). Because Holm controls the family-wise error strongly over
every component, any conclusion that is a conjunction of rejected components stays controlled however the conjunction's
members were chosen (affordability, qualification). Re-stated at-most-cost result: "**head more accurate than every
possibly-affordable rung**" iff the set $\mathcal P_K$ of possibly-affordable rungs (A3.2) is non-empty, every
$r\in\mathcal P_K$ is Holm-rejected and meets the superiority predicates (resolved coverage ≥ 50 %, median $d<0$).
The interpretation is limited to the **numerically qualified ladder rungs** under this lane's rule policy, not to every
conceivable linear deployment. A2.2's per-$K$ Holm is replaced by this family-wise Holm; the positive F5 reading at $K$
requires the result above plus A2.2's Q-LH-dim gate; precedence and outcomes of A2.2 are otherwise unchanged.

**A3.2 Cost classes, timing completeness and propagation (blocker 2; amends A2.1, A2.4).**
- Timed measurements: only **valid** fused triplets (identity hash reproduced, drift in band) count. The fused triplet
  gets the same in-job retry rule as the split triplet (≤ 2 retries on drift). No rerun job is registered.
- A subject's 1024² timing is **complete** iff every case of the A2.1 population has ≥ 1 valid fused repeat.
- Cost classes of a linear rung against head $K$: "cheaper", "equal cost", "costlier", "unresolved" (interval straddles
  a band edge), "unknown" (either side's timing incomplete). **Certified affordable** = cheaper or equal cost;
  **possibly affordable** $\mathcal P_K$ = every numerically qualified rung that is **not** certified costlier
  (certified affordable ∪ unresolved ∪ unknown). Wording "no rung certified affordable" replaces "no affordable rung".
- Seeds: $\mathcal S_K'$ = numerically qualified seeds of $K$ with complete 1024² timing; pooled error **and** cost use
  $\mathcal S_K'$; $|\mathcal S_K'|<2$ → $K$'s cost comparisons "unavailable (timing incomplete)", every
  $p_{K,r}:=1$, F5 at $K$ non-rejecting. The fixed population is never shrunk; no rung is dropped for missing cost (it
  stays possibly affordable).
- The "linear at no more cost is at least as accurate" result requires a **certified affordable** rung with error label
  "linear more accurate" or "matches".

**A3.3 Family lifecycle and bracket ordering (minor).** Once `runs/F5-FAMILY.json` is frozen at `acc1024` submission, a
size that later cannot be completed stays in $\mathcal F$ with $p_{K,r}=1$ (it can never be removed to ease
multiplicity or the negative reading). The bracket's scalar ordering statistic is the cost label's own statistic, the
median over the population of the per-case ratio $c_{L_r}/c_{H_K}$ (ties → smaller $r$); $r_{\rm lo}$ = the
largest-ratio certified-affordable rung, $r_{\rm hi}$ = the smallest-ratio certified-costlier rung.

---

## Amendment A4 (2026-10-18, before any lane job; answers `audits/codex-design-4.md`, NEEDS-RESTATEMENT)

**A4.1 The confirmatory/FWER claim is withdrawn (blocker 1).** The rule (A1.4/A2.5), the seed pool (numerical
qualification, A2.3) and the timing-complete pool (A3.2) are selected on the same validation trajectories that the
component tests use, and this lane has no independent evaluation cohort (sealed cohorts are out of scope). The lane
therefore makes **no family-wise error-rate claim**. A3.1's family-wise Holm adjustment is kept as a fixed,
pre-registered **decision rule and a descriptive multiplicity adjustment**, and every label it feeds — the at-most-cost
result, the F5 reading and Q-LH-dim — is reported as **exploratory validation evidence, conditional on the listed
selections** (rule policy, qualified and timing-complete seed pool, numerical qualification of each rung). The F5 positive
reading is re-worded "**exploratory validation evidence of an accuracy–cost benefit of the head at $K$**"; nothing in the
lane is called confirmatory. Assumptions of the signed-rank tests, stated: cases are independent units; under the null the
per-case paired difference is symmetric about 0; the exact null is used for $n\le50$ without ties among $|d_i|$, the
normal approximation otherwise (p1stats.wilcoxon). The selection steps and the excluded seeds (with identities and
reasons: Q2 at the size's rule, Q1, timing completeness) are reported beside every pooled number.

**A4.2 Scope of the seed pools (minor 2).**
- $\mathcal S_K'$ (qualified **and** timing-complete at 1024²) is used only for the 1024² joint error–cost results:
  Q-LH-dim, Q-LH-cost (equal-cost, at-most-cost, bracket) and the F5 reading, for both schemes.
- Q-K-err and the 1024² error tables use $\mathcal S_K$ (numerically qualified at 1024², no timing requirement).
- The fixed-$M=128$ panel uses its own $\mathcal S_K^{(128)}$ (numerically qualified in that panel; no timing).
- 256² (optional): $\mathcal S_K^{(256)}$ = numerically qualified in `acc256`; costs on dev6 from `time256`, labelled
  "error on eligible all38, cost on dev6", never a joint all38 accuracy–cost result.
- H-cost-N uses seed 0's timing-job subjects as registered in §5.3 (dev6, every timed mesh).

**A4.3 Cost-coverage status (minor 3).** "No rung certified affordable" is a **cost-coverage status** displayed beside
the at-most-cost accuracy result, not a verdict that overrides it; the at-most-cost result is computed on the
possibly-affordable set $\mathcal P_K$ regardless (A3.2), and is "no possibly-affordable rung" only when every qualified
rung is certified costlier.

---

## Amendment A5 (2026-10-18; design audit 5: READY for exploratory validation evidence; two minor clarifications)

- "No possibly-affordable rung" carries its reason: "every qualified rung certified costlier" or "no numerically
  qualified linear rung" (A4.3; non-positive either way).
- Q-LH-dim's paired tests are **unadjusted and exploratory**; Q-LH-dim is a gate of the F5 reading, not an input of the
  family-wise Holm adjustment (A4.1 wording corrected).
- Implementation note (no change of design): each trained head is archived as a compact `head_<name>.npz` (the head's
  MLP and skip weights, the initial-fit candidate codes $Z[::16]$, the trust radius and code statistics computed from all
  codes, and the training cfg), which with the incumbent checkpoint's bank parameters defines the decoder exactly; the
  full emitted pkl stays on the cluster only while the lane runs and its sha256 is recorded.

---

## Amendment A6 (2026-10-18, before any lane job; implementation notes from `audits/codex-code-1.md`)

- **Linear reproduction fixture corrected.** §7 named qualify's `acc1024` archive, which holds no $\Delta t_0$ LSPG runs.
  The fixture is jcp-combined's `c2grid` archive (A100 80GB; the source qualify itself used for this gate): R384
  `gauss128`, runs `LSPG|BE|0.005|prod` and `LSPG|CNR|0.005|prod`, dev6 (`inputs/repro/lin_R384_gauss128_{BE,CNR}_dev6.npz`,
  sha256 in `inputs/repro/PROVENANCE.json`); both schemes ≤ $10^{-10}$.
- **Head Jacobian control on every head arm** (analytic vs autodiff at 4 fixtures), not only the incumbent.
- **Smoke coverage.** `smk1` = training pipeline (identity + $K=4$, $K=32$ short heads), accuracy controls on the
  incumbent (all 38 cases, both schemes: the real-size cost probe of one head configuration), short-head arms, linear arms,
  and the rule-ladder probe on the incumbent, a newly trained head group (min_pass 2) and a linear rung. `smk2` = timing
  controls at 1024² and the 4096² pilot. The pilot uses **stand-in heads with the actual latent sizes** (random 512×2
  MLPs of latent size $K$, separate archives and initial-fit tables per subject) because the trained heads do not exist
  yet; their arithmetic and storage per iteration equal a trained head's, their solver iteration counts do not, which is
  why the pilot certifies memory, not time.
- **Ledger.** Every reservation (smokes included) requires a pending inventory; a retry replaces its job's pending entry;
  one lane smoke at a time and ≤ 2 GPU-h per smoke are enforced by `submit.sh` / `stage.py`; the account cap is re-checked
  on the login node in the same command that runs `sbatch`.
- **Archives.** Training jobs emit compact head archives (A5) and full pickles; only the compact archives are collected
  (the pickles' sha256 are kept in `OUTPUTS.sha256` and the training result), and the job checks the compact archive
  defines exactly the emitted head (difference 0). Accuracy and timing records keep tight and tighter internal
  trajectories, budget-exit counts and initial-fit exit records for the offline anchor audit.

---

## Amendment A7 (2026-10-18, before any lane job; decisions after `audits/codex-code-{2,3}.md`)

- **Account cap (campaign-level residual risk, accepted and reported).** The lane checks the account job count and runs
  `sbatch` inside one cluster-side lock (`flock` on `/cluster/tufts/paralab/tawal01/.campaign-sbatch.lock`), failing
  closed when the queue query fails. Lanes that do not take this lock can still race it within one command's window; a
  lane cannot enforce that, so the residual risk is accepted here and reported to the coordinator (a campaign-wide
  submission lock would close it).
- **4096² pilot = measurement probe.** It builds every planned 4096² table with actual-dimension stand-in heads, keeps
  the timed inputs and several outputs resident, compiles the level variants and the split programs, and applies a 10 %
  allowance; it does not reproduce every allocation of the timing driver and is not a proven bound. `time4096` is first
  in the defer order; if it fails on memory it is reported "not reached (memory)", never silently dropped.
- **Fixed-$M=128$ panel:** $K=32$ heads and linear $r=32$ already have $M=4K=128$, so their deployed configurations are
  the panel's entries (no duplicate runs).
- Offline rule tables are Gauss-only (the A1.4 policy has no other family); a non-Gauss rule raises.

---

## Amendment A8 (2026-10-18; decision after `audits/codex-code-5.md`)

Code audits 4 and 5 found **no blocker in the smoke drivers** (htrain, hacc, hrule, htime, hpilot, harm, hcore, hcommon,
cluster tooling); their remaining findings concern `smoke_accept.py`, a local CPU post-processor that reads the smoke
archives after the jobs finish. Decision: `smk1` is submitted now; the acceptance checker is completed (round-5 items:
identity vs drift failures and retry attempts, smk1 coverage of configured identities and rule trails, pilot stage
coverage and per-stage peak validation, duplicate equivalence records, doubled-work drift recomputation) and re-audited
**before any smoke result is accepted or any production job is staged**. No smoke number is used for a verdict.
