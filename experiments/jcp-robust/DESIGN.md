# jcp-robust — design (pre-registration): randomised rules (E1), full-recipe seeds (E2), hardware (E5)

**Status:** DRAFT v1, 2026-10-19, written before any data of this lane exists. Plan items V2 E1, E2, E5 (priority 4) of
`reports/2026-10-17-full-experiment-plan/` (PLAN-v2 + v3 + v4 amendments; later files win), lane brief
`campaign/briefs/jcp-robust.md`, common brief `campaign/briefs/COMMON-BRIEF.md` (binding). Amendments are appended as
dated sections at the end; nothing above them is rewritten after the design audit.

Branch `exp/2026-10-19-jcp-robust` (fork `21c175a1b`), lane dir `experiments/jcp-robust`, cluster namespace
`/cluster/tufts/paralab/tawal01/jcp-robust/`, raw archives `/cluster/tufts/paralab/tawal01/burgers-archive/jcp-robust/`.
Cap **3 concurrent jobs, 75 GPU-h including contingency and time-limit reservations**.

All accuracy numbers are scored against E0 ST (`refstore.py`, `JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store`);
$\epsilon_{\rm ref}$ is a conditional Richardson estimate and is labelled so. S is used only for BE at $\Delta t=0.005$ in
2D (an unmatched diagnostic otherwise; never in 3D). Selection uses dev/validation cohorts only (2D dev6 + val32 = 38
cases; 3D 923801, 64 cases). No sealed, custody or reserved-OOD data is touched. Post-hoc labels carried unchanged: E0
A10; combined A10, A11 and the retracted A15; qualify's GAL-CN-is-diagnostic decision; speed3d winners not Q2-qualified.

## 0. What this lane asks, in one paragraph

The paper's numbers come from **one** realisation of three random things: one random shift of the lattice rules (seed 0),
one training run of each bank (one seed of a non-bitwise-reproducible recipe), and one GPU type. E1 asks whether the
deployed lattice draw is typical of its draw distribution and whether qualification survives redrawing. E2 asks whether
the priority-1/2 verdicts (mesh-independent accuracy, the r-dial, the qualified fronts, the selected rules) survive
retraining the whole recipe with new seeds. E5 asks whether the dimensionless timing conclusions (cost flat in mesh, the
2D crossover verdict) are the same on A100 and H200.

```mermaid
flowchart LR
  subgraph E1[E1 randomised rules]
    D0[draw s=0 = deployed]:::frozen --> DR[16 shift draws per qualified lattice config]:::run
    BAD[bad generating vector control]:::ctrl --> GATE[Gauss / gl64 independent-family gate]:::run
    DR --> GATE
  end
  subgraph E2[E2 full-recipe seeds]
    TR[retrain bank, seed s]:::train --> ROT[ordering / rotation]:::train --> SEL[rule selection]:::run --> VER[H-acc-N, H-acc-r, fronts, Q1/Q2]:::run
  end
  subgraph E5[E5 hardware]
    A100[A100 jobs x2]:::run --> CMP[dimensionless ratios compared]:::ana
    H200[H200 jobs x2]:::run --> CMP
  end
  classDef frozen fill:#e5e7eb,stroke:#374151; classDef run fill:#dbeafe,stroke:#1e40af;
  classDef train fill:#fef3c7,stroke:#92400e; classDef ctrl fill:#fee2e2,stroke:#991b1b; classDef ana fill:#dcfce7,stroke:#166534;
```

## 1. Inputs reused (exists / runs / qualified)

| Input | Source (read-only) | Status used here |
|---|---|---|
| 2D bank dn256b (R=512, K=16) + head-based rotation `rotation_R512.npz` (sha `51149166…`) | `separable-decoder/runs/dn256b`, `burgers-bank-knob/make_rotation.py` | incumbent ("seed 0"); qualified per FRONT-2D v1 |
| FRONT-2D v1 (qualify @2374fafe1) | `2026-10-18-jcp-qualify/.../runs/FRONT-2D.json` | which 2D configurations are numerically qualified |
| 2D rollout + in-job Q1/Q2 driver | qualify `qacc.py`, `p1core.py`, `p1common.py`, `qcert.py`, `qscore.py` | copied with provenance, bank/rule made parameters |
| 2D rule names with draw suffix (`fib<n>s<k>`, `badlat<n>`) | dims `rules2d.py` | copied with provenance; seed 0 = deployed points bit for bit (gate G-E1.1) |
| 3D W1024 bank (sha `86094f68…`) and M2_512 | wide-bank `inputs/model_W1024`, combined `c3common.py` | incumbent ("seed 0") |
| 3D rollout driver + Q2 | qualify `q3run.py`, `s3ctx.py`; dims `d3run.py`, `rules3d_names.py` (`lat<n>s<k>` re-shift with the recovered generating vector) | copied with provenance |
| 3D qualified subset (4 configurations) | qualify report (`q3d2`, `q3d3`) | E1 3D population, E2 3D verdict set |
| 2D training recipe | smooth-bank `smoothtrain.py --arm base --sigma 4 --lam 0` (audited copy of the r3a recipe: data, 300k steps); dn256b `sep_coeff_extract.py`, `sep_hfit.py` (arm mid); `make_rotation.py` | copied with provenance; only the seeds change |
| 3D training recipe | wide-bank `train3d/train2w.py`, `configs/train_r1024.json` | copied with provenance; only `model_seed`, `subset_seed` change |
| Timing harnesses | p1-scaling `p1time.py` (P1c), fom2d `c2time.py` (C2) | copied with provenance; subsets, plus clock/version recording |
| Cluster mechanics | p1-scaling `cluster/stage.py`, `submit.sh`, `collect.py`, `ledger.py` | copied; submit goes through the campaign flock (§14), lane cap 3, ledger cap 75 |

Every copied file gets an entry in `PROVENANCE.json` (source branch, commit, path, sha256). Vendored files stay unedited.

## 2. E1 — randomised rules

### 2.1 Population (actual, from the qualified sets; nothing else)

Only configurations that are **numerically qualified and use a randomised rule** enter. Gauss and Smolyak are
deterministic (not applicable). Sobol: **no qualified 2D or 3D configuration uses a Sobol rule** (FRONT-2D v1; qualify 3D
subset), so the Owen-scramble part has no front population; it is run as a labelled **diagnostic** only (§2.5). Note on
terminology: the code's Sobol scramble is scipy's `scramble=True` = linear matrix scramble + digital shift (LMS), not
nested-uniform Owen scrambling; it is reported as "LMS-scrambled Sobol", never as "Owen".

| id | dimension | configuration (bank, r, rule, form, scheme, $\Delta t/\Delta t_0$) | meshes | cases |
|---|---|---|---|---|
| E1-2D-a | 2D | dn256b, R128, fib4181, LSPG, BE, 1 | 256², 1024², 4096² | 38 / 38 / 16 (4096²: dev6 + val32[0:10], P1 subset) |
| E1-2D-b | 2D | R128, fib4181, LSPG, BE, 2 | same | same |
| E1-2D-c | 2D | R128, fib4181, LSPG, BE, 2.5 | same | same |
| E1-2D-d | 2D | R128, fib4181, LSPG, CN-R, 1 | same | same |
| E1-2D-g{1,2,5} | 2D (diagnostic) | R128, fib4181, GAL, CN, 1 / 2 / 5 | same | same |
| E1-3D-a | 3D | W1024 bank, r=256 (W256), lat8192, QN-4, BDF2, $\Delta t=0.025$ | 65³ | 64 (923801) |
| E1-3D-b | 3D | W1024 bank, r=512 (W512), lat16384, QN-4, BDF2, $\Delta t=0.025$ | 65³ | 64 |

2D test modes: the frozen 1024² list for $M=4r$ (P1/qualify `frozen` policy, hashed), reused at every mesh. 3D:
`complete_M(65, 4r)` at 65³ (single mesh; actual $M$ recorded).

**Draws.** $s=0,\dots,15$ (16 draws; $s=0$ is the deployed realisation, so 15 new). 2D: the Fibonacci generating vector
$z=(1,F_{k-1})$ is fixed and the Cranley–Patterson shift is $\Delta_s=\texttt{default\_rng}(s).\texttt{uniform}(2)$ (points
$\{(i z/n+\Delta_s)\bmod 1\}$, equal weights). 3D: the CBC vector recovered from the pinned points of the deployed rule
(bit-for-bit check, dims `rules3d_names.z_of`), shift $\texttt{default\_rng}(s).\texttt{uniform}(3)$. Defer item: 3D draws
$s=16..63$ (48 more) if budget allows.

### 2.2 Per-draw quantities

For each configuration $c$, draw $s$, case $i$:
- the in-job Q1 certificate at every step (fresh Jacobian): validity, pooled non-stationary fraction ($\gamma>10^{-3}$);
- $d_{s,i}$ = scoring-lattice relative rollout distance (the Q2 metric, max over the six output times) between the draw-$s$
  rollout and the **converged independent-family rollout** of the same configuration: **gauss384** in 2D, **gl64** in 3D
  (the dims cross-family targets);
- $d^{+}_{s,i}$ = the same distance to the **draw-matched successor** (2D fib6765 with shift $s$; 3D lat16384 / lat32768 with
  shift $s$), i.e. the registered Q2 check applied to draw $s$;
- $\rho_{s,i}$ = the tested-advection indicator of the draw against the converged rule at the draw's own reached states
  (Q2 $\rho$ bar 0.116);
- $e_{s,i}$ = E0 ST error (max over output times of $\lVert u-R\rVert/\lVert R(t_0)\rVert$), with eligibility per case-kind from
  the E0 store.

$D_s=\max_i d_{s,i}$; a draw **passes** iff every run is valid, the pooled non-stationary fraction is $\le1\%$, and
$D_s\le2.5\times10^{-4}$ (the Q2 rollout bar, every case) and $\max_i\rho_{s,i}\le0.116$. The draw-matched successor check
($\max_i d^+_{s,i}\le2.5\times10^{-4}$) is reported beside it as the "own-family Q2" label.

### 2.3 Decision rules (frozen; per configuration and mesh)

- **H-E1a (qualification survives redrawing).** *Robust* if 16/16 draws pass; *fragile* if $\ge2$ draws fail;
  *inconclusive* if exactly one fails. The Clopper–Pearson 95 % upper bound on the failure probability is printed beside
  it (16/16 gives 0.171, so "robust" means "failure rate not shown above 17 %", and is worded so).
- **H-E1b (the deployed draw is typical).** Rank $k_0\in\{1..16\}$ of $s=0$ among the 16 draws (1 = smallest) for three
  statistics: $D_s$, the median over cases of $e_{s,i}$, the worst over cases of $e_{s,i}$ (ties broken by draw index; tied
  values reported). *Typical* if $3\le k_0\le14$ for all three; *atypically favourable* if $k_0\le2$ for any statistic
  where lower is better and none is $\ge15$; *atypically unfavourable* if any $k_0\ge15$ and none $\le2$; *mixed*
  otherwise. Under exchangeability $P(k_0\in\{1,2,15,16\})=1/4$ per statistic, so an atypical label is descriptive, not
  evidence of selection; it is reported with this probability.
- **H-E1c (accuracy is insensitive to the draw).** Per eligible case, $\Delta_i=\max_s|e_{s,i}-e_{0,i}|$ against the margin
  $\max(2\epsilon_{{\rm ref},i},\,0.1\,e_{{\rm conv},i})$, where $e_{\rm conv}$ is the converged-rule error (the H-acc-N
  margin of V4-5). *Insensitive* if $\ge90\%$ of eligible cases are within margin; *sensitive* if $<75\%$; *inconclusive*
  between. The distribution over draws (median, IQR, min–max of the case-median and case-worst error) is reported beside
  it.
- **Indicator coverage for the deployed draw (plan E1 text).** For $s=0$: whether the $\rho$ indicator's verdict ($\le0.116$)
  agrees with the rollout verdict ($D_0\le2.5\times10^{-4}$); over draws: the 2×2 table of indicator vs rollout verdicts.
- **Precedence.** If the must-fail control (§2.4) does not fire, every E1 verdict of that dimension is "blocked"; if the
  reproduction gate G-E1.2 fails, the configuration's E1 verdicts are "blocked (reproduction)". GAL-CN rows are diagnostic
  and carry no verdict.

### 2.4 Controls (E1)

- **Must-fail C-E1.1 (bad generating vector caught by an independent family).** 2D `badlat4181` ($z=(1,1)$, all points on
  shifted diagonals, $n=4181$, the deployed size); 3D `badlat8192` ($z=(1,1,1)$, $n=8192$). *Establishment, CPU, before any
  GPU use* (independent NumPy decode of the bank at the rule points, not the job code; dims's `fixture-badlat` method):
  $\rho_{\rm tgt}=\lVert N_{\rm bad}(c)-N_{\rm tgt}(c)\rVert/\lVert N_{\rm tgt}(c)\rVert\ge1$ at all six saved output states of
  dev6|0 (2D, E1-2D-a at 1024², qualify archive) / case 0 (3D, E1-3D-a, qualify `q3d3` archive). Not admitted → control
  "unavailable", E1 verdicts of that dimension "blocked (control unavailable)" (fixture never re-chosen). Admitted → in job,
  the bad rule is run as a draw of the configuration on all its cases; it **must be flagged**: $D>2.5\times10^{-4}$ against
  the independent target on $\ge1$ case, or a run invalid (counted as flagged, labelled "flagged by invalidity").
- **Must-fail C-E1.2 (draw identity).** A configuration evaluated with draw $s=1$ points but labelled $s=0$ must be rejected
  by the table-identity check (rule sha256 in the setting metadata) — synthetic CPU test plus one in-job check.
- **Positive gate G-E1.1.** `fib4181s0` / `lat8192s0` / `lat16384s0` points and weights equal the deployed rules bit for bit
  (sha256), and for $s\ge1$ the generating vector is unchanged (only the shift differs).
- **Positive gate G-E1.2 (reproduction).** For $s=0$ the rollouts reproduce qualify's archived results (2D `acc*`, 3D `q3d*`)
  to $\le10^{-10}$ in the scoring metric (read one archive file at a time on the cluster or in scratch, §16).
- **Positive gate G-E1.3 (permuted-case batching, plan group-E control).** Draw 0 of E1-2D-a at 1024² and E1-3D-a re-run
  with the case order permuted: per-case outputs equal to $\le10^{-12}$. A mismatch invalidates the job.

### 2.5 Diagnostic (no verdict)

LMS-scrambled Sobol, 16 scramble seeds: 2D R128 `sobol8192s<k>` (LSPG-BE 1, 1024², 38 cases) and 3D W512 `sob16384s<k>`
(QN-4 BDF2 0.025, 64 cases), $d$ to the same targets. Reported as a spread-only diagnostic ("a non-qualified family").

## 3. E2 — the full recipe repeated with new seeds

### 3.1 What a "seed" changes

The recipe = (i) bank training, (ii) ordering/rotation, (iii) rule selection, then (iv) the verdict runs. The training
**data are fixed** (they are part of the problem): 2D the seed-0 draw of 576 trajectories (256², 50 steps) and the 4608
trajectories for code extraction; 3D training seed 923701 × 1536 and bank-validation 923751 × 96. A recipe seed $s$
changes every algorithmic random stream: 2D model initialisation and minibatch/point subsampling of Stage A
(`smoothtrain --seed`; the 16,384-state pick stays the seed-0 pick, as in smoothtrain, because it is part of the data),
the head-fit seed (`sep_hfit`); 3D `model_seed = 923715 + 1000 s` and `subset_seed = 923002 + 1000 s`.
Seeds: $s\in\{1,2,3\}$ (registered); $s\in\{4,5\}$ is the first defer item. The incumbent (dn256b; W1024 j3b) is "seed 0"
and is reported beside them; it was trained under older software, so it is a member of the recipe distribution but
not a same-software replicate (labelled). Training uses default XLA autotuning as the incumbents did (not bitwise
reproducible; smooth-bank R2b); seeds, code hashes and output sha256 are recorded.

### 3.2 2D (R512 bank; prefixes r = 128, 256, 384)

Per seed: Stage A `smoothtrain.py --arm base --sigma 4 --lam 0 --nodes 256 --seed s` (300k steps); Stage B code
extraction + `sep_hfit` arm mid (200k steps, seed $s$); rotation by the deployed head-based `make_rotation.py`
procedure (QR of $G$ on 256², rows $R_Gc_i/\lVert R_Gc_i\rVert$ with $c_i=h_\theta(z_i)$, SVD; $\lVert LT-I\rVert$ recorded).
Smooth-bank showed the fast-setting result depends on the rotation procedure, so it is fixed to the deployed one.

**Rule selection (registered procedure, applied identically to every bank including the incumbent).** At 1024², 38 cases,
for each $r\in\{128,256,384\}$: candidate list ordered by $m$: fib4181 (4181), fib6765, gauss96 (9216), fib10946,
gauss128 (16384), gauss160, gauss192 (36864). The selected rule is the smallest-$m$ candidate that, under **both**
LSPG-BE $\Delta t_0$ and LSPG-CN-R $\Delta t_0$, passes: Q1 (in-job certificate, distance to the adaptive tight solve
$\le2.5\times10^{-4}$ every case), Q2 against its family successor (rollout $\le2.5\times10^{-4}$ every case, $\rho\le0.116$),
and the cross-family check $d\le2.5\times10^{-4}$ against gauss384 (every case). **Anchor gate G-E2.1:** on the incumbent
the procedure must return the deployed rules (R128 fib4181, R256 gauss128, R384 gauss128). If it does not, that is
reported, and every bank (incumbent included) is evaluated with its procedure-selected rules so comparisons stay
like-for-like; the incumbent's deployed-rule results are then a separate row.

**Verdict runs (per seed, with its selected rules; frozen 1024² mode list; in-job Q1, Q2 vs the successor):**
- H-acc-N cells: $r\in\{128,256,384\}\times$ {LSPG-BE, LSPG-CN-R} $\Delta t_0$ × {256² (38), 1024² (38), 4096² (16)}; the
  registered V4-5 rule unchanged (per-case slope vs $\log N$ Wilcoxon; TOST margin $\max(2\epsilon_{\rm ref},0.1e(1024^2))$;
  P1's pass/fail/inconclusive mapping).
- H-acc-r: transitions 128→256, 256→384 per (scheme, mesh) by the V4-5 rule (resolved coverage $\ge50\%$, improves iff
  $>50\%$ of resolved cases improve and the median paired change $<0$; ties within $2\epsilon_{\rm ref}$).
- Front at 1024²: the LSPG arms of FRONT-2D at $r\le384$: BE {1, 2, 2.5}, CN-R {1, 2, 2.5}; Q1+Q2 per arm; cost from an
  in-job A–B–A timing stage (dev6 × 3 reps, reference A = R384 gauss128 frozen CN-R 1024², P1c mechanics); dominance by
  qualify's rule (median and worst E0 error vs B/A cost). GAL-CN is not re-run (diagnostic).
- Bank diagnostics: projection floor on the 38 cases' E0 states (best approximation in the r-prefix, scoring metric),
  $\mathrm{cond}(A)$ per $r$, training loss, rotation orthogonality.

### 3.3 3D (W1024 bank; widths 256, 512, 768, 1024)

Per seed: `train2w.py` with `train_r1024.json` and only the two seeds changed (12k bank steps, ordering by host QR on 65³
with least-squares coefficients, as the incumbent). **Memory pilot first** (the smoke, §6): data generation + 200 bank
steps, measured device peak (`memory_stats`) and host MaxRSS (sacct); production needs host RSS $\times1.25\le$ the
requested memory and device peak $\le120$ GB on H200 (143,771 MiB device).

**Rule selection (registered, applied to every bank including the incumbent).** For each width and the two specs
accurate `fs|CN|0.01|K1` and fast `qn|BDF2|0.025|k4`, on 64 cases: Gauss ladder gl16, gl20, gl24, gl32, gl40, gl48 and
lattice ladder lat8192, lat16384, lat32768, lat65536 (shift seed 0); $m^*$ = smallest rung that is valid, stationary, and
within $2.5\times10^{-4}$ of its successor and of gl64 on every case (the dims (3M) predicate). Anchor gate G-E2.2: on the
incumbent, the procedure's picks are compared with dims's m* table for the same cells (agreement reported; disagreement
→ like-for-like rule as in 3.2).

**Verdict runs (per seed):**
- H-acc-r across widths 256→512→768→1024 for both specs at each width's Gauss $m^*$ (V4-5 rule);
- E0 median and worst error per width and spec (the incumbent's 3D worst errors are 4–7 %: does a seed move them?);
- Q1+Q2 of the four configurations of qualify's 3D qualified subset, re-evaluated on each seed's bank: W256 gl24 FS CN
  0.01 and 0.0125; W256 lat8192 QN BDF2 0.025; W512 lat16384 QN BDF2 0.025.
- No 3D H-acc-N: no registered priority-1/2 3D H-acc-N verdict exists to re-run (speed3d's H4.1 is a cost verdict).

### 3.4 Decision rules (E2, frozen)

For every registered verdict $V$ (each H-acc-N cell, each H-acc-r transition, each selected rule, each configuration's
Q1/Q2 label, each 1024² front membership):
- **Seed-robust** if all three new seeds give the incumbent's label;
- **seed-dependent** if at least one new seed gives the opposite definitive label (pass↔fail, improves↔does-not-improve,
  member↔dominated, qualified↔not qualified, a different selected rule);
- **inconclusive** if differences involve only inconclusive/unavailable labels.
Summary counts per family of verdicts are reported; nothing is dropped.

**Seed spread with intervals.** Per (dimension, r, scheme, mesh): case-median and case-worst E0 error per bank. Reported:
min–max over the three new seeds; the incumbent's value; and a $t$ interval (df = 2) for the mean of $\log_{10}$ of each
statistic over the three new seeds, labelled "n = 3, wide by construction". The seed-to-seed spread is compared with the
r-step it would need to overturn: for each H-acc-r transition, the ratio (max over seeds − min over seeds of the median
error at $r$) / |median paired change $r\to r'$| is reported.

## 4. E5 — hardware

### 4.1 Panels (subsets; frozen)

- **P-cost (P1c subset).** Subjects R128 LSPG-BE $\Delta t_0$ fib4181 and R384 LSPG-CN-R $\Delta t_0$ gauss128, frozen
  modes, at 256² and 1024²; reference A = R384 gauss128 frozen CN-R 1024² (P1c's A). dev6 × 3 reps, P1c's A–B–A triplets,
  drift band [1/1.1, 1.1], retries, identity hashes, burn-in, warm JIT (p1time mechanics unchanged). 4096² is **H200-only**
  (the full bank peaks at 96.6–113.5 GB; an A100 80 GB cannot hold it), so the cross-hardware comparison is a **two-mesh
  panel**, labelled; the H200 job adds 4096² for the three-mesh H-cost-N verdict on H200 only.
- **P-cross (C2 subset).** At 1024²: ROM subjects R128 LSPG-BE $\Delta t_0$ fib4181 and R384 LSPG-CN-R $\Delta t_0$ gauss128
  (fused full-field query, fom2d's contract), FOM comparators frozen from fom2d's `FROZEN-SELECTION-1024.json` (the union
  winner, sine pseudo-spectral ETDRK4, and the selected central+CN-R same-mesh comparator), dev6 + val32 subset of 16
  cases × 3 reps, c2time stage-2 triplets. No re-selection on any hardware.

### 4.2 Runs

Two jobs per GPU type, each on a different physical GPU where the scheduler allows (GPU UUID recorded; same-UUID repeats are
labelled): **A100-80G ×2, H200 ×2**; L40S ×2 if free (defer item). Each job runs P-cost then P-cross. Recorded per job:
`nvidia-smi` name, UUID, driver, CUDA, SM/memory clocks, application clocks, power limit, temperature at start/end and
between panels; jax/jaxlib/plugin versions; `JAX_ENABLE_X64`, `JAX_DEFAULT_MATMUL_PRECISION=highest`, XLA flags.

### 4.3 Quantities and decision rules (frozen)

Only dimensionless quantities are compared across hardware: (q1) P-cost B/A ratio per subject and mesh (median over 6
cases × 3 reps); (q2) P-cost H-cost-N ratio $R=\max/\min$ over {256², 1024²} per subject; (q3) P-cross speed-up
$S=$ median FOM time / median ROM time per (ROM subject, FOM comparator); (q4) the verdict labels: H-cost-N (two-mesh,
bar 1.25, P1's rule) and the K2 cell "$S\ge2$" (fom2d's rule) per hardware type.

- **Quantity agreement.** For each quantity, $\phi=q_{\rm H200}/q_{\rm A100}$ (jobs pooled within type), 95 % interval by
  a paired case bootstrap (10,000 resamples, seed 20261019, cases resampled jointly across the jobs). *Hardware-consistent*
  if the interval lies within [0.8, 1.25]; *hardware-dependent* if it lies entirely outside; *inconclusive* otherwise. The
  same statistic between the two jobs of one type (unit-to-unit) is reported as the baseline.
- **Verdict agreement.** *Verdict-robust* if every hardware type gives the same definitive label; *hardware-dependent
  verdict* if two types give opposite definitive labels; *inconclusive* otherwise.
- Absolute times are never compared across types (reported per type only).

### 4.4 Controls (E5)

- **Must-fail C-E5.1 (injected work).** P1c's doubled-work control (subject B executed twice inside the timer) is run in
  every job; $\phi$ computed between the A100 normal and the H200 doubled B/A of R128 BE 1024² must be labelled
  *hardware-dependent* (expected $\approx2$). If it is not, E5 quantity verdicts are "blocked (control did not fire)".
- **Must-fail C-E5.2 (drift detection).** A synthetic CPU fixture with a 1.5× step change inside a triplet must be rejected
  by the drift band (unit test).
- **Positive gate G-E5.1.** On each GPU the P-cost results of R128 BE / R384 CN-R reproduce P1's scores bit-for-bit or to
  $\le10^{-10}$ (numerics independent of timing), and identity hashes of timed runs equal the scoring runs.
- **Positive gate G-E5.2.** H200 job reproduces P1c's B/A for the same subjects within the P1 interval (descriptive check of
  the copied harness; failure is reported, not blocking).

## 5. Cohorts, scoring, statistics (common)

Cases are the statistical unit; time snapshots are never bootstrapped. Bootstrap: 10,000 case resamples, percentile
intervals, seed 20261019. Eligibility per case-kind from the E0 store (the 13 non-rankable 2D case-kinds shown, never
ranked). Ties within $2\epsilon_{\rm ref}$. Three labels kept separate: numerical qualification (Q1+Q2), role
qualification (not used here), frontier usefulness. Every analysis runs on CPU (`JAX_PLATFORMS=cpu`); every GPU number is
from the cluster.

Q1 contract (V4-2): every run valid; $\gamma=\lVert J^\top r\rVert/(\lVert J\rVert_F\lVert r\rVert+10^{-300})$, non-stationary if
$\gamma>10^{-3}$, pooled fraction $\le1\%$; distance to the adaptive tight solve $\le2.5\times10^{-4}$ on every case;
tight/tighter $\le2.5\times10^{-5}$ on dev6. In E1 the tight-solve distance is computed for draw 0 and for every draw that
is the worst or the best by $D_s$ (solver error does not depend on the shift to first order; this is checked on those
draws, and the check is reported). GAL-CN uses qualify's residual criterion ($\eta\le10^{-8}$, its A1.3/A2.5).

## 6. Smoke plan (real size; ≤ 2 GPU-h each; one at a time)

| smoke | GPU | content | every control shown to fire |
|---|---|---|---|
| smk1 | H200 | E1 2D: E1-2D-a and -d at 1024², draws 0–2 + badlat4181, 6 cases; E1 3D: E1-3D-a, draws 0–2 + badlat8192, 8 cases; E5: P-cost and P-cross on 2 cases × 1 rep; clocks/version record | C-E1.1 (2D, 3D), C-E1.2, G-E1.1–3, C-E5.1, G-E5.1 |
| smk2 | A100-40G/80G | E2 2D pipeline at real sizes, truncated steps (Stage A 3,000, hfit 2,000), rotation, selection on 6 cases for R128 only, one H-acc-N cell at 256² and 1024² on 6 cases; resource certificate | G-E2.1 run path (not its outcome), identity checks |
| smk3 | H200 | E2 3D memory pilot: data generation + 200 bank steps, ordering, one width (W256) selection rung pair on 8 cases; device peak + host RSS | memory gate |

Smoke acceptance (`smoke_accept.py`) recomputes every predicate from raw outputs. Production resource certificates
(per job: setup, compile, per-rollout seconds × counts × 1.5) come from the smokes (`resources.py`, speed3d pattern).

## 7. Jobs and GPU-hour budget (estimates; certified after the smokes)

| job | content | GPU | est. GPU-h |
|---|---|---|---|
| e1a | E1 2D all configurations, 16 draws, 3 meshes (incl. 4096² 16 cases), controls, Sobol diagnostic | H200 | 4.0 |
| e1b | E1 3D two configurations × 16 draws + targets + controls + Sobol diagnostic | H200 | 0.6 |
| e5a1, e5a2 | E5 on A100-80G ×2 | A100 | 1.2 |
| e5h1, e5h2 | E5 on H200 ×2 (incl. 4096² on H200) | H200 | 1.6 |
| e2t2d | 2D Stage A, seeds 1–3, one GPU per seed in one job | A100 ×3 | 9.5 |
| e2b2d | 2D Stage B + rotation, seeds 1–3 | A100 ×3 | 4.5 |
| e2sel0 | incumbent rule selection (anchor G-E2.1) | H200 | 0.5 |
| e2v2d-s{1,2,3} | 2D selection + verdict runs + timing per seed | H200 | 3 × 2.0 |
| e2t3d-s{1,2,3} | 3D bank retrain per seed | H200 | 3 × 2.0 |
| e2v3d-s{0,1,2,3} | 3D selection + verdicts per bank (incl. incumbent) | H200 | 4 × 0.8 |
| smokes | smk1–3 | mixed | 3.0 |
| **total** | | | **≈ 40.1** |

With 25 % contingency ≈ 50 GPU-h ≤ 75. The ledger reserves each job's time limit atomically and refuses a submission if
spent + reserved + new limit > 75. **Defer order** (first dropped first; brief): E2 5-seed extension → E5 L40S → E1 3D
draws beyond 16 → E2 3D seed 3. **Never dropped:** E1 2D, E2 2D seeds 1–3, E5 A100 vs H200 on the P-cost subset. If a
never-dropped item cannot finish within the cap, the outcome is "E? incomplete", no verdict, shortfall reported. Extensions
(E2 seeds 4–5 ≈ +14 GPU-h; E5 L40S ≈ +1.2; E1 3D draws 16–63 ≈ +1.0) run only if certified spend allows.

## 8. Deliverables

`report.md` generated by `make_report.py` from run JSONs (level-1 title, status line, provisional labels inline, LaTeX,
mermaid, glossary); PNG plots (E1 draw distributions with the deployed draw marked; E2 seed spread per r; E5 $\phi$ with
intervals); `runs/EXPORT.json` (verdicts, key numbers, job ids); `runs/GPU-LEDGER.json`; audits in `audits/`; raw arrays in
`burgers-archive/jcp-robust/<job>/` with committed sha256 listings; new bank checkpoints (> 50 MB) tracked by sha256 in
`LARGE_FILES.sha256` and kept on the cluster archive.

## 9. What would change the paper (registered consequences)

- E1 fragile or atypically-favourable deployed draw on a front configuration → the 2D R128 front point is reported with its
  draw distribution, not the single draw.
- E2 seed-dependent H-acc-N or H-acc-r → the mesh-independence / r-dial statements are restated as "for the incumbent bank"
  with the seed spread shown.
- E5 hardware-dependent → cost-flatness or crossover statements are qualified by hardware type.

---

## Amendment A1 (2026-10-19, after Codex design audit 1; no data exists)

Audit `audits/codex-design-1.md`: NEEDS-RESTATEMENT (5 WRONG, 10 NEEDS-RESTATEMENT). Each item is answered below; where
A1 and the text above differ, A1 governs.

**A1.1 (audit 1, 3) E1 population and draw generation.** The E1 population is the **pinned qualify population**
(FRONT-2D v1 @2374fafe1 members + other_qualified; qualify's 3D qualified subset), not configurations qualified later
by other lanes (e.g. dims). Draws are produced only by calling the vendored generators, never by re-implementing them:
2D `hari_quadrature.fibonacci_lattice(k, seed=s)` (shift `np.random.default_rng(s).uniform(size=2)`, added after the
unshifted coordinates are reduced mod 1, exactly the generator's order); 3D `rules.rank1_lattice3(n, z, seed=s)` with
`z` recovered from the pinned deployed points (dims `rules3d_names.z_of`, bit-for-bit regeneration of $s=0$ asserted).
G-E1.1 compares sha256 of the regenerated $s=0$ points and weights against the deployed tables.

**A1.2 (audit 4) A draw "qualifies" only under the full contract.** A draw $s$ of configuration $c$ is **draw-qualified**
iff (i) Q1 in full for that draw: every run valid, pooled non-stationary fraction $\le1\%$, distance to the adaptive tight
solve $\le2.5\times10^{-4}$ on **every** case (tight solve run for **every** draw and case), tight/tighter agreement
$\le2.5\times10^{-5}$ on dev6 (2D) / cases 0–7 (3D) for **every** draw; (ii) Q2 against the **draw-matched successor**
(2D fib6765 with shift $s$; 3D lat16384 resp. lat32768 with shift $s$): rollout distance $\le2.5\times10^{-4}$ on every
case and $\rho\le0.116$ at own reached states; (iii) the additional cross-family gate: $d\le2.5\times10^{-4}$ against the
independent target (gauss384 / gl64) on every case. H-E1a uses draw-qualified. The §5 sentence restricting tight solves to
draw 0 and extreme draws is **withdrawn**. GAL-CN diagnostic rows: GAL's residual criterion replaces $\gamma$ (qualify
A1.3/A2.5), same structure; still no verdict.

**A1.3 (audit 5) Bad-vector control.** The control's must-fail outcome is defined at the advection level, where the
establishment is made: in job, the rule-gate computes $\rho_{\rm tgt}$ of `badlat4181` (2D) / `badlat8192` (3D) at the
same frozen fixture states as the CPU establishment, and the control **fires** iff the in-job $\rho_{\rm tgt}\ge0.116$
(the gate's bar) at $\ge1$ state **and** the in-job value agrees with the independent CPU value to $\le1\%$ relative at
every state. The CPU establishment (independent NumPy decode, admission $\rho_{\rm tgt}\ge1$ at every state) is computed
by this lane for these exact fixtures (dims's fixtures are different sizes/states and establish nothing here). Rollout
divergence or invalidity of the bad rule is reported as additional evidence, with no required outcome. Not admitted →
"control unavailable", E1 verdicts of that dimension blocked; admitted but not fired, or CPU/in-job disagreement → E1
verdicts of that dimension "blocked (control did not fire)".

**A1.4 (audit 6) Reproduction gate and precedence.** G-E1.2 / G-E5.1 / G-E2 reproduction compare **saved outputs**
(the six output coefficient vectors and the six scoring-lattice fields when saved) of the $s=0$ / incumbent run against the
archived run: max over cases and outputs of the relative max-norm distance $\le10^{-10}$. Archive map (read on the cluster
from `burgers-archive/jcp-qualify/<job>/` one file at a time): 2D 256² → qualify `acc256`, 1024² → `acc1024`,
4096² → `acc4096` (R128 fib4181 arms; LSPG and GAL); 3D E1-3D-a → `q3d3` (`W256|lat8192|qn|BDF2|0.025|k4`), E1-3D-b →
`q3d3` (`W512|lat16384|qn|BDF2|0.025|k4`); E5 P-cost → p1-scaling `time` archive scores. Expected counts are frozen per
configuration (2D: 38/38/16 cases × 6 outputs; 3D: 64 × 6); a missing archive file, a missing case, a count mismatch or a
non-finite value makes the gate **fail** (never pass by an empty reduction). Every statistic over an empty or partially
missing set is "unavailable", never pass.

Precedence (applied in this order; the first that applies sets the label of every verdict it covers):
1. job integrity (manifest, `jax_backend=gpu`, x64 + `highest` asserted, identity hashes of tables, expected counts) → "invalid job";
2. must-fail control of the dimension/panel unavailable or not fired → "blocked (control)";
3. positive gate (G-E1.1 identity, G-E1.2/G-E2.x reproduction, G-E1.3 permuted batching) failed → "blocked (gate)";
4. missing / non-finite evidence for that verdict → "unavailable";
5. the registered decision rule.

**A1.5 (audit 7) E1 statistics.** The 16/16 bound is the **one-sided** 95 % Clopper–Pearson upper bound (0.171); the bound
on the 15 fresh draws alone (0/15 → 0.181) is reported beside it, because draw 0 was kept for having passed qualification
(selection-conditioned). Ranks are **mid-ranks** (ties share the average rank); two values are tied for ranking if equal
to relative $10^{-12}$ (distances) or within $2\epsilon_{\rm ref}$ of each other (errors, case-median / case-worst taken over
eligible cases). If all 16 values of a statistic are tied, that statistic is labelled "insensitive (all tied)" and does not
enter the typicality label. A draw with any invalid run gets $D_s=+\infty$ and $+\infty$ errors for its invalid cases (ranked
worst). Zero eligible cases → "unavailable".

**A1.6 (audits 9, 10) E2 2D recipe frozen exactly; conditional estimand.** Estimand: **algorithmic robustness of the recipe
conditional on the fixed training data set** (it does not measure sensitivity to resampling the training trajectories).
Streams: fixed — the 576 canonical trajectories, the Stage-A 16,384-state pick (seed 0, smoothtrain line 403), the
Stage-B extraction (`sep_coeff_extract.py` with the exact `run_dn256b.sbatch` environment `N=256 K=16 R=512
MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=1 EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256
IDENT_ROWS=64`; its `SEED0` is the state pick, i.e. data, so it stays 0). Varying with $s$ — Stage A `--seed s` (network
initialisation and the per-step random spatial point subsets `p_sub`; value-only training uses all picked states, so the
earlier word "minibatch" is withdrawn); Stage B head fit `sep_hfit_run.py` with the exact "full" call of
`run_dn256b.sbatch` (`ARMS=mid,wide,k32 STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 ORACLE_ITERS=150 CODEDIAG_N=2048
CODEDIAG_ITERS=60 ENC_STEPS=12000 EMIT=mid`) with `SEED0=s` (keys `SEED0+7`, `SEED0+11`). The F=576/4096 learning-curve
calls and `sep_speed_r5.py` are diagnostics of the original job and are not run. The emitted dense `mid` checkpoint (all
131,072 fitted codes) feeds the head-based rotation; smoothtrain's own least-squares rotation is **discarded** as a
deployment input (kept only as a diagnostic file). The incumbent is a **historical comparator** (older software, default
autotuning): incumbent-vs-new differences are not attributed to the seed alone; seed-to-seed spread is measured among the
three new seeds.

**A1.7 (audit 11) 3D checkpoint selection.** `train2w.py`'s own bank-validation checkpoint selection is preserved (the
incumbent picked step 11,000); the selected step is recorded per seed. Never the final step by default.

**A1.8 (audits 12, 13) Rule selection, ladders, "none", target checks.**
- 2D candidates (ordered by $m$) and their registered successors: fib4181→fib6765, fib6765→fib10946, gauss96→gauss128,
  fib10946→fib17711, gauss128→gauss160, gauss160→gauss192, gauss192→gauss256. Target gauss384; **target check per bank**:
  gauss384 vs gauss320 rollouts on dev6 under both schemes within $2.5\times10^{-5}$ (a tenth of the bar); failing →
  selection labelled "target unverified" (reported; selection still made). Outcomes per $r$: a selected rule, or **"none
  within ladder"**, or "unavailable" (invalid/missing evidence).
- 3D: candidates gl16, gl20, gl24, gl32, gl40, gl48 and lat8192, lat16384, lat32768, lat65536 (dims's candidate lists),
  certification successors gl20…gl56 and lat16384…lat131072, target gl64 with target check gl64 vs gl56 on cases 0–7 within
  $2.5\times10^{-5}$. Same outcome set. dims already reports "none" for some incumbent cells; that is an expected outcome.
- **Verdict runs when no rule is selected:** the cell's primary verdicts are "unavailable (no rule)"; a secondary row runs
  the deployed rule (2D R128 fib4181 / R256, R384 gauss128; 3D the fixed panel rule below) with its Q1/Q2 labels shown.
- **Incumbent under the identical procedure:** the incumbent bank gets the same selection, verdict, timing and front runs
  as each seed (jobs `e2v2d-s0`, `e2v3d-s0`); its historical (deployed-front) labels are kept in a separate column and are
  never mixed with the restricted-universe labels.

**A1.9 (audit 14) 3D front per bank.** Candidate universe (frozen, per bank): widths {256, 512, 768, 1024} × specs
{`fs|CN|0.01|K1`, `fs|CN|0.0125|K1`, `qn|BDF2|0.025|k4`} × {the width's selected Gauss rule, its selected lattice rule
(each if not "none")}, plus the four configurations of qualify's 3D subset. Each candidate: Q1 + Q2; cost by an in-job
A–B–A timing stage on cases 0–7 × 3 reps, reference A = `W256|gl24|fs|CN|0.01|K1` on the job's own bank; dominance by
qualify's rule (median and worst E0 error vs B/A cost). Label: "front of the tested 3D universe" (never "complete").
3D H-acc-r uses a **fixed rule gl64 for every width and bank** (isolates the bank; Q2 against gl80 on cases 0–7 reported;
a width whose gl64 run fails Q2 makes its transitions "unavailable"), in addition to the per-bank selected-rule rows.

**A1.10 (audit 15) E2 labels.** Over the four banks (incumbent + seeds 1–3), per verdict:
- **seed-dependent** if any two banks give opposite definitive labels (pass vs fail; improves vs does-not-improve; member
  vs dominated; qualified vs not qualified; two different selected rules, or a rule vs "none");
- **seed-robust** if all four labels are definitive and equal;
- **inconclusive** otherwise (any inconclusive/unavailable label without a definitive disagreement).
"Unavailable (no rule)" and "missing bank" are reported as their own states and never count as agreement. Spread ratio with
a zero paired-change denominator is reported as "∞ (no r-step)". Three seeds support a description of agreement, not a
population-level robustness claim; the report says so.

**A1.11 (audits 16–19) E5 restated.**
- The A100 restriction applies to the existing full-bank implementation (not an impossibility result).
- P-cross pins `runs/c21024/archive/output/c2time_c21024/FROZEN-SELECTION-1024.json` of fom2d (sha256 recorded in the
  config) and the exact selection keys `<ROM key>@1024|H|union|joint`, `|PS|joint` and `|CC|joint` for the two ROM subjects;
  duplicate comparator selections (union = PS = `ps|N96|0.0125`) are timed once. Labelled "frozen H200-selected
  comparators", not the best comparator per GPU. Timing cases: dev6|0–5 and val32|0–9 (16), enumerated in the config.
- Estimators inherited unchanged: P-cost B/A ratio per triplet = $t_B/(\tfrac12(t_{A1}+t_{A2}))$, per-case median over valid
  reps (p1time / p1stats); P-cross $S$ = fom2d `c2analyze.speedup` (ratio of case medians of within-triplet normalised
  ratios, common valid cases, ≥ 5 cases).
- Named subset tests (they do **not** replace the registered verdicts): **H-cost-N(2)**: P1's `h_cost_n` statistic on
  {256², 1024²} only (pass iff upper 95 % bound ≤ 1.25, fail iff lower > 1.25, else inconclusive), labelled "two-mesh subset";
  the H200 jobs also give the three-mesh value (H200 only). **S2-subset**: per (ROM subject, comparator), pass iff the lower
  95 % bound of $S\ge2$, fail iff the upper bound $<2$, else inconclusive; this is not K2 (K2 needs every configuration at
  the largest mesh).
- Aggregation: per job, per case: median over valid reps; pooled per type: mean over that type's jobs of the per-case
  values, on the intersection of cases valid in **all** jobs of both types; quantity = median over cases; $\phi$ = ratio of
  pooled quantities; paired case bootstrap (cases resampled jointly). Fewer than 5 common cases → "unavailable". Ratios only
  for q1–q3; q4 (verdict labels) only by the verdict-agreement rule. Per-job values are reported beside the pooled ones, and
  all conclusions are stated as conditional on the observed devices (two GPUs per type cannot represent a hardware population).

**A1.12 (audit 20) E5 controls replaced.**
- **C-E5.1 (within-GPU doubled work, must fire in every job):** P1's doubled-work control (B twice inside the timer,
  flanked by same-subject single executions) must give a doubled/single ratio with case-median in [1.6, 2.4]; otherwise that
  job's E5 numbers are "blocked (control)".
- **C-E5.2 (classifier must-fail, synthetic on real data):** the cross-hardware classifier is applied to (A100 job-1 per-case
  values, the same values × 1.5) → must return *hardware-dependent*; and to (A100 job 1, A100 job 2) → reported (unit-to-unit
  baseline, no required outcome); a classifier that does not flag the ×1.5 fixture blocks every E5 quantity verdict.
- **C-E5.3** drift-band unit test (synthetic 1.5× step inside a triplet must be rejected), CPU.
- An **A100 smoke (smk4, ≤ 0.5 GPU-h)** runs the E5 panels on 2 cases × 1 rep so the A100 path and C-E5.1 are shown on
  real A100 data before production.

**A1.13 (audit 21) Budget rebuilt from counts (projections; the smokes certify).**
- E1 2D: per mesh job, 4 LSPG configs × cases (38/38/16) × 16 draws × (prod + tight + successor) + tighter on dev6 + GAL
  diagnostic (prod + target) + targets once per (config, case); ≈ 23k rollouts at 0.3–2 s → three per-mesh jobs, 8.5 GPU-h
  total. E1 3D ≈ 6.5k rollouts + targets → 0.6 GPU-h.
- E2 2D: Stage A 3 GPUs × ≈ 3.1 h = 9.5; Stage B 3 GPUs × ≈ 1.5 h = 4.5; selection + verdict + timing per bank ≈ 2.5 h
  × 4 banks (incl. incumbent) = 10.0. E2 3D: training 3 × 2.0 = 6.0; selection + gl64 panel + front per bank ≈ 3.0 h
  (dims's ladder jobs used 1.5–2.9 GPU-h) × 4 = 12.0.
- E5: 4 jobs × 0.6 = 2.4. Smokes smk1–smk4 ≈ 4.5.
- **Total ≈ 58.0 GPU-h**; 25 % contingency would give 72.5, inside the 75 cap. Multi-GPU jobs are charged and reserved at
  (GPUs × time limit). The ledger refuses any submission with spent + reserved + new reservation > 75; the defer order of §7
  applies (E2 3D seed 3 frees ≈ 5 GPU-h; E2 5-seed extension is not affordable under this projection and is expected to be
  deferred).
- smk3 is extended to the **heaviest 3D evaluation cell** (W1024 gl64 target + gl80 Q2 on 4 cases) to measure compile time and
  device peak, in addition to the training memory pilot.

---

## Amendment A2 (2026-10-19, after Codex design audit 2; no data exists)

Audit `audits/codex-design-2.md`: NEEDS-RESTATEMENT (items 6, 17, 18, 23, 24, 28 WRONG; 7, 12, 14, 19–21, 25–27, 29–31
NEEDS-RESTATEMENT). A2 governs over A1 and the body where they differ.

**A2.1 (item 6) E1 2D cases and the archive map.** Qualify qualified every R128 fib4181 arm on **all 38 cases at every
mesh** (its `acc4096` adds val32|10–31 for the factor-1 arms to P1's 16), so E1 uses **38 cases at 256², 1024² and
4096²** (the 16-case 4096² subset of §2.1 is withdrawn). Reproduction archives (key `W`, shape (n, 6, 128), with
`case_ids`):

| arm | 256² | 1024² | 4096² |
|---|---|---|---|
| LSPG BE 1 | P1 `runs/acc256/.../R_R128_BE.npz` (38) | P1 `acc1024` `R_R128_BE.npz` (38) | P1 `acc4096` `R_R128_BE.npz` (dev6 + val32\|0–9) ∪ qualify `acc4096` `R_R128_LSPG_BE_1.npz` (val32\|10–31) |
| LSPG CN-R 1 | P1 `acc256` `R_R128_CNR.npz` | P1 `acc1024` `R_R128_CNR.npz` | P1 `acc4096` `R_R128_CNR.npz` ∪ qualify `acc4096` `R_R128_LSPG_CNR_1.npz` |
| LSPG BE 2, 2.5; GAL CN 1, 2, 5 | qualify `acc256` `R_R128_<form>_<scheme>_<factor>.npz` (38) | qualify `acc1024` (38) | qualify `acc4096` (38) |

P1 archives are local (`worktrees/2026-10-17-jcp-p1-scaling/.../runs/acc*/archive/output/p1acc_acc*/`); qualify archives
are on the cluster (`burgers-archive/jcp-qualify/acc*/`), fetched one file at a time into scratch and deleted after. The
file sha256 is checked against the owning lane's listing before use. Norm: per case, $\max_{k,j}|W^{\rm new}_{kj}-W^{\rm arch}_{kj}|
/\max_{k,j}|W^{\rm arch}_{kj}|$; a zero denominator, a missing case, a case-id mismatch or a non-finite entry fails the
gate. The comparison runs on CPU from the job's saved `W` (no GPU needed). 3D: qualify `q3d3` archive entries of
`W256|lat8192|qn|BDF2|0.025|k4` and `W512|lat16384|qn|BDF2|0.025|k4` (64 cases × 6 outputs), same norm.

**A2.2 (items 7, 25) Typicality by rank intervals.** For a statistic $v$ (lower is better) with tolerance $\delta$:
$k_{\rm lo}=1+\#\{s\ne0: v_s<v_0-\delta\}$, $k_{\rm hi}=16-\#\{s\ne0: v_s>v_0+\delta\}$. Tolerances: $\delta=0$ for $D_s$;
for the case-median and case-worst E0 error, $\delta=2\,\mathrm{median}_i\,\epsilon_{{\rm ref},i}$ over the eligible cases
(the aggregate margin). Per statistic: **insensitive** if $k_{\rm lo}=1$ and $k_{\rm hi}=16$; **favourable** if
$k_{\rm hi}\le2$; **unfavourable** if $k_{\rm lo}\ge15$; **typical** if $3\le k_{\rm lo}$ and $k_{\rm hi}\le14$;
**indeterminate** otherwise. Configuration label: if all three statistics are insensitive → "**insensitive; typicality not
assessed**"; otherwise, over the non-insensitive statistics: any favourable and no unfavourable → atypically favourable;
any unfavourable and no favourable → atypically unfavourable; both → mixed; all typical → typical; else indeterminate.
The probability 1/4 of an extreme label applies only to an unconditioned, tie-free exchangeable model and is quoted only
with that qualification (draw 0 is selection-conditioned).

**A2.3 (item 23) Failure vs missing evidence.** *Missing evidence* (no output file, crashed/timed-out job, absent record,
count mismatch) → "unavailable" by precedence step 4. A *recorded numerical failure* (a run that completed and is flagged
invalid: non-finite state, non-finite step, damping saturation) is evidence: that draw is **not draw-qualified** (counts
as a failure in H-E1a), and its statistics take the explicit state "failed", ranked worst ($v_s=+\infty$ for ranks); it never
passes through the generic finiteness check of step 4.

**A2.4 (item 24) Gates enumerated individually.** Blocking (precedence step 2/3): C-E1.1, C-E1.2, G-E1.1, G-E1.2, G-E1.3
(E1); **G-E2.3** — the incumbent's verdict runs at the deployed rules reproduce the P1/qualify archives by the A2.1 norm
($\le10^{-10}$; checks the copied code path) — blocks that dimension's E2 verdicts; C-E5.1 (per job), C-E5.2 fixtures,
G-E5.1 (E5). **Non-blocking anchors** (reported, never block): G-E2.1 and G-E2.2 (incumbent rule recovery; their
registered fallback is the procedure-selected rule for every bank), G-E5.2 (copied-harness reproduction of P1c ratios).

**A2.5 (item 12) Target-unverified selections.** If a bank's target check fails, its selection is a **diagnostic pick**:
the cross-family criterion is reported as "unavailable (target unverified)"; the pick is made on Q1 + successor Q2 only;
every downstream verdict from that bank carries "target unverified" and is excluded from seed-robust/seed-dependent
counts that need the cross-family criterion (the selected-rule verdict). Successor-Q2 and E0-error verdicts remain
available.

**A2.6 (items 14, 26, 27) 3D selection keys, fallbacks and stable identities.** 3D selection is per **(bank, width, spec,
family)** for the three specs `fs|CN|0.01|K1`, `fs|CN|0.0125|K1`, `qn|BDF2|0.025|k4` (the 0.0125 spec is added to the
ladders), families Gauss and lattice. Front candidates are **recipe slots** (width, spec, family) with their realised rule;
a slot whose selection is "none" has no candidate (no fallback run for W768/W1024 or any other width); the four
fixed qualify configurations are a **separate fixed-configuration comparison** (deduplicated if a slot realises exactly
the same configuration). 2D identities likewise: slots (r, scheme, factor) with the realised selected rule; when a bank's
selected rule differs from the deployed rule, the deployed-rule configuration is also run and compared in a separate
fixed-configuration table. Cross-bank comparisons of front membership are made per slot ("slot present and member /
present and dominated / absent (no rule)") and, separately, per fixed configuration. The gl64 H-acc-r panel stays separate.

**A2.7 (items 28, 31) Seed attribution and spread ratio.** Among the three new seeds only: **seed-robust** if all three
labels are definitive and equal; **seed-dependent** if two definitive labels differ; **inconclusive** otherwise. Separately,
**historical agreement**: when the new seeds are seed-robust, "agrees with incumbent" / "differs from incumbent" (a
recipe/software-version difference, not attributed to the seed) / "incumbent not definitive". "None within ladder" is a
definitive selection label (it can agree or disagree); downstream verdicts of an absent slot are "unavailable". Spread ratio
per transition $r\to r'$: numerator = max − min over seeds 1–3 of the case-median error at $r$; denominator = median over
seeds 1–3 of $|\Delta_s|$ ($\Delta_s$ = that seed's median paired change $r\to r'$); positive/0 → "∞"; 0/0 → "undefined".

**A2.8 (items 17, 18, 19) E5 estimators restated.**
- P-cross same-mesh comparator key is `<ROM key>@1024|H|CC|same_mesh` (fom2d file: R128 → `fd|CC|L1024|0.0125|t2`,
  R384 → `fd|CC|L1024|0.005|t1`), plus `|H|union|joint` (= `|PS|joint`, `ps|N96|0.0125`, timed once). `CC|joint`
  (L64/L128) is **not** used.
- Per job, H-cost-N(2) uses P1's `p1stats.h_cost_n` **unchanged** (median over all valid (case, rep) entries per mesh,
  case bootstrap) on the {256², 1024²} columns.
- Pooled cross-hardware estimators (new, named): for type $T$, subject $b$, mesh $L$, let $E_{T,b,L}$ be all valid
  (job, case, rep) B/A entries on the common case set $C$ (cases with ≥ 2 valid reps for every subject, mesh and job of
  both types; $|C|\ge5$ or "unavailable").
  - $q_1(T,b,L)=\mathrm{median}\,E_{T,b,L}$;
  - $q_2(T,b)=\max_L q_1(T,b,L)/\min_L q_1(T,b,L)$ over {256², 1024²};
  - $q_3(T,b,f)$: per case $c\in C$, $m^{\rm FOM}_c$ = median over the type's valid (job, rep) normalised FOM ratios of
    comparator $f$, $m^{\rm ROM}_c$ likewise for ROM subject $b$; $q_3=\mathrm{median}_c\,m^{\rm FOM}_c/\mathrm{median}_c\,m^{\rm ROM}_c$
    (fom2d's `speedup` form, pooled over jobs).
  - $\phi_k=q_k({\rm H200})/q_k({\rm A100})$. Each bootstrap replicate resamples case ids from $C$ jointly for both types
    and all subjects, keeps every entry of the resampled cases, and recomputes the full statistic.
- **A2.9 (items 20, 29) Estimator-specific must-fail fixtures** (on real A100 job-1 data, CPU): q1 — multiply subject
  R128-BE's 1024² entries by 1.5 in a copy treated as "H200" → $\phi_1$ must be *hardware-dependent*; q2 — multiply only the
  1024² entries of each subject by 1.5 → $\phi_2$ must be *hardware-dependent* for every subject whose A100 $q_1$ minimum is
  not at 1024² (others reported); q3 — multiply only the FOM entries by 1.5 → $\phi_3$ *hardware-dependent*. Null fixture
  (job 1 vs itself) → *consistent* with interval [1, 1]. A failed fixture for $q_k$ blocks the $\phi_k$ verdicts **and** the
  dependent q4 verdict-agreement conclusion (q2 → H-cost-N(2) agreement; q3 → S2-subset agreement).
- **smk4** (A100, ≤ 0.5 GPU-h) runs P-cost and P-cross on **5 cases (dev6|0–4) × 2 reps**, so the full E5 analysis path
  (≥ 5 common cases, ≥ 2 valid reps) and C-E5.1/C-E5.2 run end to end on real A100 data before production.

**A2.10 (item 30) Head-fit stopping rule.** The historical command stops at 200,000 steps **or** 1,500 s, whichever comes
first (`TIME_CAP=1500`); the incumbent's `mid` arm completed 200,000 steps in 339 s (`hfit_dn256b_full.json`). The command is
kept exactly; `steps_done`, whether the cap fired, and the device are recorded; Stage B runs on A100-80G (the incumbent's
type). A seed whose `mid` fit stops on the time cap is reported as a recipe deviation for that seed.

**A2.11 (item 21) Budget inventory.** E1 2D rollouts per (configuration, mesh): 38 cases × 16 draws × (prod + tight +
successor) + 6 × 16 tighter + 38 target = 1,958; × 7 configurations (GAL included, same structure) × 3 meshes ≈ 41,100;
Sobol diagnostic 38 × 16 × 2 ≈ 1,200; controls ≈ 300 → **≈ 42,600 rollouts, 9.5 GPU-h** at a mean 0.6 s plus compile and
4096² decode. E1 3D: 2 × 64 × 16 × 3 + 2 × 8 × 16 tighter + targets + Sobol diagnostic 64 × 16 ≈ 7,500 → 0.7 GPU-h.
E2 3D selection now covers three specs: 3.5 GPU-h per bank × 4 = 14.0. Revised total ≈ **61.2 GPU-h** (body/A1 lines
otherwise unchanged); with the 75 cap, E2 seeds 4–5 are expected to be deferred and E2 3D seed 3 is the next item at risk.
Every figure is a projection until the smoke certificates exist.

---

## Amendment A3 (2026-10-19, after Codex design audit 3; no data exists)

Audit `audits/codex-design-3.md`: NEEDS-RESTATEMENT (item 32 WRONG; 19–21, 29, 33–41 NEEDS-RESTATEMENT; 1–18, 22–28,
30, 31 resolved). A3 governs over A2, A1 and the body.

**A3.1 (item 41) Historical population wording.** Corrected: qualify scored all 38 cases at every mesh, and qualified
these arms under its registered Q2 subset (`q2_cases: sub16`). E1's every-case successor check is a **stronger test**; a
draw-0 successor failure on a case outside sub16 is a finding of E1, not an archive-reproduction failure (G-E1.2 compares
only the saved production `W`).

**A3.2 (items 23, 36) Failure states at case level.** For each (configuration, draw, case) the record carries one of:
`ok`, `failed-prod` (production run invalid), `failed-tight`, `failed-tighter`, `failed-succ`, `failed-target` (the
respective run completed and was flagged invalid), or `missing`. Rules: any `failed-*` except `failed-target` makes the
draw not draw-qualified; `failed-target` makes the cross-family criterion of that case "target failed" (A3.3). H-E1c: a
case whose draw-$s$ production run is `failed-prod` counts as **outside the margin** for that case (never dropped); the
case-median and case-worst error of a draw with any `failed-prod` eligible case are "failed" (ranked worst, +∞ in A2.2);
descriptive summaries (median, IQR, min–max over draws) are computed only over draws with no failed case and are printed
with the count of failed draws; no arithmetic is done on a failed value. `missing` → "unavailable" for every statistic
that needs that case.

**A3.3 (item 37) Target certification keyed per configuration.** The independent target is certified per (bank, r or
width, form, scheme, $\Delta t$, mesh): 2D — target gauss384 vs check gauss320, every case of dev6, all six outputs,
distance $\le2.5\times10^{-5}$, both runs valid; 3D — gl64 vs gl56 on cases 0–7. This is run for **every** E1 configuration
(7 per mesh in 2D, 2 in 3D) and every E2 selection cell. If a configuration's target is not certified, its cross-family
criterion (A1.2 iii) and the target-dependent quantities ($d$, $\rho$ vs target, $D_s$, H-E1b's $D_s$ statistic) are
"unavailable (target not certified)"; H-E1a is then evaluated on Q1 + own-family Q2 only and labelled so; H-E1c (E0 error,
target-free) is unaffected except its margin's $e_{\rm conv}$ term, which falls back to $2\epsilon_{\rm ref}$ alone (labelled).

**A3.4 (item 40) Rank-interval tolerance for error statistics.** Conservative bound: for the case-median and case-worst
error, $\delta=2\max_i\epsilon_{{\rm ref},i}$ over the eligible cases (a uniform bound on every case's reference error
propagates through any order statistic, since $|\mathrm{med}(a)-\mathrm{med}(b)|\le\max|a-b|$ and likewise for the maximum).
The A2.2 median-based $\delta$ is withdrawn.

**A3.5 (item 38) Slot and fixed-configuration states.** Every 2D/3D front slot and fixed configuration per bank has one
state: `member`, `dominated` (dominance computed among Q1+Q2-qualified candidates only), `unqualified` (selected or fixed
but failing Q1/Q2 in that cell), `absent` (no rule selected), `missing`, `blocked`. Qualification agreement is compared
across banks separately from membership; an unqualified candidate is never called dominated. For E2 labels (A2.7), the
definitive labels are member/dominated/unqualified/absent; missing/blocked are non-definitive.

**A3.6 (item 35) G-E2.3 scope.** G-E2.3 is the incumbent replay check, enumerated:
- 2D: incumbent bank, R128 fib4181 LSPG BE 1 and CN-R 1 at 256²/1024² (P1 `acc256`/`acc1024` `R_R128_BE.npz`,
  `R_R128_CNR.npz`), R384 gauss128 BE 1 / CN-R 1 at 1024² (P1 `acc1024` `R_R384_BE.npz`, `R_R384_CNR.npz`), R256 gauss128
  BE 1 / CN-R 1 at 1024² (P1 `acc1024` `R_R256_BE.npz`, `R_R256_CNR.npz`), all 38 cases, key `W`; file sha256 recorded in the
  config at staging (from the P1 lane's archive listing).
- 3D: incumbent W1024 bank, qualify `q3d3` archive for `W256|lat8192|qn|BDF2|0.025|k4` and `W512|lat16384|qn|BDF2|0.025|k4`
  and qualify `q3d2` for `W256|gl24|fs|CN|0.01|K1`, 64 cases.
- Norm and failure handling as A2.1. Scope: a failure blocks every E2 verdict of that dimension (the code path is suspect).
  Never applied to newly trained banks. **Smoke coverage:** smk2 runs the 2D incumbent replay of R128 BE 1 on dev6 at 1024²;
  smk3 the 3D incumbent replay of `W256|lat8192|qn|BDF2|0.025|k4` on cases 0–7.

**A3.7 (items 32, 29) Estimator-level fixtures with provable outcomes.** All fixtures are built from one real, positive
A100 array (job 1; at least 5 cases × 2 valid reps), pair a baseline copy with an injected copy, and run through the same
estimator and bootstrap code:
- q1: baseline = the entries of R128-BE 1024²; injected = the same entries × 1.5 → every resample gives $\phi_1=1.5$ exactly →
  must be *hardware-dependent*.
- q2: baseline = the R128-BE 1024² entries **copied to both mesh columns** (so baseline $q_2\equiv1$ in every resample);
  injected = the same with the 1024² column × 1.5 (so $q_2\equiv1.5$) → $\phi_2\equiv1.5$ → must be *hardware-dependent*.
- q3: baseline = the real (FOM, ROM) arrays; injected = FOM entries × 1.5 → $\phi_3\equiv1.5$ → *hardware-dependent*.
- null: baseline vs itself → $\phi\equiv1$ → *consistent*.
Each fixture is required (coverage cannot be empty, since it is built from one fixed subject); a fixture that does not
return its required label blocks the corresponding $\phi_k$ verdicts and the dependent q4 agreement.

**A3.8 (item 33) Panel-specific case sets.** $C_{\rm cost}$ = P-cost cases (dev6) valid (≥ 2 valid reps) for both subjects at
256² and 1024² in every job of both types; $C_{\rm cross}$ = P-cross cases (dev6|0–5, val32|0–9) valid for both ROM subjects and
all timed comparators in every job of both types. H200-only 4096² cells never enter a cross-type intersection. Requested and
retained case ids are reported for each statistic; $|C|<5$ → "unavailable".

**A3.9 (items 18, 34) q4 constructors.** Per job, P1's `h_cost_n` on {256², 1024²} returns `procedure_outcome`
(pass/fail/inconclusive on the two-mesh data) and a non-definitive registered `outcome`; **H-cost-N(2) uses
`procedure_outcome`**. Per-type label: from the pooled type statistic $q_2(T,b)$ with its own case bootstrap (pass iff upper
95 % bound $\le1.25$, fail iff lower $>1.25$, else inconclusive); likewise S2-subset from pooled $q_3(T,b,f)$ (pass iff lower
bound $\ge2$, fail iff upper $<2$). Per-job labels are reported beside them; if the two jobs of a type give opposite
definitive per-job labels, the type's label is "inconclusive (jobs disagree)" regardless of the pooled interval; a missing
or blocked job makes the type label rest on the remaining job and is labelled "single job".

**A3.10 (items 21, 39) Workload inventory and cap.** Unique executions per job (projection; compile signatures in brackets;
each job's certificate comes from the smoke measurements):

| job | executions | compile signatures |
|---|---|---|
| e1-256, e1-1024, e1-4096 (2D, one per mesh) | 7 configs × 38 × 16 draws × 3 (prod, tight, succ) = 12,768; tighter 7 × 6 × 16 = 672; targets 7 × 38 = 266; target checks 7 × 6 = 42; bad vector 7 × 38 = 266; Sobol diag 38 × 16 = 608; permuted gate 38 → ≈ 14,660 per mesh | per (form, scheme, factor, m): 7 forms/steps × {4181, 6765, 147456, 102400, 8192} |
| e1-3d | 2 × 64 × 16 × 3 = 6,144; tighter 2 × 8 × 16 = 256; targets 128; checks 16; bad 128; Sobol 1,024; permuted 64 → ≈ 7,760 | 2 specs × 6 rule sizes |
| e2sel-2d-b (b = 0..3) | 3 r × 7 candidates × 2 schemes × 38 × 3 = 4,788; tighter 3 × 7 × 2 × 6 × 1 = 252; terminal successor gauss256 3 × 2 × 38 = 228; target + check 3 × 2 × (38 + 6) = 264 → ≈ 5,530 | 3 r × 8 sizes × 2 |
| e2ver-2d-b | 256² and 4096²: 3 r × 2 schemes × 38 × 3 × 2 meshes = 1,368; 1024² factors 2/2.5 BE+CN-R: 3 × 4 × 38 × 3 = 1,368; fixed deployed rows when the rule differs (≤ 2,736); timing 3 r × 6 arms × 6 cases × 3 reps × 2 (A–B–A) = 648 → ≤ 6,120 | |
| e2sel-3d-b | 4 widths × 3 specs × 10 rungs × 64 × (prod + ADT tight) = 15,360; tighter 4 × 3 × 10 × 8 = 960; successors gl56/lat131072 top rungs 4 × 3 × 2 × 64 = 1,536; target + check 4 × 3 × (64 + 8) = 864 → ≈ 18,720 | 4 widths × 13 sizes × 3 specs |
| e2ver-3d-b | gl64 panel + gl80 Q2: 4 × 2 specs (accurate, fast) × (64 + 8) = 576 (gl64 reused from selection); fixed configs 4 × 64 × 3 = 768; front timing ≤ 24 slots × 8 cases × 3 reps × 2 = 1,152 → ≈ 2,500 | |
| e5 (×4) | P-cost 2 subjects × 2 meshes × 6 × 3 triplets (+ 4096² on H200) + doubled; P-cross (2 ROM + 3 FOM) × 16 × 3 triplets | inherited |

Per-bank 3D cost is therefore projected at ≈ 3.5–4.5 GPU-h (selection) + 0.7 (verdicts), refining A2.11's 3.5. Revised
projection: E1 2D 9.5, E1 3D 0.7, E2 2D (training 14.0 + 4 × 2.5) 24.0, E2 3D (training 6.0 + 4 × 4.5) 24.0, E5 2.4, smokes
4.5 → **≈ 65.1 GPU-h**, leaving ≈ 9.9 GPU-h (15 %) of the 75 cap as contingency — less than the 25 % used in earlier
planning. The cap is hard: the ledger reservation check (§7) and the registered defer order govern. Consequences fixed now:
E2 seeds 4–5 deferred; if the smoke certificates push the projection above 75, **E2 3D seed 3** is dropped next (both its
training and evaluation), then E1 3D draws are kept at 16 (already the minimum). If even that does not fit, the E2 3D
incumbent evaluation runs on the four existing fixed configurations only (labelled), before any never-dropped item is
touched.

---

## Amendment A4 (2026-10-19, after Codex design audit 4; no data exists)

Audit `audits/codex-design-4.md`: one BLOCKING item (prior 19/34: the single-job fallback contradicts the
every-job case intersection); the rest RESOLVED or IMPLEMENTATION-NOTE.

**A4.1 (blocking item) No single-job fallback.** The A3.9 sentence "a missing or blocked job makes the type label rest on
the remaining job" is **withdrawn**. E5 requires both jobs of each compared type. If any required job is missing, failed,
or blocked by C-E5.1, then every cross-type quantity ($\phi_1$–$\phi_3$) and every per-type q4 label involving that type is
"unavailable (job missing)", and the lane resubmits that job (a new attempt, charged to the ledger) before reporting; per-job
values of the surviving jobs are still reported, labelled "per-job only". Case sets $C_{\rm cost}$, $C_{\rm cross}$ are
therefore always built over the full set of four (or six, with L40S) jobs.

**A4.2 Implementation notes carried into the code audit** (not design changes):
- validity is stored as separate flags for production, tight, tighter, successor, target and check runs; the A3.2 status
  is derived from them with the precedence `missing` > `failed-prod` > `failed-tight` > `failed-tighter` > `failed-succ`
  > `failed-target`;
- a target run that fails on an individual case makes that case's target-dependent quantities unavailable (it does not
  fail the draw); a configuration-level cross-family gate then needs every case's target valid, otherwise it is
  "unavailable (target failed on case …)"; H-E1c uses the target-free margin $2\epsilon_{\rm ref}$ on that case;
- target certification is labelled "checked on the registered subset (dev6 / cases 0–7)", not an every-case proof;
- E2 2D 4096² verdict runs use the registered H-acc-N 4096² population of P1 (16 cases: dev6 + val32|0–9); E1 uses 38;
- 3D archive reads use qualify's saved `wkeys` / case mapping (`q3run.py:44–67`), not the 2D `W` key;
- every staged job carries a generated execution inventory (unique keys: bank, mesh, rule, spec, solver level, case set;
  reuse marked) and a smoke-derived reservation certificate; the 9.9 GPU-h margin of A3.10 is a projection, not certified;
- if the A3.10 emergency reduction of the 3D incumbent evaluation fires, historical agreement is reported only for the
  fixed-configuration quantities and incumbent agreement for selected-rule fronts is "unavailable";
- bad-vector admission requires finite values and a strictly positive target norm at every state (dims `dstats.py:173–176`);
  no predicate may pass through an empty reduction.

---

## Amendment A5 (2026-10-19, during the code audits; before any smoke output exists)

**A5.1 Smokes run while the acceptance code is still being audited.** Code audits 1–3 (`audits/codex-code-{1,2,3}.md`)
found no remaining blocking defect in the code that runs on the cluster (drivers `r2acc.py`, `e1run3d.py`, `e2run3d.py`
over the copied `d3run.py`, `e5time.py`, `train2d/` and `train3d/` scripts, `cluster/stage.py` / `submit.sh`); audit 3
confirmed staging of all four smoke specs and the E5 timing mechanics. The open blocking items concern the offline
analysis / acceptance code (`e1analyze*.py`, `select*.py`, `e2analyze*.py`, `e2verdicts.py`, `e5analyze.py`,
`smoke_accept.py`, `certify.py`). Because the smokes only produce raw outputs, they are submitted now (one at a time,
≤ 2 GPU-h each, brief), and **no smoke output is read for acceptance until the acceptance code passes its own code
audit**. Production stays gated on that audit, on smoke acceptance (ALL = true, every control fired), and on the
coordinator's release. A job-side defect found later in a smoke is fixed and the smoke re-run (a new attempt).

**A5.2 smk1 includes the H200 E5 panel** (`configs/e5smk1.json`, same panels as `e5smk4`, H200), as §6 registered, in
addition to the A100 `smk4` of A2.9.

## Amendment A6 (2026-10-19, after code audit 4; no smoke output read yet)

**A6.1 Smoke resource contract (bootstrap).** The smokes are the measurement source of every production certificate
(§6, A4.2), so a smoke cannot itself carry a smoke-derived certificate. Its contract instead: attempt name `smk*`,
reserved time ≤ 2 GPU-h (stage.py refuses more), a staged execution inventory listing every step (driver steps with
their unique execution keys from `certify.INV`, shell steps with their arguments and declared projection), collected
with the job and validated by `smoke_accept.py`. Production jobs keep the full certificate (code-tree, job-spec, time
limit and GPU count bound).

**A6.2 Smoke populations** are fixed contracts in the analysis code, not read from job metadata: 2D smokes dev6
(tighter and check on dev6); 3D E1 smoke cases 0–7, check 0–7, draws 0–2; E5 smokes dev6|0–4 × 2 reps. Smoke
results are labelled "smoke" and never enter a verdict.

**A6.3 What the smokes demonstrate for the selection → verdict chain.** smk2 runs the registered selection code
(`select2d.py`, smoke contract: R128, candidates fib4181 and gauss96) on the smoke-trained bank, and acceptance feeds its
result into the H-acc-N row construction (the 256² row exists only if smk2u evaluated the selected rule, otherwise the
registered "verdict not reached" path must result). The full chain (selection → generated verdict configs → verdict
jobs) first runs in production on the incumbent (`e2s0`, `e2v0*`), where the non-blocking anchor G-E2.1 and the blocking
replay gate G-E2.3 check it before any seed's verdict is read. smk3 evaluates the 3D rung-selection predicate on the
pilot bank (own-family label of gl24 against gl32).

## Amendment A7 (2026-10-19, before any smoke output)

**A7.1 smk1 moved to an A100-80G as `smk1b`; the H200 E5 panel becomes `smk5`.** smk1 (6070380) waited in the H200
queue behind other lanes' jobs and was cancelled before it started (0 GPU-h). The E1 2D/3D smoke has no H200
requirement (1024² and 65³ fit an A100-80G), so it runs as `smk1b` on A100-80G; the H200 E5 smoke panel of A5.2 runs as
its own smoke `smk5` (≤ 0.5 GPU-h). Smokes stay one at a time. Nothing else changes.

## Amendment A8 (2026-10-19, coordinator production release; recorded before any production job)

**A8.1 Partial production release (coordinator SendMessage, 2026-10-19).** Released: the 2D items — E1 2D (`e1m256`,
`e1m1024`, `e1m4096`), E2 2D seeds (`e2t2d` training, then the per-bank `e2s*` / `e2v*` / `e2t*` jobs, incumbent
included), and E5 on the 2D P1 panel subset (`e5a1`, `e5a2`, `e5h1`, `e5h2`, after the A100 / H200 E5 smokes `smk4` /
`smk5`). Conditions: `smk3` accepted, and each job's Codex job audit says SUBMIT. Submission through the shared campaign
sbatch lock (account cap 8, exit 75 = retry later), lane cap 3, ledger cap 75 GPU-h.
**Held:** every 3D item — E1 3D draws (`e13d`) and E2 3D W1024 seeds (`e2t3d*`, `e2q*`, `e2f*`) — until jcp-dims hands back
its 3D quadrature front and the coordinator releases them. smk3 (3D memory pilot and 3D replay) is a smoke and has
already run; its acceptance is a condition of the 2D release, not a 3D production step.

## Amendment A9 (2026-10-19, after smk3 timed out; before any production job)

**A9.1 The full-size 3D training pilot does not fit a ≤ 2 GPU-h smoke.** smk3 (6074365, H200, 2.09 GPU-h) timed out at
its 2 h limit inside `train2w.py`: data generation at 33³/65³/129³ completed (the 129³ training set alone took 1,687 s,
against 1,373 s for all three meshes in the incumbent job j3b), and the host-side stage after it (snapshot POD before the
bank fit) was still running, with host RSS rising to 176.8 GB (sacct MaxRSS 176,766,044 K; 42 % of the 420G request).
The registered memory gate (host RSS × 1.25 ≤ request; device peak ≤ 120 GB) could therefore not be completed in a
smoke. **Decision:** the pilot is not retried as a smoke; the first released 3D training job (`e2t3d1`, when the
coordinator releases 3D) is itself the memory pilot — it records device peak and host RSS (memwrap + sacct), and the
other two seeds are submitted only after it shows host RSS × 1.25 ≤ 420G and device peak ≤ 120 GB; its time limit is
raised to 4 h (data generation was ~1.4× slower than the incumbent's). The pilot-bank rung check of smk3 moves to that job's
bank (e2q runs). The 3D evaluation part of smk3 (incumbent W1024 gl64/gl80 heavy cell, W256 lattice replay G-E2.3) runs
as its own smoke **smk3c** (≤ 1 GPU-h). For the 2D release (A8) "smk3 accepted" is read as "smk3c accepted"; this is
recorded here and reported to the coordinator.
