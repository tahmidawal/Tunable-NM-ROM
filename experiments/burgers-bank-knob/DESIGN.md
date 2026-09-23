# DESIGN — burgers-bank-knob: the nested bank truncation $R'$ as the tunability knob on 2D Burgers

Written before the first cluster job (2026-09-23). Lane `exp/2026-09-23-burgers-bank-knob`, forked from
`exp/2026-09-22-burgers-repanel` @ `219feb6e`. Cluster namespace `/cluster/tufts/paralab/tawal01/bbank_20260923/`,
one directory per attempt, never reused. At most 2 lane jobs at once. **Local commits only; never pushed.**

## 1. Question

The sibling lane `exp/2026-09-23-poisson-bank-knob` found on 2D Poisson that truncating a once-rotated bank to its
first $R'$ columns is a better accuracy/cost knob than the correction rank $q$. On Poisson the linear rung is
closed-form. On Burgers the advection is nonlinear, so every truncated arm still needs a Levenberg–Marquardt solve
per step and an empirical-quadrature (EQ) rule for the advection. This lane measures what $R'$ does on Burgers at
$256^2 \dots 4096^2$ with the frozen checkpoint and **no retraining**, and whether $R'$ alone (no $q$) reaches the
current accurate error (0.51–0.87 % on dev6).

## 2. The rotation (offline, training data only) — `make_rotation.py`

$G$ = frozen bank on the training mesh ($256^2$), $G = Q_G R_G$. Sample rows $a_i = R_G h_\theta(z_i) / \lVert R_G
h_\theta(z_i)\rVert$ for **every stored training code** $z_i$ (`ck['Z_tr']`, 131072 codes). SVD
$A = U S V_s^\top$; $T = R_G^{-1} V_s$, $L = V_s^\top R_G = T^{-1}$. The rotated bank $G T = Q_G V_s$ is
orthonormal at $256^2$ and ordered by training energy. This is the Poisson lane's construction
(`pbk_core.make_rotation`) with one stated difference: only head coefficients enter (Burgers training fields are not
stored with the checkpoint), not training-field projections. Committed as `inputs/rotation_R512.npz`
(sha256 `51149166…`); every job verifies the hash. Diagnostics (`inputs/rotation_R512.json`): $\lVert LT - I\rVert =
8.2\times10^{-12}$; cumulative training energy 0.99594 / 0.99936 / 0.99996 / 1.0000 at $R' = 32/64/128/256$.

A truncated model uses $a = L_{[:R']} c$ for its coefficient vector $c$: head and corrections become
$L_{[:R']} h_\theta(z)$ (folded exactly into the head's last linear layer and linear skip) and
$C' = L_{[:R']} C_q$; the bank is the first $R'$ rotated columns (held as nested column blocks, so an $R'$ arm
reads only $R'$ columns). At $R' = R$ this is the parent model up to round-off (**parity gate**, §5).

## 3. Arms (identical at every mesh; `make_configs.py`)

$R' \in \{512, 384, 256, 128, 64, 32\}$, $K = 16$:

| family | unknowns | $M$ (tests) | rule | solver variant, tolerance |
|---|---|---|---|---|
| (a) head only, $q=0$ | 16 | 64 | `q0scaled` (the paper's certified $q=0$ rule, $m=1024$) | the paper's FAST variant: LU, clip, damping carry-over, quadratic predictor, gtol $10^{-3}$ |
| (b) linear rung, head dropped | $R'$ | $4R'$ | `lat64` ($63\times63$ lattice, $m=3969$) | the paper's ACCURATE variant at the mesh: Cholesky, clip, carry-over, predictor; $256^2/512^2$: first step exact (`x1`), gtol $10^{-2}$; $\ge 1024^2$: gtol $10^{-3}$ |
| (c) head + corrections, $q=\min(256, R'-16)$ | $16+q$ | $4(16+q)$ | `lat64` | ACCURATE variant (as (b)) |

The linear rung solves $R'$ unknowns by the same fused LM (all unknowns damped and trust-clipped; trust radius =
0.01 × the spread of the training coefficient vectors $L_{[:R']}h_\theta(z_i)$, the same definition the head arms
use on codes), same weak residual, same predictor/carry-over/stopping rule; its initial fit is the closed-form
least-squares solution of the same Gauss-point state fit (what the LM initial fit converges to for a linear map).

Also, at $L \le 2048$ (both banks fit): the **unrotated parent** at the paper's fast and accurate settings (parity
twins of (a) and (c) at $R'=512$); at every mesh one **certificate control**: (c) at $R'=512$ with the `bad0` rule
(regressed in b-panel), untimed — it must fail the $\rho$ bar.

Newton–BiCGStab grid: the 15 settings of the repanel lane (7 audited `b_panel` + 8 `lean`), all in the same
allocation.

## 4. Quadrature-rule certification, re-measured per arm at each mesh

The rules' nodes and weights do not depend on $R'$, but the tested advection does, so every EQ arm's rule is
re-certified **on the states that arm itself reaches** on a held-out population at the target mesh: the
burgers-eqcert populations (`params_draw(20260922, 56)` at $256^2$, `params_draw(20260921, 56)` elsewhere;
5 certification draws × 8 trajectories + 1 confirmation draw × 16), disjoint from dev6, hold64 and training.
$\rho = \lVert P_q^\top a_{\rm EQ} - \Phi^\top a\rVert / \lVert\Phi^\top a\rVert$ on every reached state
($k \ge j$ for an `x_j` arm), exact side by the full-grid separable projection. **Never the NNLS fit residual.**
Bar 0.116 (the eqcert bar). An arm's rule is **confirmed** iff $\rho_{\max} \le 0.116$ on all 5 certification
draws AND the confirmation draw. Deviation from burgers-eqcert, stated: eqcert's certificate ALSO required the
bar on a dense-query ("held-out population") state set; re-running the dense residual for each of ~19 truncated
models at up to $4096^2$ is out of budget, so this lane certifies on deployed (reached) states only.

## 5. Gates

| gate | bar |
|---|---|
| backend / x64 / highest | log `jax_backend=gpu`; asserted |
| cohort | dev6 sha256 `108f12dc…` |
| rotation file | sha256 equals the committed file |
| rotated operator = parent operator × T | $\le 10^{-10}$ relative ($L\le 2048$) |
| Φ-free operator vs explicit Φ | $\le 10^{-12}$ ($L \le 512$) |
| **parity at $R'=R$** | rotated (a)/(c) at $R'=512$ reproduce the unrotated parent's fields to $\le 10^{-10}$ relative on every case with identical per-step iteration counts and stop reasons ($L \le 2048$; at $4096^2$ the parent does not fit beside the rotated bank — stated, not gated) |
| certificate control | the `bad0` arm is NOT confirmed |
| repetitions | ≥ 5 retained per (arm, case); every timed field SHA256 identical to the quick run |
| **order effect** (fixed before any timing) | the panel runs in two phases: arms whose quick run took < 1 s, then the rest; a cool-down `min(2 s, previous)` precedes any invocation that follows one of ≥ 0.5 s, then the 0.25 s burn. Gate, per phase: each invocation normalised by its (arm, case) median; "long neighbour" = predecessor arm's median ≥ 4× this arm's median. Pooled over all arms: median normalised time after long / after short − 1 ≤ 3 %; per arm with ≥ 6 samples in both groups: ≤ 5 %. The repanel definition (≥ 1 s predecessor, unnormalised) is reported as a diagnostic |
| independent NumPy audit | `audit_bankknob.py` recomputes every error from saved fields (restricted grid for all, full grid for the audit case), recomputes $\rho$ for spot states from the checkpoint and the rotation in NumPy, recomputes the certificate statuses, and must detect two injected controls (a swapped case, a perturbed error) |

## 6. Pre-registered setting rule (fixed now, not changed after results)

Per mesh, over the (a)/(b)/(c) arms (parents and controls excluded; the rotated $R'=512$ arms stand for them):

- **accurate** = the arm with the smallest worst-case evolved error whose rule is **confirmed** (§4). If a more
  accurate unconfirmed arm exists it is reported beside, marked.
- **fast** = the cheapest arm (median GPU ms) whose worst evolved error ≤ the error of the current paper fast
  setting ($q=0$, $R=512$: arm (a) at $R'=512$) at that mesh, among arms whose rule is confirmed. A cheaper
  unconfirmed arm meeting the error bar is reported beside, marked.
- **Table-1 speedup** = GPU ms of the fastest tested full-order setting whose worst evolved error ≤ the accurate
  arm's, divided by the accurate arm's GPU ms, same job (one FOM per row). Every arm is also reported against
  the fastest FOM at least as accurate as itself.
- Error = worst over the 6 dev cases of max over the five evolved output times of
  $\lVert u - u_{\rm ref}\rVert / \lVert u_0 \rVert$, reference `fft_tight` at the same mesh.
- The chosen accurate and fast settings at $4096^2$ are frozen and committed, then evaluated on hold64
  (`params_draw(20260916, 64)`) at $4096^2$ together with the paper's current fast/accurate settings
  ($R'=512$ (a), (c)) and the FOM grid, in one allocation.
- **Can $q$ be dropped?** Yes on a mesh iff some (a) or (b) arm with a confirmed rule reaches worst evolved error
  ≤ the (c) $R'=512$ arm's error at that mesh (the current accurate setting, 0.51–0.87 % on dev6).

## 7. Jobs

| attempt | mesh | GPU | notes |
|---|---|---|---|
| `bk256`, `bk512` | $256^2$, $512^2$ | A100 | x1 accurate variant |
| `bk1024` | $1024^2$ | A100-80G | |
| `bk2048` | $2048^2$ | H200 (or A100-80G) | parent kept (2 × 16 GiB) |
| `bk4096` | $4096^2$ | H200, 240G | rotated bank only (64 GiB) |
| `bkh64` | $4096^2$ hold64 | H200, 240G | chosen settings only, after the dev6 selection is committed |
