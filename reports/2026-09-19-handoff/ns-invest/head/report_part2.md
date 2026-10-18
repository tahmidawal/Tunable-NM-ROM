
---

## 2. Evidence from the artifacts

### 2.0 What was run (all CPU/NumPy except one 13-minute GPU regeneration)

| script | input | output | what it computes |
|---|---|---|---|
| `gen64.py` (GPU, one run, 797 s on the GB10 through `jaxrun-slot`, `jax_backend=gpu`, `x64`, `highest`) | the lane's `ns2d_fom.make_fom` at **N=64**, `dt 2e-3`, 500 steps, `ntol 1e-11` — the certified FOM, imported read-only | `data64.npz`: dev cohort (64 traj, seed 20260918) and the **first 282** training trajectories (seed 20260917; time-capped), 26 states each, worst Newton residual 2.5e-15 | held-out and training fields for coverage, POD and regression tests. 64² is used because every floor and oracle in the lane is flat across 64/128/256 to three digits (report tables), and `analyze_b.py` reproduces the archived `oracle_N64.npz` numbers to 1e-13 |
| `common.py` | `runs/ns203, ns204, ns302, ns303/archive/.../output/{ckpt, oracle_N*.npz}` | — | NumPy re-implementation of the periodic bank, the head, its Jacobian, thin-QR whitening |
| `analyze_a.py` → `analyze_a.json` | checkpoints + oracle npz only | code geometry, chart smoothness, Mahalanobis/hull tests, head Jacobian rank, image spectra, 8-start vs 1-start | §2.1, §2.2 |
| `analyze_b.py` → `analyze_b.json` | `data64.npz` + ns203/ns204 checkpoints | sanity vs archived numbers, data coverage, residual structure, parameter-conditioned regression baselines, quadratic/cubic manifold on POD-K coordinates | §2.3–2.6 |
| `analyze_c.py` → `analyze_c.json` | `data64.npz` | quadratic manifold with (log ν, t) inputs; translation-aligned POD-K and quadratic manifold | §2.6, §2.7 |
| `analyze_burgers.py` → `analyze_burgers.json` | `b-seeds/checkpoints/sep_hfit_seed1.pkl` (stage-C head + its 131 072 codes) and `runs/s1/.../sep_coeff_N256_K16_R512.npz` (whitened coefficient targets, `mu_tr`, `t_tr`, 408 test states) | the same statistics on the working Burgers case | the "Burgers" column everywhere below |

Sanity (from `analyze_b.json`): my NumPy bank/head reproduce the archived `e_bank` and `e_oracle` at N=64 to **4.8e-14 / 2.4e-14** (ns203) and **1.8e-13 / 1.3e-14** (ns204) relative, and my POD-16/32 of 282 regenerated trajectories give dev medians 0.239 / 0.141 against the lane's 0.242 / 0.141 from 512.

### 2.1 The latent codes: not collapsed, not extrapolated — but the chart is unstructured

From `analyze_a.json` (training codes `Z_tr` from the checkpoint; oracle codes `Z` from `oracle_N64.npz` / `oracle_N256.npz`):

| statistic | ns203 (K16) | ns204 (K32) | ns303 (K16, 3-mode) | ns302 (K16, 2048 traj) | Burgers seed1 (K16) |
|---|---|---|---|---|---|
| participation ratio of `Z_tr` (of K) | 15.8 | 31.2 | 15.6 | 15.7 | BURGERS_PR |
| `Z_tr` singular values, last/first | 0.80 | 0.76 | 0.77 | 0.79 | BURGERS_SV |
| code step between consecutive snapshots (median) | 0.167 | 0.283 | 0.128 | 0.458 | BURGERS_STEP |
| nearest code of **another** trajectory (median) | 0.758 | 1.04 | 0.528 | 0.478 | BURGERS_NN |
| CV R² of (12 amplitudes, log ν, t) → codes, ridge deg 1 / 2 / 3 | 0.21 / 0.19 / −2.6 | 0.09 / 0.01 / −2.7 | 0.18 / 0.21 / 0.06 | 0.20 / 0.26 / 0.20 | BURGERS_R2 |
| CV R² of codes → parameters, deg 1 / 2 | 0.26 / 0.22 | 0.25 / −0.11 | 0.43 / 0.49 | 0.25 / 0.29 | BURGERS_R2INV |
| oracle codes outside the training box / beyond the 99 % Mahalanobis radius | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % | BURGERS_MAH |
| oracle code → nearest training code, over the training codes' own other-trajectory NN | 0.66 | 0.84 | 0.68 | 0.83 | BURGERS_ONN |
| 8-start LM lands on a different code than 1-start (> 0.1 code norm) | 45 % | 50 % | 23 % | 70 % | — |
| median `e_single / e_oracle` | 1.000 | 1.000 | 1.000 | 1.06 | — |
| head Jacobian (whitened) local dimension at oracle codes, σ > 0.1 σ₁ / > 0.01 σ₁ | 13 / 16 | 23 / 32 | 10 / 16 | 13 / 16 | BURGERS_JAC |
| dims for 90 / 99 % energy of the head image at oracle codes | 11 / 32 | 13 / 39 | 6 / 15 | 12 / 30 | — |

Reading.
- **No auto-decoder collapse**: the codes fill all K directions (participation ratio ≈ K; singular values flat 1.0 → 0.8), the head's Jacobian has full rank 16/32 at every oracle code and 13/23 directions above 10 % of the leading one; ns303's manifold is locally 6–10-dimensional (its family has 7 intrinsic dims + t), ns203's 12–13 (13 + t). The head is using its latent dimensions.
- **No extrapolation in z**: every held-out oracle code sits *inside* the training cloud (Mahalanobis median 2.4–3.8 against a training median of 3.2–4.5; none outside the per-PC box), and its nearest training code is *closer* (0.66–0.84×) than training codes of different trajectories are to each other. The held-out fits land between training trajectories, where the surface is defined only by the MLP's interpolation.
- **Non-identifiability without harm**: in 45–70 % of states the 8-start LM ends at a different code than the single start, at *identical* error (ratio 1.000) — the head is many-to-one on flat valleys, but this does not cost accuracy.
- **The chart is unstructured**: the true parameters explain only 9–26 % of the code variance under any polynomial degree (deg 3 overfits to R² < 0), and the codes explain 25–49 % of the parameters. Nearby parameters do not get nearby codes. Within a trajectory the codes are continuous (step 0.13–0.46 vs 0.5–1.0 to the nearest other trajectory), so the training set is, in latent space, **512 separate smooth curves placed arbitrarily** in a 16-dim ball. The MLP is asked to interpolate between them. (Burgers column: BURGERS_CHART_NOTE.)

### 2.2 The held-out residual: driven by the state's nonlinear content, not by distance to the data

From `analyze_a.json` / `analyze_b.json` (ns203 unless stated):

| statistic | ns203 (K16) | ns204 (K32) |
|---|---|---|
| corr(per-state oracle error, per-state POD-K error) | **0.94** | **0.99** |
| corr(per-state residual norm, distance to the nearest training state at the same time) | 0.06 | 0.06 |
| corr(case log ν, case worst oracle error) | −0.77 | −0.84 |
| CV R² of the whitened residual `c − h(z*)` against (params, t), ridge deg 1 / 2 (8 folds over cases) | −0.10 / −0.36 | −0.08 / −0.30 |
| residual energy inside the top-16 / 64 / 128 directions of the head's *own* training image | 16 / 78 / 95 % | 2 / 51 / 84 % |
| error of the head's **linear skip subspace alone** (K-dim linear, from `h_lin`) | 0.547 | 0.372 |
| best K-dim linear subspace inside the bank span (POD-K in bank coordinates) | 0.239 | 0.141 |
| per-time ratio POD-K / oracle, t = 0, 0.2, …, 1 | 1.63, 1.37, 1.10, 1.08, 1.08, 1.17 | 0.83, 1.15, 1.18, 1.09, 1.12, 1.16 |

Reading.
- The head fails on exactly the states where the linear POD-K fails (r = 0.94–0.99) and at the same rate (ratio 1.08–1.18 on evolved times); low-ν (high-Re) cases are worst for both. The head's residual is **not** a smooth function of the parameters (CV R² < 0 with 64 held-out cases) and is uncorrelated with how far the state is from the training set (r = 0.06), i.e. it is not "far from data → bad"; it is "more non-linear content → bad", *for every state alike*.
- The residual lies mostly (78–95 %) inside the span the head can already produce: the failure is interpolation *within* the head's range, not a missing direction. The correction ladder (ns304) confirms this from the other side — the residual directions `C_q` close the gap only at q = R.
- The head's linear part is poor (0.55 vs the 0.24 optimum): the MLP carries the fit, and at t = 0 — where the family is an exact 12-dim linear subspace — the K=32 head is *worse* than POD-32 (0.83) because a free-code MLP has no reason to reproduce a linear subspace exactly.

### 2.3 Data coverage: the held-out states are as far from every training state as their own norm

From `analyze_b.json` (`coverage`, NS, 282 training trajectories × 26 states; distances in the field 2-norm over ‖ω(0)‖) and `analyze_burgers.json` (`coverage`, Burgers, 131 072 training states in the whitened bank metric over ‖u(0)‖):

| nearest training state (median over held-out states) | NS ns203 family | Burgers (b-seeds seed1) |
|---|---|---|
| any time | **0.586** | **0.086** |
| same time index | 0.644 | 0.138 |
| same time, t = 0 | 0.742 | 0.202 |
| same time, late (NS t = 1; Burgers t ≥ 25) | 0.463 | 0.108 |
| nearest trajectory in **parameter** space, same time | 0.809 | — |
| (for scale) POD-K / head oracle / bank floor | 0.24 / 0.20 / 0.012 | POD-16-in-bank BURGERS_LIN16 / BURGERS_ORACLE / 0.004 |

With 512 instead of 282 trajectories the NS distance shrinks by (282/512)^{1/13} ≈ 5 %; with 2 048 by 11 %. To reach the Burgers coverage (0.09–0.14) on this family one would need (0.6/0.12)^{13} ≈ 10⁹ trajectories. The Burgers head at 72 trajectories (HFIT learning curve: 7.2 % held-out) already beats its linear competitor (≈ 60 %) by 8×; the NS head at 2 048 trajectories (16.7 %) beats POD-16 (24 %) by 1.46×.

### 2.4 Learning curves, NS vs Burgers (both head-only on a frozen bank, whitened space, held-out oracle)

| trajectories | NS ns301 `plain` (K16, 512×3) | NS train-oracle | Burgers HFIT `wide` (K16, 1024×3) | Burgers `refit_ctl` (256×2) |
|---|---|---|---|---|
| 72 | — | — | 0.0721 | 0.0202 |
| 128 / 144 | 0.275 | 0.017 | 0.0414 | 0.0136 |
| 256 / 288 | 0.207 | 0.030 | 0.0239 | 0.0108 |
| 512 / 432–576 | 0.198 | 0.048 | 0.0149 / 0.0138 | 0.0095 |
| 1152 | — | — | 0.0080 | — |
| 2048 / 4096 | 0.167 (ns302, joint) | 0.124 (recon) | 0.0060 | — |
| log-log slope, first two doublings | −0.41, −0.06 | | −0.80, −0.79 | −0.57, −0.34 |

(Burgers numbers from `separable-decoder/HFIT.md`, "Trajectory learning curve"; NS from `artifacts/ns301/result.json` and `ns302`.) The Burgers head-only learning curve is 3–13× steeper per doubling than the NS one; the NS train-oracle rises with n (0.017 → 0.048 → 0.124) while the held-out number barely moves — the head runs out of capacity to memorise before it starts to generalise.

### 2.5 Parameter-conditioned interpolation from the *true* chart is far worse than the free-code head

`analyze_b.py` fits ridge polynomials (deg 1–3, 15/120/680 features) and an RBF kernel-ridge regression from the standardised inputs (a/U_rms, b/U_rms — in which the t = 0 state is exactly linear — log ν, t) to the whitened bank coefficients of all 282 × 26 training states, selects λ/γ on dev cases 32–63, and reports field-space errors on dev cases 0–31 (the ns301 protocol):

| interpolant from (params, t) | dev median (report) | per-time medians t = 0 … 1 |
|---|---|---|
| ridge, degree 1 | 0.553 | 0.48, 0.51, 0.59, 0.64, 0.70, 0.71 |
| ridge, degree 2 (120 features) | 0.467 | 0.41, 0.39, 0.50, 0.53, 0.55, 0.56 |
| ridge, degree 3 (680 features) | 0.499 | 0.41, 0.45, 0.56, 0.57, 0.56, 0.61 |
| RBF kernel ridge (γ = 0.02, 3 666 states) | **0.441** | 0.28, 0.35, 0.49, 0.53, 0.53, 0.57 |
| learning curve, KRR, 64 / 128 / 256 / 282 trajectories | 0.556 / 0.490 / 0.426 / 0.429 | |
| POD-16 (linear, no parameters) | 0.239 | 0.06, 0.28, 0.31, 0.27, 0.23, 0.20 |
| free-code head (ns203 oracle) | 0.203 | 0.03, 0.20, 0.28, 0.26, 0.22, 0.18 |

A smooth interpolant that *knows the parameters* is 2× worse than plain POD-16 and 2.2× worse than the auto-decoder head. The 13-dim parameter → state map is not learnable from 282–512 samples by generic smooth regression; the free-code head does *better* than the true chart because it is free to reuse the near-linear structure of the family. (Burgers control, same script logic on 131 072 states of a 5-dim family: BURGERS_PREG.) This is direct evidence against "the head fails because its chart is unstructured": giving it the perfect chart makes it worse.

### 2.6 Quadratic and cubic manifolds on the POD-K coordinates do not beat POD-K either

`analyze_b.py` / `analyze_c.py`: POD of the 282 × 26 regenerated training states; the residual `u − V_K V_Kᵀu` is fitted by ridge regression on all monomials of the K POD coordinates up to degree 2 (153 features at K=16, 561 at K=32) or 3 (969), optionally with (log ν, t) as extra inputs (190 / 630 features); errors on the 384 held-out states with the linear encoder `z = V_Kᵀu` and, for the K=16 quadratic manifold, with the best-fit `z` (LM, every 4th state):

| manifold | train recon | dev, linear encoder | dev, best-fit z | POD-K | ratio POD-K / best |
|---|---|---|---|---|---|
| K=16 quadratic in z | 0.259 | 0.247 | 0.217 | 0.239 (0.223 same subset) | **1.03** |
| K=16 quadratic in (z, log ν, t) | 0.255 | 0.249 | — | 0.239 | ≤ 1.0 |
| K=16 cubic in z | 0.155 | 0.480 | — | 0.239 | 0.5 (overfits) |
| K=32 quadratic in (z, log ν, t) | 0.114 | 0.173 | — | 0.141 | ≤ 1.0 |

The quadratic term explains essentially nothing of the residual beyond POD-K on held-out states (and, at t = 0, makes it worse: 0.19 vs 0.06). The non-POD-K content of a held-out NS state is **not a low-order function of its dominant coordinates**, even with ν and t supplied: states with similar low-mode content have different harmonic content depending on their history. A quadratic manifold — the natural "exploit the degree-2 residual" design — is predicted to fail the 2× bar on this family.

### 2.7 Translation alignment (symmetry reduction)

ALIGN_SECTION

### 2.8 Summary of the evidence

| candidate cause | for | against | verdict |
|---|---|---|---|
| bank | — | floor 0.4–1.2 % (10–30× below the oracle); residual is inside the head's range | excluded (the lane's finding) |
| optimisation / LM oracle | 3–5 % budget exits | 8-start = 1-start to 1e-9; audit's independent LM beats the driver on ≤ 1/48 states | excluded |
| auto-decoder collapse / ill-conditioned z | 45–70 % non-unique codes | full-rank Jacobian, full-participation codes, oracle codes interior, non-uniqueness costs nothing | excluded |
| unstructured chart (nearby parameters → unrelated codes) | R² 0.1–0.26 | the *true* chart interpolates 2× worse than POD-16 and 2.2× worse than the head | not the binding constraint |
| head capacity / recipe | training fit degrades 2.4 → 6 → 12 % with 128 → 512 → 2048 trajectories; linear skip far from the POD optimum; t = 0 worse than POD-32 | ns301 (wd, early stop, small head) and HFIT (wide/deep/huge) move held-out numbers ≤ 2× on both PDEs; 8-dim family (ns303) keeps the ratio at 1.4 with training fit 1.4 % | secondary (worth ≈ 1.2× at most) |
| **sampling density per intrinsic dimension × weak nonlinearity of the family** | NN coverage 0.59 vs 0.09 (Burgers); residual ∝ non-POD-K content (r 0.94–0.99), uncorrelated with distance to data (0.06); every interpolant tested (parametric, quadratic, cubic, KRR, MLP) lands at 1.0–1.2× POD-K; ratio unchanged by 4× data and by an 8-dim family | none found | **binding** |
