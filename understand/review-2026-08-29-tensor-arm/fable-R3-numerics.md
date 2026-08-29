# R3 — numerical and statistical review of the 1D Burgers tensor-vs-oracle claim

Reviewer: adversarial, numerics/statistics. Scope: report `reports/2026-08-29-b1d-tensor-sample-free-burgers.md`, `TENSOR-NOTES.md`, run/audit JSONs, code on `exp/2026-08-29-b1d-tensor`. All checks CPU-only from the committed checkpoints (`runs/b1dqf/b1ds_n{128,1024}/out/*.pkl`); scripts beside this file (`json_stats.py`, `cpu_checks.py`, `field_diff.py`).

**Verdict: numerics sound with caveats.** The central measurement (tensor arm reproduces the oracle arm's rollouts) is correct and I reproduced it independently on CPU, in a *stronger* metric than the report uses. The explanation of the residual mismatch is wrong in its order-of-magnitude reasoning ("second order"), one "identical solver decisions" statement is contradicted by the JSON, the tensor-vs-NNLS percentages are not distinguishable from zero, and f32 would not have supported the parity claim.

## 1. The exact discrepancy — first order in the undershoot, not second (major, wording)

With `U = pad(u,1)`, the code's stencil is `ux = (c-U[x-1])/dx if c>0 else (U[x+1]-c)/dx`, applied identically at the walls (ghost zeros on both sides enter both branches through the same padded array). Subtracting the tensor's fixed backward branch:

$$N_{\text{upwind}} - N_{D^-} = \mathbb 1[u\le 0]\;u\,\big(u_x^{+}-u_x^{-}\big) = \mathbb 1[u\le 0]\;u\,\Delta x\,\Delta_h u ,$$

so $q_T - q_{\text{or}} = -\Delta x\,\Phi^\top\!\big[\mathbb 1[u\le0]\,u\,\Delta_h u\big]$ — the formula in the brief is right, including the wall treatment. The `c>0` vs `c>=0` branch is immaterial: at $u=0$ exactly the prefactor $u$ vanishes. Verified on all 8192 decoded training states: $|(q_T-q_{\text{or}}) - \text{pred}|_{\max} = 2.6\times10^{-13}$ at $N{=}1024$ ($|q|_{\max}=73$), $8.0\times10^{-14}$ at $N{=}128$.

The order question turns on what $\Delta_h u$ is at the undershoot points. If the undershoot were an isolated bump of depth $\delta$ and width $W$, $\Delta_h u \sim \delta/W^2$ and the term would be $O(\delta^2)$. It is not: the undershoots sit at the feet of the blob, where the Laplacian is the blob's own curvature. Measured at $u\le0$ points, median $|\Delta_h u| = 1.9$ ($N{=}1024$) / $1.6$ ($N{=}128$), versus the bump-curvature model $\delta/W^2 \approx 0.1$ — twenty times smaller. The empirical log-log slope of the relative mismatch against undershoot depth $\delta$ is **1.3–1.4** on training states and **1.14–1.28** over the LM-candidate rows in the audit JSONs (corr 0.95), i.e. essentially linear in $\delta$. The correct statement is: *first order in $\delta$ times $\Delta x\,|\Delta_h u|$* — small because $\Delta x$ is small (the forward/backward difference gap), not because $\delta$ enters squared. Consequences: the mismatch scales like $\Delta x$ (measured medians $1.8\times10^{-6}\to2.3\times10^{-7}$ from $N{=}128\to1024$, exactly $\propto 1/N$), and it would *not* shrink quadratically if the decoder undershot less. A rigorous per-state bound $|q_T-q_{\text{or}}|_i \le \Delta x\,\delta \sum_{u\le0}|\Phi_{xi}||\Delta_h u_x|$ holds with the measured mismatch at a median 0.4 of the bound (max bound $2.3\times10^{-4}$ at $N{=}1024$, $1.4\times10^{-3}$ at $N{=}128$, bracketing the audit's max $1\times10^{-4}$ / $4\times10^{-4}$). "Second-order small" should be replaced by this bound.

## 2. Error propagation and the stopping rule (major caveat, not blocking)

Facts from the JSONs: the oracle's final residual norm per step is $2\times10^{-3}$–$8\times10^{-3}$ while `tol_abs` $=10^{-9}\,\bar u\sqrt{n_i}$ — `rn/tol` median $6$–$8\times10^{5}$, minimum $10^{5}$. The residual is 32 equations in 8 unknowns; **the tolerance is unreachable by construction** and reason 1 never fires. "Stall" (reason 2, relative decrease $<10^{-3}$ after an accepted step) is therefore the de-facto convergence rule, and ~99 % reason-2 is not a solver pathology — it is what least-squares convergence looks like under this labelling. Checking that it *is* convergence: at the last accepted state of each step, the stationarity $|J^\top r|/(\|J\|\|r\|)$ has median $10^{-5}$–$6\times10^{-5}$ (converged), but 2–5 % of steps stop with stationarity $>10^{-2}$ (premature stall on a slow step). Recommend relabelling reason 2 as "LS-converged / stalled" and reporting the stationarity split; `STEP_TOL` is decorative and should be removed or set to the LS floor.

Is the agreement masked by the stopping rule? Partly, and it is worth saying so. Worst-case propagation is large: $\|r\|/\sigma_{\min}(J) \approx 42$ (median) at training states, so a $10^{-4}$ relative residual perturbation *in the worst direction* would move the LS minimiser by $4\times10^{-3}$ in latent — 25 % of the trust radius (0.017), 1 % of $|z|$. The observed latent deviations ($2\times10^{-4}$ at $N{=}128$, $5\times10^{-6}$ at $N\ge512$) are 20–1000× smaller, so the actual perturbation lies mostly outside $J$'s sensitive directions; that is a measured fact about this decoder, not a structural guarantee.

I re-ran both arms' *reference* rollouts (`make_device_ref`, f64, CPU) and measured the quantity the report should have reported — the direct relative $L^2$ difference between the two arms' decoded fields, per step:

| N | traj | field diff max | field diff mean | err-metric diff (report's number) | latent dev | njac equal |
|---|---|---|---|---|---|---|
| 128 | 1 | 8.7e-6 | 3.4e-6 | 1.2e-6 | 1.3e-4 | yes |
| 128 | 7 | 7.4e-6 | 1.4e-6 | 9.9e-8 | 2.1e-4 | **no** |
| 128 | others | ≤8.0e-7 | | ≤1.6e-7 | | yes |
| 1024 | worst (6) | 4.9e-7 | 3.9e-7 | 2.3e-7 | 5.6e-6 | yes |
| 1024 | others | ≤2.0e-7 | | ≤4.1e-8 | | yes |

The report's per-trajectory numbers reproduce on CPU to all printed digits. The direct field difference is 7–75× larger than the error-metric difference (traj 7: $7.4\times10^{-6}$ vs $9.9\times10^{-8}$) because differences of errors cancel; the claim survives, at $\le 9\times10^{-6}$ field-relative, but the report should state this number, not the metric difference. The JSONs do not store $Z$, so it cannot be recovered from them — add per-step `||u_T-u_or||/||u_or||` to the comparison block.

**Contradiction with prose:** `comparison.tensor.per_traj[7].njac_equal = False` at $N{=}128$ (mean accepted steps 5.72 vs 5.74 — one extra accepted LM step somewhere in the tensor arm), reproduced on CPU. The stop-reason *histogram* is identical because it is a coarse statistic (398/2) that almost any run of this solver produces. The glossary sentence "identical histograms mean the solver took the same decisions" is false for that trajectory and should be reworded; pass criterion (ii) is weak evidence on its own.

## 3. Precision: is f64 needed? (minor; f64 required for the parity claim, not for the ROM)

Built $T$ in f32 and contracted in f32/f64 on all training states ($N{=}1024$; $N{=}128$ similar):

| variant | q rel err median | p95 | max |
|---|---|---|---|
| f32 build + f32 contract | 8.1e-7 | 1.4e-6 | 7.2e-6 |
| f64 build, f32 contract | 6.8e-7 | 1.2e-6 | 3.5e-6 |
| f32 build, f64 contract | 4.1e-7 | 8.9e-7 | 6.6e-6 |

Cancellation costs ~2.5 digits (f32 $\epsilon = 6\times10^{-8}$ → $10^{-6}$ typical, $7\times10^{-6}$ worst), consistent with the median contraction ratio $\sim 600$. The $6\times10^{8}$ maximum is a normalisation artefact (per-entry $|q_i|$ near zero for high modes) and never bites: the audit's `vs_qmax_max` field says $\le120$. So f64 is safe with ~13 digits to spare, and the notes' "at most ~3 digits" is right. But note: in f32 the tensor's roundoff ($10^{-6}$) would *exceed* the physical undershoot discrepancy at $N{=}1024$ ($5.8\times10^{-8}$ median) — the parity claim as stated is an f64 claim; an f32 deployment would be a different (still adequate, decoder floor $5\times10^{-3}$) claim.

## 4. Tensor vs NNLS-32: indistinguishable from zero (major, wording)

Per-trajectory paired tests (8 trajectories per N; the same 8 recur across N, so N-pooling is not independent):

| N | tensor wins | exact sign-test p | mean-error ratio | paired bootstrap 95 % |
|---|---|---|---|---|
| 128 | 6/8 | 0.29 | 0.914 | [0.76, 0.99] |
| 256 | 5/8 | 0.73 | 0.989 | [0.97, 1.01] |
| 512 | 7/8 | 0.07 | 0.939 | [0.86, 0.98] |
| 1024 | 4/8 | 1.00 | 0.986 | [0.93, 1.04] |

Per-trajectory ratios span 0.47–1.08; the $N{=}128$ and $512$ means are pulled by one trajectory (traj 2: NNLS-32 5.4e-3 vs tensor 2.5e-3). Averaged over N, tensor beats NNLS-32 on 5/8 trajectories. Against learned-32 every interval straddles 1. "1–9 % better than NNLS-32" is a point estimate with no support; the report's own caveat ("at seed-noise scale") is correct and should replace the headline figure with "not distinguishable from NNLS-32/learned-32 at n = 8".

## 5. The $10^{-5}$ pass bar (minor)

It was pre-registered (LAB-LOG 08-29 proposal: "oracle error to ≤1e-5") before the jobs ran, so not post hoc — but it was never derived. A principled bar is the bound of §1 propagated through the solver: at $N{=}128$ the max residual mismatch $4\times10^{-4}$ times $\|r\|/\sigma_{\min}(J)\approx 40$–$110$ allows latent deviations up to $\sim 10^{-2}$; the bar should instead be stated on the *direct field difference* relative to two references: (a) the solver's own reproducibility floor (ref-vs-fast parity $\sim10^{-7}$ field, JSON `lat_dev_max` $10^{-7}$–$10^{-9}$), and (b) the decoder floor ($5\times10^{-3}$). I would set: field diff $\le 10^{-4}$ (100× below the error, 1000× above solver noise), and report it per trajectory. The measured $9\times10^{-6}$ passes comfortably; $10^{-5}$ on the error-metric difference happens to pass but is the wrong observable.

## 6. Prose vs JSON (nits, one minor)

- Generator reads the right fields; every table number I recomputed matches (`err_fast`, `err_abs_diff_max`, `TQ`, `TL_candidates`, `TC_contraction`).
- **Minor:** "residual mismatch median $4\times10^{-7}$–$8\times10^{-6}$ at those candidates" — the medians are over *all* candidates including the 32–48 % that are exactly positive (mismatch $10^{-14}$). Restricted to candidates with a $u\le0$ point, the medians are $5\times10^{-6}$–$2.5\times10^{-5}$, p95 $6\times10^{-5}$–$2\times10^{-4}$ (from `TL_candidates.rows`). Maxes are unaffected.
- The E1 table's "TJ grad cos min = 1.000000" prints `min(unpert, pert)` rounded to 6 dp; fine, but the perturbed-state J mismatch of $1.2\times10^{-3}$ (min $u=-0.3$) is the number a reader needs for "what if the solver leaves the trust region" and is in the table only as a column pair.
- `njac_equal=False` for $N{=}128$ traj 7 is in the JSON but not in any table or sentence (see §2).

## Severity summary

| # | finding | severity |
|---|---|---|
| 1 | Mismatch is first order in $\delta$ (slope 1.1–1.4), coefficient $\Delta x|\Delta_h u|$ with $|\Delta_h u|$ = blob curvature, not undershoot curvature; "second order" is wrong | major (wording/mechanism; numbers unaffected) |
| 2a | Tolerance unreachable ($\|r\|/\text{tol}\sim10^{6}$); "stall" = LS convergence 95 %, premature 2–5 %; relabel and drop `STEP_TOL` | minor (solver hygiene) |
| 2b | Report compares differences of errors; direct field difference is 7–75× larger (still $\le9\times10^{-6}$). Add it to the JSON and the table | major caveat |
| 2c | "Identical solver decisions" contradicted by `njac_equal=False` (N=128 traj 7) | minor (prose) |
| 3 | f64 safe (~3 digits lost of 16); f32 would put tensor roundoff ($10^{-6}$) above the undershoot term | minor |
| 4 | 1–9 % vs NNLS-32 is not distinguishable from zero (sign-test p ≥ 0.07, bootstrap CIs touch 1) | major (wording) |
| 5 | Bar pre-registered but underived; should be on field difference, ~$10^{-4}$ | minor |
| 6 | Candidate-mismatch medians diluted by exact candidates | nit |

Nothing blocking. The measurement stands — reproduced on CPU at all printed digits and strengthened to a direct field-difference bound of $9\times10^{-6}$ — but three sentences in the report (second-order, identical decisions, 1–9 % better) should not survive revision.
