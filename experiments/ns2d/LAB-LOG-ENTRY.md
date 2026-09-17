## 2026-09-17

### ns2d — 2D incompressible Navier–Stokes: Phase 1 CERTIFIED (28/28 gates, 52-check independent audit); Phase 2 running

Lane `ns2d` of the nine-lane ICLR campaign, answering reviewers GwrW and 5mgh ("Navier–Stokes,
Euler, … anything nonlinear"). Branch `exp/2026-09-17-ns2d`, worktree
`worktrees/2026-09-17-ns2d`, cluster namespace `/cluster/tufts/paralab/tawal01/ns_20260917/`.
Phased so that whatever phase is reached is a gated, reportable result.

**Formulation (pre-registered, `experiments/ns2d/DESIGN.md`).** Vorticity–streamfunction NS on
the periodic unit torus, $\omega_t = J(\psi,\omega) + \nu\Delta\omega$, $\Delta\psi=-\omega$.
Arakawa Jacobian + 5-point Laplacian + exact FFT Poisson, **implicit midpoint** with
Newton–Krylov (FFT-Helmholtz preconditioner). Implicit midpoint rather than the IMEX the
campaign brief suggested, because an IMEX residual is *linear* in the new state, the
correction block would be analytically eliminable, and the lane would not test the claim it
exists to test. Family: band-limited random $\omega_0$ (12 amplitudes) rescaled to
$U_{\rm rms}=1$, $\nu\sim\log\mathcal U[10^{-3},10^{-2}]$ so $\mathrm{Re}=1/\nu\in[100,1000]$.

**Phase 1 certified — job 3780151 (`ns101`), A100 `pax105`, 47 m 33 s, `jax_backend=gpu`, f64,
matmul `highest`, commit `8a01627b`.** All 28 gates pass. Headline numbers:

- **Taylor–Green against the closed-form FULLY DISCRETE solution** (Arakawa gives
  $J_A(\psi,c\psi)=0$ exactly, so $\omega^n=((1-a)/(1+a))^n\omega_0$): **1.7e-14 / 4.4e-14 /
  9.7e-14** at $N=64/128/256$. Against the semi-discrete form 1.64e-7; against the continuum
  6.34e-4 → 3.95e-5 with orders 2.001, 2.004, each within 0.03–0.41 % of its closed-form
  prediction. Controls fire: wrong-$\lambda$ 3.0e-1, backward Euler 6.2e-4.
- **Manufactured solution with $J\neq0$**: spatial orders 1.997, 1.998; temporal order
  2.000006 from three $\Delta t$ levels; **flipped-$J_A$ control 5.2e-2**.
- **Budgets**: per-step enstrophy/energy identities $\le1.4\times10^{-13}$; at $\nu=0$ the
  drifts are $\le4.4\times10^{-16}$ over 100 steps; backward-Euler control drifts 5.4e-3.
- **Mesh refinement over $64\to128\to256\to512$**: orders 2.009/2.002 at Re 100 and
  1.965/2.004 at Re 1000. **Re 1000 is resolved at every mesh** — the design's worry that 64²
  might be under-resolved did not materialise, so no mesh was dropped.
- **Independent NumPy/SciPy FOM** (own stencils, own FFT, dense-Newton LAPACK solve):
  **9.14e-16** over 100 steps at 64² (JAX 1.45 s vs NumPy 653 s).
- **Dataset**: 512 train + 64 dev + 64 sealed trajectories at 64²/128²/256², SHA256 per
  cohort in `configs/phase1-hashes.json`; worst Newton residual 9.8e-15; max CFL 0.31/0.61/1.23.

**Independent NumPy audit** (`audit_phase1.py`, imports neither the driver nor JAX,
recomputes every reported number from the saved fields): **52 checks, ALL_MATCH = true.**
Archived as bounded Git chunks (`artifacts/ns101`, 336 MB, checksum-verified both sides); the
exact remote attempt directory was then deleted.

**What was wrong and got fixed/retracted.**

1. **The independent NumPy Arakawa had the opposite sign.** Arakawa's eq. 46 is written as
   $J(\zeta,\psi)$; the lane's convention is $J(\psi,z)$. Caught because the analytic-$J$ error
   converged to *exactly 2.0* while every conservation identity read 1e-17 — **conservation
   identities are sign-blind**. Only an analytic-$J$ or MMS check pins the advection sign,
   which is why F-MMS carries the flipped-sign control.
2. **The mesh-order estimator was biased.** Using the finest level as the reference turns a
   pure $Ch^2$ error into an observed order of $\log_2 5 = 2.32$ (the smoke read 2.28/2.21).
   Replaced by successive-level differences. Gate arithmetic retracted, not the scheme.
3. **F-JAC's negative control was an absolute threshold on a mesh-scaling quantity**
   (DESIGN §A2). The centred-difference Jacobian's conservation leak is itself $O(h^2)$, so
   the $10^{-3}$ floor "failed" at $N\ge64$ **for correct code** while the Arakawa identities
   read 2–9e-18. Restated as a ratio: the control must exceed Arakawa's violation by $10^6$;
   measured 1.5e14 / 6.4e13 / 2.4e13. This is the fourth instance in this lane alone of the
   failure mode the project keeps naming, and it was caught by a *passing* system reporting
   FAIL, not by a suspicious number.
4. **The advection tensor was unaffordable as first written** (DESIGN §A3). The direct build
   costs $24nMR^2 = 4.7\times10^{14}$ FLOPs at $n=256^2,M=1152,R=512$ — hours of A100 time,
   which would have forced the top rung $q=256$ to be cut. But the test modes *are* Fourier
   modes, so $\Phi^\top$ is an **FFT**, not a dense $(M,n)$ matmul. Rewritten: same exact
   quantity, ~500× cheaper (2.3 s at $N=256,M=256,R=128$ on the shared GB10). The direct build
   is retained as the independent reference and gate **R-TFFT** compares them (4.37e-16).
   **No rung was cut for cost.**

**Phase 2 in flight**: jobs **3783796** (`ns201`, $K=16,R=256$) and **3783797** (`ns202`,
$K=32,R=512$), A100 `pax049`/`pax106`, one job per attempt directory, both confirmed
`jax_backend=gpu`. Periodic separable bank (integer Fourier features, no boundary mask,
mean-zero columns) + auto-decoder head; gates are the bank floor vs the POD-$R$ truncation
floor on held-out dev trajectories, the head oracle vs the linear POD-$K$ floor, and the
query-time single-start initialiser vs the oracle.

**Open.** Phase 3 (correction ladder $q\in\{0,16,64,256\}$ against POD-LSPG at matched online
dimension and the in-job FOM tolerance ladder) is written, smoke-verified end to end on the
FFT tensor path, and gated by the pre-registered criterion: monotone in $q$ **and** $\ge2\times$
error reduction at the top rung, else FAIL. Stop rule: if Phase 3 cannot start by 2026-09-22,
the deliverable is the certified FOM, the hashed dataset and the Phase-2 floors.

**Not verified by an independent model.** Codex was unavailable all day (quota to 2026-09-19
11:33; its one launch also died in its own sandbox before reading a file), and a Claude
subagent audit was killed by the account-wide quota exhaustion at ~06:07 EDT that also killed
this lane for three hours. In their place, `reports/self-audit-fom-verification.md` is a
**written self-audit**, explicitly labelled as such, listing what it cannot establish: that
`ns2d_fom.arakawa` is the standard Arakawa form rather than merely a conserving,
antisymmetric, second-order bilinear one (the gates test the latter, which is what the tensor
needs), and the calibration of the 2× criterion.
