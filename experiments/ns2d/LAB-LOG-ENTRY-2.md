## 2026-09-17

### ns2d — 2D incompressible Navier–Stokes (INTERIM): Phase 1 CERTIFIED (28/28 gates, 52-check audit); Phase-2 attempt ns201 FAILED its gates (rank-capped bank); reruns ns203/ns204 queued; Phase 3 not started

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

**Phase 2, attempt `ns201` — job 3783796 ($K=16,R=256$), A100 `pax049`, 1 h 32 m, exit 0, commit
`dddb41fb`: FAILED 8 of 17 gates.** Collected, checksum-verified, audited (`audit_phase2.py`,
67 checks), archived (`artifacts/ns201`), remote deleted. The numbers:

- **B-ORTH FAIL at every mesh: bank rank 128 of $R=256$**, $\kappa(R_b)\approx1.6$–$2.0\times10^{15}$.
  Structural: the bank MLP's last layer is linear over `g_hidden`$=128$ units, so
  $\operatorname{rank}G\le\min(R,\texttt{g\_hidden})$. The parent lane paid for exactly this in
  August (`sep_burgers_r3.py`: "`G_HIDDEN >= 2R`"); this lane's decoder inherited the old
  default 128. The audit recomputes rank 128 from the pickled weights at 64²/128²/256².
- **H-ORACLE FAIL at every mesh**: oracle median 0.2031 / 0.2018 / 0.2028 vs POD-16 median
  0.2417 / 0.2404 / 0.2402 (ratio 1.19, bar 2.0); bank floor median 0.038; oracle worst 0.53.
  Training reconstruction rel-$L_2$ mean 0.209 at 30 000 full-batch steps (rel-MSE
  5.8e-2 → 4.8e-2 from 10k to 30k). B-FLOOR passes (bank/POD-256 ratio 1.82–1.85 ≤ 2);
  H-SOLVED passes (single-start 0.213 vs oracle 0.203).
- **B-DATA FAIL at 256² by hash (train and dev), PASS at 64²/128²**; `ns202` on `pax106`
  matched all Phase-1 hashes with the same commit. The 256² cohort hash is node-dependent at
  the bit level — the campaign's "value gates, not hash gates, across machines" landmine.

**What was wrong and got retracted / fixed (DESIGN §A4, commit `beffbb1b`).**

1. **The bank was rank-capped by construction** (above). Fix: `G_HIDDEN = 2R` default and a
   new gate **B-RANKCAP** asserted before any GPU time; both Phase-2 heads (`ns201`, and
   `ns202` still running at $R=512$) are rank-128 banks and neither feeds Phase 3.
2. **The reported oracle numbers of `ns201` are contaminated at ≤ 1.1e-2 relative** (64²;
   7.6e-3 at 128², 3.0e-3 at 256²): the driver evaluated $\sqrt{r_n^2+\|u-Pu\|^2}/n_0$ with
   $r_n=\|R_b(h(z)-c)\|$, $c=R_b^{-1}Q_b^\top u$, and through the singular $R_b$ the coefficient
   $c$ carries $\sim10^{12}$ garbage that cancels catastrophically. Found by the audit's exact
   recomputation $\|Gh(z)-u\|/n_0$ on the 48 archived states (the rank-128 projection agrees
   with the exact norm to all digits). No verdict changes (0.20 vs the 0.12 needed). Fix: the
   drivers now report field-space errors directly and store the formula value beside them
   (`formula_vs_field_worst_rel`; 1.3e-16 on a full-rank smoke). The six mismatching audit
   checks stay recorded as MISMATCH for `ns201`.
3. **B-DATA/R-DATA were hash-only gates across nodes.** Now hash-or-value: on a hash mismatch
   the dev cohort must agree with Phase 1's archived first-8 trajectories at the six
   evaluation times to ≤ 1e-8 relative (`configs/dev8_eval_ref.npz`, 32 MB, cut from job
   3780151's `dev8_N*.npz`); the train cohort inherits the dev verdict of the same node,
   mismatch recorded. Dev is generated first so the node is certified before train is gated.
4. **The previous session left the phase-3 driver with an uncommitted, broken reordering**
   (`w0s`/`nus` used before definition in the moved FOM block; a `jax.jit` inside a
   768-iteration loop in the decomposition; a 250 MB npz rewritten after every invocation).
   Fixed: FOM tolerance ladder first (same-job controls survive a walltime cut), ROM fields
   written per case, decomposition through `oracle_fit` (one jit per rung) and evaluated in
   field space. Re-smoked end to end (`runs/SMOKES.md`): R-TB 0, R-TFFT 6.8e-16, R-TQ 6.8e-16,
   R-LIN 1.2e-15, `audit_phase3.py` ALL_MATCH on 54 checks.

**Submitted (Phase-2 reruns, the DESIGN's pre-registered "rerun after a failed gate"; not an
advance to Phase 3).** `ns203` = job **3787319** ($K=16,R=256$, `G_HIDDEN=512`, head 512×3,
100 000 steps, a100, 10 h) and `ns204` = job **3787320** ($K=32,R=512$, `G_HIDDEN=1024`, same
recipe, a100, 14 h), commit `beffbb1b`, one job per attempt directory, `squeue` checked before
and after, both PENDING (Priority). Recipe justified by the parent lane's `HFIT.md` (longer
training ≈ 2×, wider head small, latent Fourier features harmful — not adopted). Expectation
stated before they run (§A4): full rank by construction; H-ORACLE is the open question, and a
second failure with a full-rank bank at 100k steps is reported as the negative finding, not
answered by lowering the bar. Jobs used: 5 of 8 (ns101, ns201, ns202, ns203, ns204).

**Still running.** `ns202` = job 3783797 ($K=32,R=512$, rank-capped): to be collected and gated
when it lands; B-ORTH fails by construction, its $K=32$ oracle vs POD-32 is still informative.

**Open.** Gate `ns203`/`ns204` on landing (B-RANKCAP, B-ORTH rank $=R$, B-FLOOR, H-ORACLE
ratio ≥ 2, H-SOLVED, hash-or-value B-DATA); if a head passes, stage Phase 3 on that checkpoint
immediately (driver ready: ladder $q\in\{0,16,64,256\}$, POD-LSPG at $k'\in\{16,64,256\}$ and at
$k'=K+q$, FOM ladder first, three timed reps, three error metrics, three-layer decomposition).
Stop rule 2026-09-22 stands. Codex quota-blocked to 2026-09-19 11:33: written self-audits
`reports/self-audit-fom-verification.md` and `reports/self-audit-phase2.md` substitute (§A5).
Report: `experiments/ns2d/reports/2026-09-17-ns2d.md` + `summary.json` (163 rows, interim).
