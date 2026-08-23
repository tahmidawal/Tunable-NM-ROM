# Separable-decoder first cell (2026-08-22)

Design + audited spec: `reports/2026-08-22-separable-eq-decoder-design.md` on main.

**Model** (pure neural, NO POD anywhere): u(x;z) = bc(x) * ( g(x)^T h(z) );
g = Fourier-feature MLP (2->128->128->r), h = MLP head (k->128->128->r) + linear
skip — nonlinear in z. Auto-decoder training (joint Adam over g, h, per-state
codes), f64, warmup-cosine, fixed data-scale constant folded into the features.

**Cells** (one cluster job, sequential): Poisson-2D and Burgers-2D at N=64,
(K,R) in {(8,32),(16,64)}, M=4K, m=4M=16K NNLS-EQ grid nodes, 40k train steps,
seed 0. Incumbent solvers only: ctol_tol.lm_tau_poisson (trust 1.0x radius) and
blat_common lm_step_jit/rollout_jit (trust 0.01x radius).

**Arms**: `meshfree` (network evaluated in-loop, incumbent path) vs `cached`
(feature banks G_q / stencil G_st cached once; no spatial network in the
compiled iteration). GATE 0 asserts the two arms' weak residual/Jacobian agree
<= 1e-12 relative — the discretization (incl. the FOM sign-upwind stencil) is
never changed, only how it is evaluated. Error/counters come from one untimed
invocation and cost from 7 timed repetitions of the same jitted call (see
AUDIT-2026-08-23.md — not literally the same invocation; Burgers timing excludes
the online IC fit); FOM CG ladder / reference Newton rollout timed in the same job.

**Smokes** (local GB10, N=16): Poisson gate0 dev 0.0, solve error = oracle
error (manifold-limited); Burgers gate0 dev 1.6e-15, 50/50 steps, no blowups.

**Job**: Slurm 2802238, /cluster/tufts/paralab/tawal01/sepdec/sepdec_r1,
stage manifest-checked both directions.

---

# N=128 scaling round (2026-08-23, this branch)

Contract: `HANDOFF.md` (binding measurement rules after the N=64 timing FAIL in
`AUDIT-2026-08-23.md`). Namespace `/cluster/tufts/paralab/tawal01/sepdec_n128/`,
budget <= 4 cluster jobs, branch `exp/2026-08-23-sepdec-n128`.

**Measurement implementation (differences vs the N=64 cell)**

- Poisson: the timed pipe is grid-source -> online modal projection (verified
  == `pc.weak_source_term` to <1e-12) -> incumbent trust-LM -> full-grid decode.
  Errors/counters are read from the LAST timed repetition; max deviation
  between first and last timed reps is stored (`dev_first_last_max`). All raw
  reps stored per (method, source, rep). Balanced ordering: the unit list
  (2 arms x taus + the full CG ladder) runs REPS times, order reversed on odd
  reps. Stop-reason distributions per row. Two cohorts: `held_seed0`
  (seed-0 indices 512+) and `fresh_seed1` (an untouched fresh draw).
- Burgers: END-TO-END timed path = IC latent fit (batched jitted LM on the
  data misfit to the known u0; inits = nearest training t=0 codes selected via
  the offline QR of the readout bank + mean code) -> incumbent `rollout_jit`
  -> full-grid decode of all 51 states. Split (ic, roll, dec) + raw reps per
  trajectory. Errors computed from the timed invocation's decoded fields;
  deviation vs an untimed incumbent `bc.rollout` recorded (~1e-10 scale).
  Baselines in-job on the same GPU: tolerance-terminated Newton ladder
  (same residual/BiCGStab/NaN-guard as the truth generator, Newton loop stops
  at ||R|| <= ntol*||u_prev||) = the STRONG baseline; and the fixed-8-Newton
  truth rollout, labelled over-solved, never the headline comparator.
- GATE 0 unchanged and asserted in every cell.

**Jobs**

- j1 = Slurm 2825804 (A100, pax105): broad sweep, K in {8,16,24,32}, R=6K,
  M=4K, m=16K, 60k steps, n_ff=128; Poisson N_TEST=16 x 2 cohorts, REPS=7,
  taus {1e-3,1e-2}; Burgers N_TEST=8, REPS=7, ntols {3e-2,1e-2,1e-3,1e-5}.
  (A first submission 2825787 died in 2 s on a bash `local k=$1 r=$((6*k))`
  expansion-order bug — nothing ran, nothing to pull.)

Results land in `runs/sepdec_n128_j*/`; tables are generated only by
`summarize_n128.py`.
