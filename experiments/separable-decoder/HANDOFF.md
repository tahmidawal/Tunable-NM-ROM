# HANDOFF — scaling the separable EQ-decoder to N ∈ {128, 256, 512, 1024}

Written 2026-08-23 for the four per-resolution sub-agent sessions. You are one of four
agents, each owning exactly one resolution. This file is the contract: what is verified,
what your mission is, what you may tune, and what you must not touch.

## 1. Verified starting state

You are branched from `exp/2026-08-22-separable-decoder` (commit `dacb176` + this handoff).
That branch contains a working first cell at N=64, run as cluster job 2802238 (H200,
`sepdec_r1`), results committed under `experiments/separable-decoder/runs/sepdec_r1/`.
An independent Codex adversarial audit (`AUDIT-2026-08-23.md`, in this directory —
READ IT) verified on the committed checkpoints: the decoder is genuinely nonlinear in z
(superposition fails 30-70%, Jacobians vary with z, changing tangent spaces), no POD/SVD
anywhere, no test-truth leakage (mutation test: corrupting truth changed only the metric,
not the solution), gate 0 independently recomputed at 0.0 / 1.8e-15.

The SAME audit returned **Timing: FAIL** on the N=64 measurement methodology. The 3.3x
Burgers number is NOT a supported speedup. Your round must not repeat these defects —
see the MANDATORY MEASUREMENT RULES below.
The full design rationale and the gate ladder are in
`reports/2026-08-22-separable-eq-decoder-design.md` on main. The pipeline mechanics are in
`experiments/separable-decoder/README.md`.

Reference numbers to beat/compare (from `runs/sepdec_r1/out/*.json` — regenerate any table
from those JSONs, never hand-type):

- Burgers N=64 K=16: cached rollout median ~54 ms vs same-job FOM Newton rollout ~176 ms;
  per-trajectory rel-L2 1.4–2.6e-2 with ONE bad trajectory (IC fit failed at 0.36 → 1.5e-1).
- Poisson N=64 K=16: ~2 ms solve, err 3.75e-2 ≈ oracle 4.43e-2 (decoder capacity-limited,
  only ~21 s of training). FOM CG at tol 1e-1 was still faster at N=64 — the whole point
  of your job is to find where that flips.

## 2. Your mission (per resolution N)

Answer two questions, with numbers on committed branches:

1. **Accuracy scaling:** best achievable test error at your N (both PDEs), and what it
   cost (training time, K, R).
2. **Speed scaling:** ROM online cost vs the same-job classical baseline at your N. The
   ROM's per-iteration cost is O(K³), N-independent; the FOM's grows with N. Find the
   crossover and report honestly which side your N is on.

Run BOTH PDEs (Poisson stationary solve, Burgers 50-step rollout) at your N. Use
hyperparameter optimization — several cluster jobs, sequential refinement — to get the best
accuracy AND the best speed. You have a budget of **up to 4 cluster jobs** (you may re-use
a job dir for a resubmit only after the previous job finished and was pulled).

### HPO search space (what you may tune)

- `K` (latent dim): 8, 16, 24, 32 — f(K) scaling is itself a deliverable, so ≥2 values.
- `R` (feature rank): 4K to 8K (N=64 used R=4K; larger N likely wants more).
- Fourier features: `n_ff` (64→128/256), `ff_scale` (4.0; higher N may want higher —
  finer structure needs higher frequencies, but too high hurts trainability).
- `g_hidden`/`h_hidden` (128→256), layers (2→3).
- `STEPS` (40k→100–200k) and `lr` (1e-3, with the warmup-cosine schedule). At N=64
  training was 20–47 s; you can afford much longer — training cost is offline.
- `M = 4K` test modes, `m = 16K` EQ points is the verified recipe; you may probe
  `m` up (32K) if EQ residuals degrade, but record gate 0 for every variant.
- Burgers IC fit: more/better inits (the N=64 run had 1/8 trajectories fail IC fit from
  mean+t0 inits). More restarts, or a small IC-encoder trained offline, are fair game.
- `N_TEST`: keep ≥16 Poisson, ≥4 Burgers trajectories; more is better for the report.

### What is NOT tunable (non-negotiables)

- **Gate 0 in every job**: cached arm vs incumbent-operator meshfree arm agreement
  ≤1e-12 relative, asserted before any result is recorded. If you change the stencil,
  the weak form, Δt, ν, the mode set, or the tolerance rule, you are no longer solving
  the same problem — don't.
- **Incumbent solvers only**: `ctol_tol.lm_tau_poisson`, `blat_common.make_weak_ops` /
  `_finish_ops` / `rollout` / `fit_eq_weights`, `pro_common`, `ms_parametric`. No new
  solver code; the decoder and its cached banks are the only new ingredients.
- **No test-truth leakage**: test truth is for error metrics and the clearly-labelled
  representation oracle only. Never in the solve path, never in IC-fit inits beyond u0
  (u0 is known data — using u0 at t=0 is legitimate; using u(t>0) is not).
- **MANDATORY MEASUREMENT RULES** (each fixes an audit FAIL from N=64; a job that
  violates one produces unreportable numbers):
  1. **End-to-end Burgers timing**: the timed online cost MUST include the IC latent fit
     (u0 -> z0), not just the latent stepping. Report the split (IC-fit ms, rollout ms,
     total ms) so both views exist. Optimize the IC fit if it dominates — that is real
     online cost, not overhead to hide.
  2. **Symmetric outputs**: the timed ROM path must decode full-grid fields (G_all
     readout included) if the timed baseline returns full fields. Time what a user gets.
  3. **Retain raw timing repetitions** (all reps, per trajectory/source) in the JSON —
     never store only medians. Use a balanced order (alternate ROM/baseline, AB/BA).
  4. **Same-invocation cost and error, or prove equivalence in-job**: either extract
     error from a timed invocation, or record in the JSON the max deviation between the
     timed call's outputs and the error-bearing call's outputs (N=64 audit measured
     3e-10 — do this check in every job and store it).
  5. **Strong classical baselines, same job, same GPU**: Poisson CG ladder; Burgers BOTH
     the truth-generating Newton rollout AND an optimized classical stepper (tolerance-
     terminated Newton or the incumbent cubic-history solver). Label the truth-generator
     as over-solved; never headline a ratio against it.
  6. **Report censoring honestly**: at N=64 all Burgers steps stopped on 'stalled' and
     Poisson tau=1e-3 was 100% censored. State stop-reason distributions next to every
     error number.
- Cluster rules (from CLAUDE.md, all mandatory): `gpu` partition only, submit from
  `tufts-login`, venv `/cluster/tufts/paralab/tawal01/ae-research/venv`,
  `jax_backend=gpu` preflight exit-42, `JAX_DEFAULT_MATMUL_PRECISION=highest`, output
  under paralab never home, regenerate data from seed in-job, one job per directory,
  `squeue` before AND after submit, pull + sha256 + delete remote dir when done.
- **Your cluster namespace is `/cluster/tufts/paralab/tawal01/sepdec_n<N>/`** — never
  write outside it. Four agents share the account; a collision corrupts both.
- **Your worktree is your only write target.** Read siblings freely; write only your own.
- Memory sizing: N=1024 Burgers data and N≥512 FOM baselines are big — check the
  hard-won-lessons section of CLAUDE.md (H200 + `--mem 240G` territory; never close jit
  over the data array).

### Job template

Start from `experiments/separable-decoder/cluster/` (stage + `run.sbatch` from
`sepdec_r1`): adapt paths to your namespace, your N, and your hyperparameter cells.
`sep_poisson.py` / `sep_burgers.py` read everything from env vars — a job is a list of
env-var lines. Keep each job ≤4 h; split cells across jobs rather than serializing.

## 3. Required reporting (before your session ends)

1. All run JSONs, checkpoints, and logs pulled, checksummed, committed to YOUR branch,
   remote dir deleted.
2. A summary table generated by script from the JSONs (commit the script): per (PDE, K, R):
   train time, recon error, oracle error, solve/rollout error, ROM median ms, baseline
   median ms, gate-0 deviation, jacobian count.
3. Append a session entry to the canonical lab log
   (`/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md` — resolves to main
   from inside your worktree): what ran, job ids, namespace, numbers, failures and
   retractions, open items. This is a closing step, not optional.
4. State every caveat next to its number, ranked. A slower-than-baseline honest result
   is a fine deliverable; an unflagged flattering asymmetry is not.

## 4. Known open items you inherit (fix if cheap, else record)

- IC-fit failure mode (Burgers, 1/8 trajectories at N=64): more inits or better restart
  strategy.
- Poisson decoder is capacity-limited at 40k steps — longer training should close the
  err≈oracle gap and push the oracle floor down.
- The N=64 Burgers baseline was the truth-generating fixed-Newton rollout, not an
  optimized classical solver — this round must add a stronger classical baseline arm.
- Poisson used a same-seed held-out cohort (indices 512+ of the seed-0 draw). Add a
  fresh-seed test cohort arm for confirmation at your N (cheap: regenerate with a new
  seed in-job).
- EQ worst-row errors were large at N=64 (311 Poisson K=16 / 92 Burgers K=16) even with
  small global fit — record per-row EQ diagnostics and watch whether they grow with N.
