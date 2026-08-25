# HANDOFF — push Burgers accuracy as far as it will go, without losing the speed

Written 2026-08-25. This worktree is **disposable**: if the campaign fails, the branch
`exp/2026-08-25-burgers-accuracy` is deleted and nothing of value is lost. Everything
established up to now lives on `exp/2026-08-23-n256-push` and in `LAB-LOG.md` on `main`.

## The one-sentence mission

Burgers 2D: get the **rollout** (real, no-truth) relative-L2 error as close to **1e-3** as
possible, while keeping the round-4 speed (N=1024 single-query 1.61x vs a properly swept
classical ladder). Accuracy is the only open half; do not trade the speed away silently.

## Where things stand (verify against the JSONs, do not trust this file blindly)

Read first: `LAB-LOG.md` on main (2026-08-25 entry), the report
`reports/2026-08-24-separable-decoder-architecture-and-results.md`, and on this branch
`PROFILE.md`, `CROSS-RESOLUTION.md`, `PUSH-PLAN.md`, `runs/SUMMARY-R3.md`.

The error ladder at N=1024 (K=16, R=512, G_HIDDEN=1024):

```
span least-squares floor   2.60e-4     <- the basis is FINE, below target
K=16 oracle                8.24e-3     <- h reaches only 1/32 of its own span   <-- THE PROBLEM
rollout (the ROM error)   ~6e-3 (M=256 EQ) .. 2.6e-2 (coarse M=64 EQ)
```

- **Everything except h is tight.** Solver output ~= weak-EQ optimum ~= oracle; oracle ->
  rollout is only ~1.3x; no error compounding over 50 steps.
- **The h gap is resolution-independent**: 36.8x at N=256, 31.7x at N=1024. So it is h's
  capacity/optimization, not the grid, discretization, or quadrature. **Iterate at N=256
  (cheap, ~1.5 h/run) and confirm the winner at N=1024.**
- **K is not the constraint**: the family is intrinsically ~6-D (5 params + time).
- **The rank cap is fixed**: `G_HIDDEN >= 2R` gives rank 512/512. `G_HIDDEN = R` reaches
  full rank but with a badly conditioned tail (2e-6) — do not use it.

## The four levers, in the order the evidence ranks them

1. **h itself.** It is a small MLP (128->128) plus a linear skip. Widen and deepen it;
   try alternative parameterisations of the latent->coefficient map. Online cost is
   **dispatch-bound, not compute-bound** (PROFILE.md), so more capacity in h is close to
   free at solve time — verify that claim by timing, do not assume it.
2. **The latent codes Z.** They are free variables trained jointly with the weights. If
   they have not converged, every snapshot is fitted toward a wrong target and h is being
   asked to interpolate noise. Nobody has checked this. Try code-only refinement passes,
   per-code restarts, and report a converged-code diagnostic (e.g. residual of a
   code-only re-solve on training snapshots).
3. **Early-time loss weighting.** The error is concentrated where the field is sharp:
   oracle 3.0e-2 at t=0 decaying to ~5.5e-3 by t=20; mean over t<=5 is 1.83e-2 vs 6.09e-3
   after. Trajectory error is the mean over 51 states, so the early phase dominates the
   headline. Weight the loss accordingly, and consider whether `ff_scale` should be
   retuned for the sharper early states.
4. **K = 24 / 32.** Nearly free per the cost model. Six arms (2837061-66, on the
   n256-push branch) were testing h-capacity and K when this worktree was cut — **read
   their results before re-running the same thing.**

## Hard blocker for any 1e-3 claim: the quadrature

The control EQ set (M=64, m=256) has a **relative fit error of 6.08e-3** with worst row
~9.8e2, and a single-step weak-opt/oracle ratio of 1.119 mean / 4.02 max. You cannot claim
1e-3 through a quadrature rule that is itself 6e-3 wrong. M=256/m=1024 reaches 5.13e-4 but
still has ~9.8e2 tails. Needed: **tail-capped NNLS** (bound per-row error, not just the
global fit) and larger m. This costs online time — spend it, the solve is dispatch-bound.
Whatever you fix, report the ROM error **for the EQ set you actually used**, and never
compare a coarse-EQ speed number against a fine-EQ accuracy number.

## Non-negotiables

- **Gate 0 <= 1e-12** on every arm, after every architecture change.
- Incumbent discretization, residual and Jacobian definitions untouched. Solver internals
  (damping, stall rule, warm start, batching, fusion) may change.
- **PURE NEURAL** — no POD in any model. SVD only as a diagnostic.
- **No test-truth in any solve path.** Span floors and oracles use truth and must stay
  labelled diagnostics.
- Timing: end-to-end incl. IC fit and full decode, raw reps retained, balanced ordering,
  errors from the timed invocation, censoring/stop-reasons reported.
- **Matched-accuracy comparisons only.** Sweep the classical (newton_tol, lin_tol) ladder
  and compare against the cheapest rung at least as accurate as the ROM. A 5.2x claim was
  already retracted for getting this wrong — `lin_tol`, not `newton_tol`, was carrying it.
- Batched columns: a `vmap` of a `lax.while_loop` runs until every lane finishes, so
  batched numbers penalise the classical solver more than the ROM. Report batched ratios
  as **upper bounds** until someone implements a fair batched classical solver.
- Summaries **generated by a committed script**; never hand-type a number into a report.
- Pull + sha256 + commit + push results, then **delete the remote job dir**.

## Mechanics

- Cluster namespace: `/cluster/tufts/paralab/tawal01/burgacc/` — one subdir per job.
  Do **not** write into `n256_push/`; another agent owns it.
- Write only in this worktree. Reading the sibling `2026-08-23-n256-push` tree is expected
  and encouraged (its runs/ has every checkpoint and result you inherit).
- Cluster rules from `CLAUDE.md` are mandatory: tufts-login, gpu partition only, the
  paralab venv, `jax_backend=gpu` preflight exit-42, `JAX_DEFAULT_MATMUL_PRECISION=highest`,
  output under paralab never home, regenerate data from seed, squeue before AND after
  every submit, one job per directory.
- Local: `source /etc/profile.d/jax-mem.sh`; `jaxrun /home/tahmid/Dev/.venv/bin/python`;
  never bare python; at most one local jaxrun at a time.
- **Watchers:** one background until-loop per submission wave
  (`while ssh -o BatchMode=yes tufts-login 'squeue -h -j <ids> -o %T' | grep -q .; do sleep 180; done; echo done`),
  verified alive with `pgrep` before ending any turn. Never end a turn with unfinished
  work and no live watcher. Several previous agents stranded themselves this way.
- **Do not close a jit over a large array.** A ~2 GB captured array costs +10 GB host RSS
  and 16 s of compile per jit; it OOM-killed an N=1024 job at 245 GB RSS. Pass banks and
  data as explicit arguments.

## What "done" looks like

Report, per configuration: the full ladder (recon / span floor / oracle / rollout, on
fresh trajectories), the per-timestep error curve, the EQ set used and its own fit error,
and the matched-accuracy Pareto (single-query and batched) against the swept classical
ladder — with jacs/step and gate 0 beside every timing. Name the binding rung each round.
A well-supported negative ("h capacity does not close the gap; here is what does bind") is
a real result and must not be dressed up. Append to `LAB-LOG.md` on main before finishing.
