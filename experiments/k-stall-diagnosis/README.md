# Why some latent solves diverge — diagnosis and fix

Run 2026-08-17/18, locally on the GB10, f64. This cell explains and fixes a defect that
made the reported error at some latent dimensions look 6–12× worse than the decoder is
capable of. **It is a diagnostic cell, not a production measurement**: everything here uses
the full-grid weak objective with NNLS-EQ switched off, so the numbers are internally
consistent but are not drop-in replacements for anything in `cost-to-tolerance/`.

## What was wrong with the earlier reading

The published error at k = 6, 12, 24, 32 sat far above the decoder ceiling, while k = 4, 8, 16
sat at it. That was read as "the solver fails at particular latent dimensions". It is wrong.

The published number is a **mean over 16 test cases**, and at those k it is dominated by 1–5
diverging solves. Recomputed from the archived per-source arrays:

| k | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| ceiling | 1.55e-2 | 8.84e-3 | 7.04e-3 | 6.24e-3 | 4.13e-3 | 3.98e-3 | 3.28e-3 |
| mean | 1.74e-2 | 5.85e-2 | 8.48e-3 | 4.79e-2 | 6.54e-3 | 1.49e-2 | 4.02e-2 |
| median | 1.55e-2 | 8.52e-3 | 7.25e-3 | 7.58e-3 | 5.05e-3 | 4.93e-3 | 3.66e-3 |
| diverged | 0/16 | 2/16 | 0/16 | 3/16 | 0/16 | 1/16 | 5/16 |

The median is within 0.96–1.24× of the ceiling at every k. Running all **64** held-out sources
rather than 16 shows k = 8 and k = 16 failing too (3 and 4 cases at N=32). The 16-source panel
drew none of them. There is no powers-of-two structure; the failure rate rises smoothly with k.

## Root cause

The latent Levenberg–Marquardt has **no globalisation**:

- `lam0 = 1e-6`, i.e. an essentially undamped Gauss–Newton first step, measured at 10–350× the
  norm of the latent vector itself;
- the acceptance test is `v_new < val` — any decrease at all, with no sufficient-decrease or
  gain-ratio condition.

When an overshoot happens to lower the residual slightly it is accepted, the iterate leaves the
region of latent space the decoder was trained on, and it converges to a spurious stationary
point out there.

**Discriminator, exact on all 9 traced cases.** Norm of the iterate after the first accepted
step, relative to `R_train = max_i ||Z_i − z_mean||`: succeeds at 0.77–1.32×, fails at
1.73–4.60×. Clean separation at ≈1.5.

## What it is not

- **Not the objective.** A 1-D slice from `z0` to the ceiling latent is monotonically decreasing
  with its minimum at the ceiling in all 9 cases, failures included. Every successful solve ends
  *below* the objective at the oracle latent (0.7–0.9×); every failed one ends 27–90× *above*.
- **Not hyper-reduction.** Reproduced with NNLS-EQ entirely absent.
- **Not latent scaling.** Per-dimension std of `Z_tr` has max/min ratio 1.12–1.36 at every k,
  with no anomaly at the suspect values.
- **Not conditioning at the optimum.** `cond(J)` grows smoothly with k (2.3 → 4.4e3), full rank
  everywhere, no spike. Conditioning at `z0` sets the *scale* of the overshoot and explains why
  the failure rate rises with k, but not which individual solves fail.
- **Not initialisation.** `z0` is the same mean latent for every source at a given k.

## The fix

Trust region tied to the training latent cloud: reject any step with `||dz|| > Delta`,
`Delta = max_i ||Z_i − z_mean||`, treated as a failed step so `lam *= 10`. Everything else
unchanged — same decoders, same objective, same tau, same budget, same `z0`, same sources.

ROM/ceiling at N=64:

| k | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| base | 1.11 | 7.61 | 1.10 | 7.85 | 1.26 | 3.40 | 15.49 |
| trust region | 1.09 | 1.05 | 1.09 | 1.18 | 1.28 | 1.45 | 1.50 |

No arm regresses and iterations fall at large k (57 → 46 at k=32). Confirmed at N=256 for
k=6 and k=12 (7.80× → 1.05×, 7.97× → 1.17×, 3/16 blown → 0/16).

Raising `lam0` from 1e-6 to 1.0 is numerically identical here and is a one-character change, but
it is not scale-invariant and has no principled magnitude. A Nielsen gain-ratio LM is faster
(15–40 iterations) but **less robust** — still 1/16 blown at k=12 and k=32. **Use the explicit
trust region.**

## Where to apply it

- `cost-to-tolerance/ctol_tol.py` — `lm_tau_poisson`, `lm_tau_generic`
- `poisson2d-rom-objective/followup/fu_eq.py` — `make_lm_jit`
- `poisson2d-rom-objective/pro_common.py` — `lm_generic`
- `ms_autodecoder.lm_solve` (upstream)

The last two are inherited by Heat, Burgers and Wave, **so the project's coordinate-ROM accuracy
numbers are likely broadly pessimistic** by an amount nobody has yet quantified. Some earlier
"stalls above its ceiling" results may be this same artefact.

## Open

- 3 of 64 cases at k=4 survive every fix and never leave the trust region — inferred to be a
  representation limit at k=4 rather than a solver defect, but per-source ceilings were not
  computed on the 64-source set, so that is an inference.
- At k=32 the *median* escape ratio is already 1.64, i.e. the typical solve leaves the training
  ball even when it succeeds. Whether the surviving accuracy there is luck or genuine
  extrapolation is not established.
- The fix has **not** been measured through the EQ + timing path.

## Files

```
kcommon.py     shared loaders and the weak-form objective
klm2.py        the four LM variants (base, trust region, lam0=1.0, Nielsen gain-ratio)
exp1.py        start-from-the-ceiling-latent arm (killed early; hypothesis settled another way)
exp2.py/.json  per-iteration traces and the z0 -> ceiling objective slices
exp3.py/3b.py  the fix tables (exp3_N64.json, exp3_N256.json)
exp5.py/.json  the 64-source lottery test at N=32
ceil_N*.json   per-source ceilings
```
