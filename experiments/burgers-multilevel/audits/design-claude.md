# Design review: fixed-target Burgers multilevel pilot

**Verdict:** The design is honest about scope and its controls are good. It has one structural problem that caps any possible gain, and one design error in nested iteration. It deserves a bounded pilot only after a cheap pre-gate (Stage 0 below) that can kill it without spending timed GPU budget. As written, the most likely outcome is a well-documented negative result whose cause you could predict in advance.

I reviewed the design text only. I did not check it against `p1core.py` or the archived target code. Items marked *verify* depend on that code.

## Fundamental problems

### 1. The coarse level isn't cheap, so the speedup has a hard ceiling

N, M, m and quadrature are fixed, so every coarse residual evaluation costs exactly what a fine one costs. The only savings come from:
- **Jacobian:** JP has 128 columns instead of 256. That is at most 2× fewer flops, and only if it is computed as 128 JVPs, not as the full J then sliced.
- **Linear solve:** 128 instead of 256. This is negligible; the normal matrix is tiny next to the M×m×r Jacobian work.

Classical multilevel gains come from coarse work being orders of magnitude cheaper. Here a coarse iteration costs about 0.5–1.0 of a fine iteration. At these sizes the GPU is likely latency- or launch-bound, so it may be close to 1.0. The best possible outcome is a modest constant factor, and that only if coarse iterations replace fine ones one-for-one, which nothing guarantees.

**Fix:** measure the ratio (coarse iteration wall time) / (fine iteration wall time) and write down the implied speedup bound before the pilot. If the bound is below about 1.3×, the pilot cannot show anything worth the overhead.

### 2. The two-level correction mostly duplicates the fine LM step

The fine LM step minimises a damped Gauss-Newton model over all of ℝ²⁵⁶, and that space contains range(P). The coarse damping, λ·diag(PᵀJᵀJP), is just the leading diagonal block of the fine damping, with the same trust radius. So:
- In the near-linear regime, right after a fine pre-step the gradient is already small. The coarse correction is then about zero and only costs time.
- It adds anything only through nonlinearity across its 4 re-linearised coarse iterations, or when fine damping is heavy.

What remains is block-coordinate LM: alternating 256-dimensional and 128-dimensional steps. With no smoothing property and a prefix that isn't a frequency hierarchy, nothing suggests the slowly converging error components lie in range(P). The design admits this, but it is the central question and the pilot doesn't test it directly.

**Fix (cheap, decisive):** on archived tight histories, compute:
- the fraction of ‖x_k − x*‖ lying in range(P) after fine LM iterations;
- the principal angles between range(P) and the span of the smallest right singular vectors of J.

If the slow error isn't concentrated in the prefix, the two-level arm is dead on arrival.

### 3. Nested iteration zero-padding throws away the predictor's tail

The warm predictor already supplies all 256 coefficients from full fine histories. Solving min_z f(Pz) and zero-padding discards coefficients 129–256. That is probably worse than the predictor alone, so the safeguard will often revert and you pay for the coarse solve anyway. Classical nested iteration zero-pads because the coarse solution is the only prior; that isn't the case here.

**Fix:** freeze the tail at the predictor and solve min_e f(x_pred + Pe), which is the two-level correction with no pre-step. If you keep zero-padding, report it as a deliberately weak variant.

### 4. Missing control: is the *prefix* special?

Any gain could come from "cheaper partial-Jacobian steps", not from the ordered bank. Add:
- a random 128-dimensional subspace and the trailing 128 columns as coarse spaces;
- a fine-LM baseline with cheap-Jacobian strategies, such as Jacobian reuse/chord over k iterations or a Broyden update.

Without these, a positive result can't be attributed to the hierarchy, and the strong-baseline claim is incomplete.

### 5. The target and its anchor (*verify*)

- **Is S part of the archived target?** M=1024 > r=256, so this is overdetermined least squares, and the minimiser depends on the weight S. If S = diag((a₀+Δt b₀νλᵢ)⁻¹) is new to this study rather than the archived objective, you have changed the target. Then the "fine anchor" no longer matches the archived target. It also needs to be clear what basis λᵢ diagonalises in. R must be expressed in the Laplacian eigenbasis for a diagonal S to make sense.
- **Floating-point precision.** γ=1e-7/1e-8 anchors and the ≤1e-6 tight/tighter gate need float64 throughout, including quadrature accumulation. State this.
- **Anchor solver.** If the anchors use normal equations, cond(JᵀJ) = cond(J)² may stop the anchor from reaching 1e-8. Use QR/lstsq for the anchors.

### 6. Tolerance-to-error gating

A stationarity tolerance γ does not fix the distance to the solution when J is ill-conditioned. The thresholds 2.5e-4 and 2.5e-6 need either justification from the measured cond(J) or an explicit statement that they are empirical. The direct baseline must pass the same gate first; otherwise "target missed" is uninformative.

For fairness, report **work-to-accuracy curves** (cost vs. anchor distance), not just cost at a fixed γ. Different arms can stop at the same γ at different distances from the solution.

## Smaller points

- **Notation:** R is used for both the residual R(c) and the restriction R_c. Rename the restriction.
- **Coarse Jacobian:** confirm that JP really costs about half of J, using a microbenchmark in Stage 0. If autodiff builds J and slices it, the arm has no mechanism at all.
- **dev6, three repetitions:** that is too few to aggregate. Report per case, with maxima, as you plan; don't quote mean speedups.
- **3D extension:** a negative result explained by the cost bound or the subspace-alignment diagnostic doesn't need 3D confirmation. Restrict 3D to a positive 2D result.
- **dt range:** if Δt varies within a run, S changes per step. Note that the coarse/fine cost ratio and conditioning should be reported per Δt.

## Recommended structure

| Stage | Cost | Kill criterion |
|---|---|---|
| **0. Pre-gate** | ≤10 GPU-min, untimed or microbenchmarks only | Coarse/fine cost ratio > 0.7, **or** the slow-error fraction in range(P) is no better than in a random 128-dim subspace |
| **1. Pilot (as written, with fixes 3–6)** | ≤1 h | As in the design: stop if overhead offsets gains or the target is missed |
| **2. 3D** | Remainder of cap | Only after a positive 2D result |

With the zero-padding fix, the prefix-vs-random control and Stage 0 added, the pilot is worth one bounded slot. Even a negative result would be cheap and clearly explained. Without them, a negative result can't be told apart from a poorly chosen design.
