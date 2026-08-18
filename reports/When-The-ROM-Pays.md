---
marp: true
paginate: true
---

# When does the reduced model pay?

Poisson and Burgers · 2-D · f64 · A100

Three questions:

1. What does one solve cost, and what sets that cost?
2. At a given accuracy, are we cheaper than a classical solver?
3. Can we hand our answer to a classical solver and finish faster?

---

## The method in one slide

Instead of solving on the grid, solve for a handful of numbers on a learned surface, then decode.

Work shrinks at every stage:

| stage | size |
|---|---|
| grid points | 900 – 260,100 |
| quadrature points we actually touch | **256** |
| equations we solve | **64** |
| unknowns | **8** |

*The grid never enters the solve. That is the whole idea.*

---

## 1. Cost does not depend on grid size

Iterations to reach the stopping tolerance, k = 8:

| grid | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| iterations | 10.1 | 10.4 | 10.3 | 10.3 | 9.4 |

A 256-fold change in unknowns. The cost does not move.

*This is the central claim, and it holds.*

---

## How few points are enough?

Poisson, 64×64:

| points | share of grid | error | cost |
|---|---|---|---|
| 64 | 2% | 5.48e-2 | 11.1 ms |
| 256 | 7% | 8.66e-3 | 15.1 ms |
| **512** | **13%** | **7.68e-3** | **15.2 ms** |
| every point | 100% | 7.65e-3 | 35.6 ms |

13% of the points, same answer, 43% of the cost.

Burgers is stronger: 13% of the points, 15% of the cost.

---

## The same 256 points, finer and finer grids

| grid | points in grid | our 256 is | error |
|---|---|---|---|
| 32×32 | 900 | 28% | 1.08e-2 |
| 64×64 | 3,844 | 7% | 8.48e-3 |
| 128×128 | 15,876 | 1.6% | 8.40e-3 |
| 256×256 | 64,516 | 0.4% | 8.38e-3 |
| 512×512 | 260,100 | **0.1%** | 8.61e-3 |

*The number of points the problem needs is set by the problem, not by the grid.*

---

## Where the time goes now

| grid | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| spent decoding the answer | 4% | 6% | 11% | 50% | **84%** |

We made the solve grid-free and it worked — so the solve is no longer the bottleneck.

At 512×512 the solve is 9% of the time. Writing the answer onto every grid point is the rest.

*Good problem to have. Now the binding one.*

---

## 2. Speed against accuracy

Poisson, cheapest route to ~1% error:

| grid | ours | classical solver | |
|---|---|---|---|
| 64×64 | 3.4 ms | 2.2 ms | we are slower |
| 256×256 | 4.8 ms | 7.9 ms | **1.6× faster** |
| 512×512 | 8.8 ms | 26.5 ms | **3.0× faster** |

Our cost is flat. Theirs is not. **The advantage only exists on fine grids.**

---

## One model, two settings

No retraining — just stop the solver earlier or later.

| grid | loose | | tight | |
|---|---|---|---|---|
| 64×64 | 1.8 ms | 8.4e-2 | 3.4 ms | 1.16e-2 |
| 256×256 | 3.2 ms | 8.3e-2 | 4.8 ms | 1.13e-2 |
| 512×512 | 7.3 ms | 9.3e-2 | 8.8 ms | 1.18e-2 |

At 512×512: **20% more time buys 8× the accuracy.**

The trade gets better as the grid refines.

---

## Where POD sits

POD is the standard reduced method.

- Cheaper than us when 5% error is fine — 0.37 ms against our 3 ms
- **Cannot reach 1% at any number of modes.** It saturates near 5%

So we do not really compete. POD owns the rough end; below ~5% we are the only reduced method left, and the comparison becomes ours against the classical solver.

---

## Burgers: we lose

| grid | ours | classical solver | |
|---|---|---|---|
| 64×64 | 106 ms | 29 ms | 0.27× |
| 256×256 | 290 ms | 159 ms | 0.55× |

The gap narrows as the grid refines. It does not close in the range we ran.

---

## 3. Warm-starting the classical solver

Produce a rough answer, hand it over, let the classical solver finish. Same final accuracy — only cost is in question.

| grid | classical alone | with our warm start | |
|---|---|---|---|
| 64×64 | 4.8 ms | 6.8 ms | 0.71× |
| 256×256 | 19.9 ms | 22.6 ms | 0.88× |
| 512×512 | 61.7 ms | 65.8 ms | 0.94× |

**It never pays.** On Burgers, 0.15× to 0.49×.

---

## Why it fails, and it differs by problem

**Poisson** — our answer looks accurate but is a poor starting guess. Conjugate gradients measures error in a different norm, and in that norm we are five times further off than the 1% suggests.

**Burgers** — the guess is genuinely good, cutting Newton iterations at every grid size. But producing it costs 273 ms against a 48–228 ms classical solve.

Extrapolating from the previous two time steps beats our warm start and costs nothing.

---

## One result we withdrew

Our error looked catastrophic at certain latent sizes — 6× to 12× worse than the model can represent.

It was an average over 16 cases dominated by 1–5 diverging solves. The **median** sits at the model's own limit at every latent size.

Cause: the optimiser took wild first steps and accepted any improvement, wandering outside the region it was trained on. A step-size limit removes it.

*Reported because the same optimiser is used everywhere, so our accuracy numbers are likely pessimistic across the board.*

---

## What to take away

**Holds.** Cost independent of grid size, from the quadrature up. Accuracy at 1% with 8 numbers, where POD cannot go below 5%.

**Does not hold.** We are not faster on coarse grids, we are not faster on Burgers at all, and warm-starting a classical solver does not pay.

**Next.** Decoding is now 84% of the cost. The optimiser needs a step limit. And on 2-D Poisson a direct solver beats everything, which we should say before a reviewer does.
