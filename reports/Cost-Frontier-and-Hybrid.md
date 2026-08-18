## Cost against latent dimension

![cost against latent dimension](talk_figs/6_cost_vs_k_poisson.png)

Iterations needed to reach the stopping tolerance, at k=8:

| grid       | 32   | 64   | 128  | 256  | 512 |
| ---------- | ---- | ---- | ---- | ---- | --- |
| iterations | 10.1 | 10.4 | 10.3 | 10.3 | 9.4 |

Now the same sweep across k, against what the decoder itself is capable of. "Ceiling" is the


| k              | 2       | 4       | 6       | 8       | 12      | 16      | 24      | 32      |
| -------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| ceiling        | 1.24e-1 | 1.55e-2 | 8.84e-3 | 7.04e-3 | 6.24e-3 | 4.13e-3 | 3.98e-3 | 3.28e-3 |
| we get, mean   | 5.46e-1 | 1.74e-2 | 5.85e-2 | 8.48e-3 | 4.79e-2 | 6.54e-3 | 1.49e-2 | 4.02e-2 |
| we get, median | —      | 1.55e-2 | 8.52e-3 | 7.25e-3 | 7.58e-3 | 5.05e-3 | 4.93e-3 | 3.66e-3 |

The cause is in the optimiser, not the model. It starts almost undamped, so its first step is


| k                     | 4    | 6    | 8    | 12   | 16   | 24   | 32    |
| --------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ----- |
| ratio to ceiling, now | 1.11 | 7.61 | 1.10 | 7.85 | 1.26 | 3.40 | 15.49 |
| with the fix          | 1.09 | 1.05 | 1.09 | 1.18 | 1.28 | 1.45 | 1.50  |


**Where the time goes**

![breakdown of online time](talk_figs/8_where_time_goes_poisson.png)

We made the solve independent of grid size, and it worked. So the solve is no longer where the
time goes:

| grid           | 32 | 64 | 128 | 256 | 512 |
| -------------- | -- | -- | --- | --- | --- |
| spent decoding | 4% | 6% | 11% | 50% | 84% |

At 512×512 the latent solve  the part we optimised  is 9% of the total.

### How few points the residual actually needs![error and cost against quadrature points](talk_figs/9_eq_points.png)

At 64×64, k=8, M=64 test modes:

| points            | share of grid | error   | cost             |
| ----------------- | ------------- | ------- | ---------------- |
| **Poisson** |               |         | *ms per solve* |
| 64                | 2%            | 5.48e-2 | 11.1             |
| 128               | 3%            | 1.80e-2 | 13.9             |
| 256               | 7%            | 8.66e-3 | 15.1             |
| 512               | 13%           | 7.68e-3 | 15.2             |
| 1024              | 27%           | 7.65e-3 | 18.9             |
| every point       | 100%          | 7.65e-3 | 35.6             |
| **Burgers** |               |         | *ms per step*  |
| 64                | 2%            | 6.54e-2 | 3.4              |
| 128               | 3%            | 1.95e-2 | 4.5              |
| 256               | 7%            | 1.74e-2 | 6.2              |
| 512               | 13%           | 1.68e-2 | 10.8             |
| 1024              | 27%           | 1.67e-2 | 20.7             |
| every point       | 100%          | 1.65e-2 | 70.9             |

Both problems behave the same way. Thirteen percent of the points gives essentially the
full-grid error — identical on Poisson, 1.7% worse on Burgers — for 43% and 15% of the cost.
Seven percent is already within 13% and 5% of it, at 42% and 9% of the cost.

Error stops improving once the quadrature fit reaches about 1e-3; below that the decoder's own
ceiling binds, not the quadrature. That makes the fit residual a usable way to choose the
number of points without needing held-out error.

The share matters more as the grid grows. Holding the point count at 256 and refining:

| grid              | interior nodes | share of grid | error   | solve              |
| ----------------- | -------------- | ------------- | ------- | ------------------ |
| **Poisson** |                |               |         | *ms per solve*   |
| 32×32            | 900            | 28.4%         | 1.08e-2 | 8.8                |
| 64×64            | 3,844          | 6.7%          | 8.48e-3 | 7.2                |
| 128×128          | 15,876         | 1.6%          | 8.40e-3 | 8.3                |
| 256×256          | 64,516         | 0.4%          | 8.38e-3 | 6.4                |
| 512×512          | 260,100        | 0.1%          | 8.61e-3 | 6.0                |
| **Burgers** |                |               |         | *ms per rollout* |
| 32×32            | 900            | 28.4%         | 1.62e-2 | 331                |
| 64×64            | 3,844          | 6.7%          | 1.58e-2 | 307                |
| 128×128          | 15,876         | 1.6%          | 1.67e-2 | 304                |
| 256×256          | 64,516         | 0.4%          | 1.76e-2 | 362                |

The same 256 points cover 28% of the coarsest grid and 0.1% of the finest, and the error and
the solve cost barely move. That is the mesh-independence result seen from the quadrature side:
the number of points the residual needs is set by the problem, not by the grid.

---

## Speed against accuracy

Poisson, cheapest way to reach about 1% error. Every timing below is from a single run on a
single GPU, so the rows are comparable to each other:

| grid | ours | error | classical solver | |
|---|---|---|---|---|
| 32×32 | 3.3 ms | 1.30e-2 | 1.2 ms | 0.35× |
| 64×64 | 3.4 ms | 1.16e-2 | 2.2 ms | 0.65× |
| 128×128 | 3.6 ms | 1.15e-2 | 4.6 ms | **1.26×** |
| 256×256 | 4.6 ms | 1.13e-2 | 7.9 ms | **1.73×** |
| 512×512 | 7.9 ms | 1.18e-2 | 26.7 ms | **3.39×** |

Our cost is nearly flat — 3.3 to 7.9 ms across a 256-fold change in unknowns — while the
classical solver's grows 23×. We cross over between 128×128 and 256×256, and the advantage keeps
widening after that.

POD is absent because it cannot reach 1% at all. Its best is 5.10e-2 at 64×64, and more modes
do not help. At 0.37 ms it is roughly ten times cheaper than us where 5% is good enough, so the
two methods do not really compete: POD owns the rough end, we own everything below it.

One trained model, two settings:

| grid     | fastest |        | most accurate |         |                            |
| -------- | ------- | ------ | ------------- | ------- | -------------------------- |
| 64×64   | 1.8 ms  | 8.4e-2 | 3.4 ms        | 1.16e-2 | 1.9× time, 7.2× accuracy |
| 256×256 | 3.2 ms  | 8.3e-2 | 4.8 ms        | 1.13e-2 | 1.5× time, 7.4× accuracy |
| 512×512 | 7.3 ms  | 9.3e-2 | 8.8 ms        | 1.18e-2 | 1.2× time, 7.9× accuracy |

The trade gets better as the grid refines.

At k=8 on a 64×64 grid across the 3 PDEs we have run:

|         | ceiling | ours    | POD     |
| ------- | ------- | ------- | ------- |
| Poisson | 7.11e-3 | 7.65e-3 | 1.77e-1 |
| Heat    | 1.16e-2 | 1.87e-2 | 1.29e-1 |
| Burgers | 1.15e-2 | 1.65e-2 | 2.09e-1 |

On the three that work we land within 1.1–1.6× of the decoder's own floor, so the solve is
close to as good as the representation allows. Wave fails outright.

---

## Warm-starting the classical solver

Let the reduced model produce a rough answer, hand it to the classical solver to finish. Same
final accuracy, so only cost is in question.

| grid     | classical alone | with warm start | iterations, plain → warm |        |
| -------- | --------------- | --------------- | ------------------------- | ------ |
| 64×64   | 4.8 ms          | 6.8 ms          | 158 → 168                | 0.71× |
| 256×256 | 19.9 ms         | 22.6 ms         | 658 → 616                | 0.88× |
| 512×512 | 61.7 ms         | 65.8 ms         | 1342 → 1262              | 0.94× |
