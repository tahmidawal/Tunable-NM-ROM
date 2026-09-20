# SPEED-LOG — hires-heat

Running record of the profile → hypothesis → fix → re-measure loop. Every row cites the job whose `summary.json` holds the numbers; an optimisation is kept only with a same-job parity gate and before/after timing.

| # | date | hypothesis | change (arm name) | parity gate | measured effect | verdict |
|---|---|---|---|---|---|---|
| 0 | 2026-09-20 | (orientation, local GB10 smoke, NOT a result) 3D query is dominated by sequential LM solves: 20 CN steps x ~4-9 LM iterations plus a 4-start initial fit of up to 48 iterations; decode and encode are < 3 % | — | — | stage profile at 32^3, shared noisy box: init 22-296 ms, evolve 141-506 ms, decode <= 2.5 ms | hypothesis to test on the cluster in job h3d-profile03 |
