# SPEED-LOG — hires-burgers

Running log of the profile → hypothesis → fix → re-measure loop (protocol §"Standing mandate").
One row per hypothesis. "Kept" requires a parity gate against the unoptimised path and a
same-job before/after timing; numbers are copied from the named `summary.json`, never typed
from memory. Status `pending` means the measuring job has not landed.

## Starting point (orientation from audited earlier panels, not new claims)

Burgers 2D, frozen checkpoint $K=16$, $R=512$. b-panel `bpn301` (A100, $256^2$): $q=0$ EQ
43.4 ms at tol $10^{-3}$; $q=256$, $M=1088$ EQ (b-eqtop rule, $m=2560$) 466 ms at $10^{-3}$ /
746 ms at $10^{-6}$; dense 3940 ms. b-speed: the query is kernel-launch bound, ≈17 µs per step
+ 141 µs per LM iteration at $q=0$; `fuse` 1.25×, `lean` fold 1.15×, block Gauss–Jordan loses.

Static reading of the audited $q>0$ path (`topfix.make_query` → `varpro.make_block_lm`), before
any measurement — this is the hypothesis list the first job tests:

| # | where the accurate rung's time can go | evidence before measuring |
|---|---|---|
| H1 | forward-mode Jacobian pushes $K+q=272$ tangents through the $R=512$-wide stencil bank: $m\cdot5\cdot R\cdot(K+q) \approx 1.8$ GFLOP per Jacobian, more than the $M\times m\times(K+q)=0.76$ GFLOP test projection | code reading |
| H2 | every accepted LM iteration evaluates the residual twice and the Jacobian once under `lax.cond` (b-speed `fuse`) | b-speed measured 1.25× at $q=0$ |
| H3 | `blocks()` per time step: two extra forward-mode Jacobians ($z$ and $y$ blocks) purely for a stationarity diagnostic that equals the LM's own exit gradient | code reading; ≈ +1 Jacobian per step on 2–3 iterations per step |
| H4 | QR of $R_cC$ ($2304\times q$) recomputed inside every query's initial fit | code reading |
| H5 | LU of the $272\times272$ damped normal matrix (cuSOLVER launch-bound) — Cholesky is the cheaper factorisation of an SPD matrix | to measure |
| H6 | decode as six separate $G h$ matvecs reads $G$ six times: 17 GB at $2048^2$, 68 GB at $4096^2$ | bandwidth arithmetic: ≈6×68 GB / 4.8 TB/s ≈ 85 ms at $4096^2$ vs ≈15–20 ms fused |
| H7 | test count: $M=544$ at $q=256$ halves the projection and the Gram cost; b-qxm measured 0.757 % (vs 0.519 % at $M=1088$) at $256^2$ | b-qxm `bqx201` |
| H8 | iterations per step (2–3 at tol $10^{-3}$, 5 at $10^{-6}$): predictor quality | b-panel rows |

## Log

| date | job | hypothesis | change | parity gate | before → after (same job) | kept? |
|---|---|---|---|---|---|---|
| 2026-09-20 | hb2k01 (pending) | H1–H4, H6 | `hfast.py`: `fold`+`ajac`+`fuse`+`hoist`+`nodiag`+`decfuse` vs audited base, same rule and tolerance | fields ≤1e-9, integers reported | pending | pending |
| 2026-09-20 | hb2k01 (pending) | H5 | `solver='chol'` vs `'lu'` | labelled arm | pending | pending |
| 2026-09-20 | hb2k01 (pending) | H7 | $(q,M)=(256,544)$ vs $(256,1088)$ | accuracy measured directly | pending | pending |
