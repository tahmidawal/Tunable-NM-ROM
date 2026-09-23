
### quadratic-manifold — `qmn102` (job 4186220): the GWW / Barnett–Farhat quadratic manifold beats POD-LSPG at matched dimension, and does not come near the head

Worktree `worktrees/2026-09-22-quadratic-manifold`, branch `exp/2026-09-22-quadratic-manifold` at
`83d7f462`, forked from `exp/2026-09-17-b-panel` @ `25434a27`. Namespace
`/cluster/tufts/paralab/tawal01/qman_20260922/` (now empty, both job dirs deleted). **2 of the
4-job budget spent.** Design `experiments/quadratic-manifold/DESIGN.md` (pre-registered before the
first GPU job, amendments §A1–A2); independent audit `checks/design-audit.md`; report
`experiments/quadratic-manifold/reports/2026-09-22-quadratic-manifold.md` with its generator
(`render_report.py` + `make_table.py`) beside it; numbers `reports/summary.json` (hashes in `artifacts/qmn102/SHA256SUMS`),
audit `checks/qmn102-audit.json`, raw `artifacts/qmn102/result.json.gz`, log and sbatch beside it.

**What was run.** b-panel's same-allocation 256² Burgers harness (`exp/2026-09-17-b-panel` @
`25434a27`, every copied file hashed in `COPIED-FROM.json`) with **one new subject family**,
`qman`: $u = u_{\rm ref} + V_r a + W\,\mathrm{vech}(a \otimes a)$, $W$ from one
ridge-regularised linear solve on the panel's own truth snapshots — no network, no training run.
The trial map is the only thing that changes: the columns $[u_{\rm ref} \mid V_r \mid W]$ become
an `arms.GridBank` and `qman.head` the coefficient map, so the residual, weak test projection,
initializer, LM driver, budgets and output contract are the ones POD-LSPG and the NM-ROM arms
already run; `jax.jacfwd` supplies $V + 2W(a\otimes\cdot)$, checked against the analytic Jacobian
to 1e-13. 29 arms in ONE allocation on one A100-80G, 35.2 min, 5 timed repetitions, randomised
order: the $r$ ladder with $W$ on and off, two pre-declared sensitivity arms, the NM-ROM fast and
accurate settings, POD-LSPG at 8/16/32/64/256/512 and the eight Newton–BiCGStab full-order
settings. All audit gates pass, cross-job fidelity gates included, so the panel is measuring the
archived model.

**What was found.**

| $r$ | $P$ | ridge | $\lVert W\rVert_F$ | quad worst ev. % | lin % | POD % | gain | quad ms | cost vs lin |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 36 | 0.01 | 0.0578 | 40.779 | 52.439 | 52.560 | 1.29× | 42.0 | 1.88× |
| 16 | 136 | 0.01 | 0.0733 | 22.214 | 28.433 | 28.725 | 1.28× | 85.3 | 1.83× |
| 32 | 528 | 0.01 | 0.0743 | 12.685 | 18.827 | 18.799 | 1.48× | 223.9 | 2.72× |
| 64 | 2080 | 1 (grid top) | 0.0039 | 6.942 | 7.097 | 7.084 | 1.02× | 696.5 | 4.64× |

1. *Accuracy.* The quadratic term is real: against an **identical** linear part it cuts the worst
   evolved error 1.29× / 1.28× / 1.48× at $r=8,16,32$, and beats POD-LSPG at the same rank at every
   one of those rungs. It does **not** reach the head — 22.214 % against
   **1.8891 %** at 16 solved unknowns, a factor of
   11.8. Between POD-LSPG and us, much
   nearer POD-LSPG. At $r=64$ it stops paying entirely (1.02×): the ridge pins at the grid top and
   $\lVert W\rVert_F$ collapses two orders of magnitude, because $P=2080$ cannot be fitted from
   3328 snapshots. **That is the prediction DESIGN §3.1 recorded in writing before the job.**
2. *Cost.* 1.88× its own linear control at $r=8$ rising to 4.64× at $r=64$. No quadratic arm beats
   the paper-rule FOM; best 0.210×, i.e.
   4.8× slower.
3. *Unknowns.* For **every** quadratic arm the cheapest NM-ROM at least as accurate is the same one,
   `q0_M64_eqcert_fastL4` at **16 unknowns, 39.6 ms,
   1.8891 %**: 4× fewer unknowns than the best quadratic arm,
   17.6× cheaper,
   3.7× more accurate.
4. *Why.* Representational, not numerical. Every reduced arm's representation floor, $t=0$
   compression and worst all-times error agree to three decimals — the LSPG solve is finding
   essentially the best fit its manifold admits.

**Reported against ourselves, not buried.** The pre-declared fixed ridge $10^{-4}$ beats the
rule-selected $0.01$ by 1.42× at $r=32$
(8.946 % against
12.685 %), so **the ladder is a lower bound on a tuned quadratic
manifold**. This lane gives the baseline no hyper-reduction while GWW and BF both do; the
dense-against-dense ratio is in the report §5 and the head still wins it
2.44× on cost and
3.7× on error. The M-sensitivity arm
clears the design: quadrupling the test modes at $r=32$ changes the error by
0.993×, so $M=4r$ was not the limiting factor.
Every quadratic arm's error is above the mesh's own discretisation error
(4.0265 %); the NM-ROM rows are below it.

**Wrong / retracted.**

* **The ridge holdout was leaky, and it handicapped the BASELINE — caught by the independent
  pre-job audit, before any number existed.** It split snapshot *columns* at random, but the
  snapshot matrix concatenates trajectories at 26 states each, so consecutive columns are
  near-duplicate states $\Delta t$ apart: almost every held-out column kept its own temporal
  neighbours in the training half, the criterion could not see overfitting, and it rewarded
  interpolation. Confirmed by measurement rather than argument (`checks/probe64.json`): the
  64-interval $r=8$ rung refitted with the ridge forced to $10^{-4}$ instead of the $0.0$ the
  column split chose goes from NOT converged (520 LM iterations), 83.27 % — 25 % worse than its own
  linear control — and 143.0 ms, to converged in ≤12 iterations, 64.92 % (better than both its
  linear control and POD-8) and 21.7 ms. **The holdout is now by trajectory.** Anyone reusing
  `qman.fit` must keep that.
* **`qmn101` (job 4185793) died in 9 min 55 s with no numbers**, caused by my own fix for an audit
  finding, not by the finding: flooring a denormal Gram eigenvalue at float-tiny made the condition
  number overflow to `inf`, and `ladder.dump` writes with `allow_nan=False`. A singular Gram now
  reports `None` plus `gram_rank_deficient`, and `qman.fit` asserts its info dict is JSON-encodable
  before returning, so a future non-finite diagnostic fails in seconds instead of after the
  reference solves. DESIGN §A2 keeps the manifold fits that job did produce, because they are the
  first look at the construction at 256² and they confirm the trajectory split selects an interior
  $\gamma$ on the real 128-trajectory set.
* **A `make_table.py` bug was caught before anything was written up**, not shipped: the ladder was
  keyed on (rank, variant), so at $r=32$ the two sensitivity arms silently overwrote the canonical
  rung and the ladder reported the fixed-ridge arm's numbers as its own. The ladder now keys on the
  canonical arm ($M=4r$, rule-selected ridge) and lists the sensitivity arms separately. Any figure
  or sentence built from a `summary.json` generated before `83d7f462` is wrong at $r=32$.
* Codex could not audit: `codex exec -s read-only` failed to start its sandbox on this box
  (`bwrap: loopback: Failed RTM_NEWADDR`) and correctly returned an access blocker having read
  nothing. An independent subagent auditor was commissioned with the same eight-question brief.

**Open / for the paper lane.** The lane's question is answered and 2 of 4 jobs are unspent. The row
is ready for the main table; DESIGN §7 pre-registered that if the quadratic manifold had won, the
positioning claim would be rewritten — it did not, and the honest statement is the one above,
including the lower-bound caveat and the missing hyper-reduction. Two threads if it is reopened: a
ridge chosen better than the held-out-snapshot rule (worth ~1.4× at $r=32$), and an EQ rule for the
quadratic manifold (a lane's worth of work; it would have to buy more than an order of magnitude to
change the verdict). **Worktree not merged — ask the user.**

