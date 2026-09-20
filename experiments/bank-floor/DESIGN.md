# bank-floor — can a better spatial bank lower the ROM's error floor by 3× at acceptable online cost?

Pre-registration, written 2026-09-20 before any GPU job. Lane of the 2026-09-20 speed-and-accuracy
campaign (`reports/2026-09-20-speed-accuracy-campaign-protocol.md` on main). Upside lane, not
load-bearing: arms that clearly fail are stopped, not rescued.

Worktree `worktrees/2026-09-20-bank-floor`, branch `exp/2026-09-20-bank-floor`, forked from
`exp/2026-09-16-b-head-train` @ `a53b8c9f`. Cluster namespace
`/cluster/tufts/paralab/tawal01/bankfloor_20260920/`, one directory per job. Budget: ≤ 2 running,
≤ 8 total GPU jobs.

## Question

The decoder is $u(x;z) = G(x)\,h(z)$ with a frozen spatial bank $G\in\mathbb{R}^{n\times R}$
(random-Fourier-feature coordinate network times the boundary polynomial). Every online operating
point, including the full-bank rung $q=R$, is bounded below by the bank projection floor

$$\varepsilon_G(u) = \frac{\lVert u - P_G u\rVert_2}{\lVert u\rVert_2},\qquad P_G = Q Q^{\top},\; Q = \operatorname{orth}(G).$$

Measured incumbents: Burgers 2D (256 intervals) worst $\varepsilon_G$ = 0.392 % at $R=512$ on six
development cases; Poisson 2D 0.742–0.746 % at $R=512$. The user wants ≤ 0.5 %, ideally ~0.1 %.
**Can a better bank lower the held-out floor by ≥ 3× without breaking the online cost?**

## Arms (per PDE, all on ONE common training snapshot set)

| arm | bank | how it is obtained |
|---|---|---|
| `inc512` | incumbent coordinate network, $R=512$ | frozen, control; must reproduce the logged floor |
| `pod{256,512,1024,2048}` | grid-bound orthonormal modes | **control that must run**: exact minimiser of the learned arms' own training loss (eigen-decomposition of the unit-normalised snapshot Gram); `podraw*` = classical un-normalised POD beside it |
| `pod*_sub` | same, from only the incumbent bank's own training subset | positions `inc512` against the optimum on its own data (free: a sub-block of the same Gram) |
| `ft512` | incumbent network, $R=512$, longer/better-conditioned training | variable-projection fine-tune (below) |
| `cat1024`, `cat2048` | incumbent network **concatenated** with a fresh RFF coordinate network of $R-512$ columns | variable-projection training of both blocks |

**Variable projection (the "better-conditioned training").** The earlier banks were trained jointly
with a head and per-snapshot codes, so the bank only ever saw the head's $K$-dimensional image.
Here the bank is trained against the floor itself, no head and no codes:

$$\mathcal{L}(\theta) = \frac{1}{B}\sum_{i\in\text{batch}} \frac{\lVert u_i - \hat G_\theta\,(\hat G_\theta^{\top}\hat G_\theta + \epsilon I)^{-1}\hat G_\theta^{\top}u_i\rVert^2}{\lVert u_i\rVert^2},$$

$\hat G$ = column-normalised features on the full training grid, $\epsilon = 10^{-9}$ (unit-diagonal
Gram, so Cholesky is finite in f64 whatever the conditioning; the gauge freedom of $G$ makes an
orthonormality penalty unnecessary — the penalty is what destroyed the retracted R=1024 arm). Adam,
cosine schedule, fixed step budget, **final iterate, no checkpoint selection**, so no held-out data
can leak. The `pod*` control is the global optimum of exactly this loss at $\epsilon=0$ over all
rank-$R$ subspaces, so `learned / pod` at equal $R$ is the distance to optimal.

**Why concatenation and not `widen`.** `widen` kept the incumbent's bank columns but randomised new
head columns and then calibrated a penalty at that broken start (amendment A4 of b-head-train,
RETRACTED). A concatenated bank contains the incumbent's span by construction. Also the g-network's
last hidden width bounds the rank ($\le$ hidden+1: 1025 Burgers, 513 Poisson), so a wider output
layer alone cannot reach $R=2048$; the fresh block has hidden width $\max(1024, R-512)$ and
multi-scale frequencies $\{4, 8, 16\}$.

**Warm-start assertions (hard, in-job, before step 1):** block 0's weight hash equals the incumbent
checkpoint's; block-0 features equal the incumbent bank to 0.0; and on a fixed probe set the
concatenated bank's floor is ≤ the incumbent's per snapshot (+1e-9). After training the train-probe
floor must be below its initial value, otherwise the arm is labelled `training_did_not_take`.

## Data and cohorts (f64, regenerated from seeds on the cluster)

- **Burgers 2D**, 256 intervals, $n=65025$, $dt=0.005$, horizon 0.25. Train: first 1024 trajectories
  of `common.incumbent_draw()` (the incumbent's own draw; its first 576 are the incumbent bank's
  training trajectories = the `_sub` subset), every 2nd state (26 per trajectory, 26 624 snapshots),
  FOM tolerances 1e-9/1e-7 as b-head-train. Held-out: **dev6** = `params_draw(7090702,4)` +
  `params_draw(911702,2)` at the six output times (the cohort behind 0.3918 %), and **hold64** =
  `params_draw(20260916,64)`, all 51 states. Disjointness asserted on values.
- **Poisson 2D**, 255 intervals, $n=64516$. Train: the 2611 fit sources of `bank_R512_S3072`
  (`source_params(0,3072)`, split seed 20260916) = `_sub`, plus `source_params(20260920, 13773)`,
  total 16 384. Held-out: **dev12** = `source_params(7090703,6)`+`source_params(7090732,6)`;
  **common256** = `source_params(20260916,256)` (note: this cohort selected the incumbent among six
  banks on 09-16, a small bias in the incumbent's favour); **fresh256** = `source_params(20260921,256)`,
  never used by anything. Learned banks are also scored at 1023 intervals on dev12 (mesh transfer);
  POD is grid-bound and is scored at 255 only.
- Sealed/final cohorts are not touched. Nothing is selected on any held-out cohort.

## Numerical guards

Floors use a twice-orthonormalised basis: $R_f=\operatorname{qr}(G)$ (recomputed if non-finite —
GB10 landmine), SVD of $R_f$, numerical rank by the $\sigma_1 n\,\epsilon_{mach}$ threshold,
$Q = G V_r\Sigma_r^{-1}$ then one more QR; assert finite and $\lVert Q^\top Q - I\rVert_{\max}<10^{-10}$;
the residual is formed explicitly ($u - Q Q^\top u$), never as a difference of squares. Effective
rank and $\kappa$ are reported; `rank_valid` requires rank $=R$. POD via the snapshot Gram is
accurate while $\sigma_R/\sigma_1 \gg 10^{-8}$; the ratio is reported and asserted $>10^{-7}$.
The bank is always an explicit jit argument (captured-constant landmine).

## Phase 1 (representation; jobs `bfb01`, `bfp01`, optionally `bfb02` for `cat2048` Burgers)

Report per arm: train-probe floor, held-out floor (worst / median / mean per cohort), rank, $\kappa$,
training seconds, and the **head-transplant defect**: the incumbent head's decoded fields
$G_{inc}h(Z)$ on 4096 stored training codes, projected onto the new bank. A defect ≪ the head's
2.5 % / 3.1 % error means the existing head is reusable on the new bank through the linear re-fit
$h' = (G'^{+}G_{inc})\,h$ at zero training cost. **Heads are not retrained** (b-head-train did not
reproduce the incumbent; not this lane's line).

Cost model reported beside each arm: decode $\propto nR$; operator pre-assembly $M\times R$ with
$M = 4(K+q)$; bank storage $8nR$ bytes.

## Phase 2 (solve + cost; only for arms that pass the Phase-1 gate; ≤ 3 jobs)

For the incumbent and the promoted banks, in ONE allocation per PDE: the full-bank ($q=R$,
identity head) solved error on the held-out development cohort through the unchanged
head-ablation machinery; ROM-fast ($q=0$, transplanted head) and ROM-accurate rows beside the named
FOM and the other required comparators of the protocol (Burgers: tolerance-matched Newton FOM;
Poisson: DST direct as labelled control, CG as named FOM, coarse-grid FOM at matched accuracy);
GPU burn-in, synchronised timing, ≥ 5 repetitions, medians, GPU model + UUID. Any EQ rule used on a
new bank is re-fitted and **certified by held-out $\rho$ on reachable states, never by NNLS fit**;
an uncertified rung is reported dense.

## Pass bar and stop rules (pre-registered)

- **Gate P1 (promotion):** an arm is promoted to Phase 2 iff its worst held-out floor is ≤ 1/3 of
  `inc512`'s on the primary cohort (dev6 Burgers; dev12 Poisson) **and** the same direction holds on
  the large cohort (hold64 / fresh256) with ratio ≤ 1/2.
- **Lane success:** a promoted bank whose full-bank solved worst error is ≤ 0.5 % (stretch ≤ 0.2 %)
  with online cost ≤ 4× the incumbent's same-rung cost. Missing it is a result.
- **Stop an arm** if the warm-start assertion fails (bug, fix once, else drop), if training does not
  take, or if its held-out floor is > 0.8× the incumbent's after the full budget.
- **Stop the learned line for budget reasons** if `pod2048` itself fails P1. (Codex r2 finding 1:
  POD minimises the *training mean-square* floor, not the held-out worst case, so this is NOT a
  lower bound on what a rank-2048 bank can do and is never reported as one.)
- If POD passes and the coordinate networks do not, the deliverable is the POD bank (grid-bound:
  fixed mesh, no mesh transfer) with that limitation stated.

## Controls that must fail / fidelity gates

- `inc512` must reproduce 0.3918 % (Burgers dev6) and 0.7459 % (Poisson dev12, 255 intervals) to
  1e-3 relative; otherwise the floor routine or the cohort is wrong and nothing is reported.
- A random (untrained) fresh block of 512 columns alone must give a floor > 5 % (shows the metric
  is not trivially small at this rank).
- Train-probe floor of `pod R` must be ≤ every learned arm's train-probe **mean-square** loss at
  equal $R$ (optimality of the control; a violation means a bug).
- Independent NumPy re-computation of held-out floors from the saved bases (POD) / re-built banks.

## Deliverables

Table floor vs $R$ vs cost per PDE (generated from JSON); statement on sub-0.5 % / sub-0.2 %
reachability and its online cost; bank checkpoints under `experiments/bank-floor/ckpt/` (git-ignored)
with `ckpt/MANIFEST.json` (path, bytes, SHA256, arm, source job) committed as
`experiments/bank-floor/CKPT-MANIFEST.json`.

## Glossary

- **bank $G$**: the $n\times R$ matrix of spatial basis functions, produced by a coordinate network
  so it can be evaluated on any mesh; frozen online.
- **floor**: relative $L^2$ error of the best possible approximation of a true solution inside
  span($G$); no ROM using that bank can do better.
- **head**: the network mapping the $K$ latent coordinates to the $R$ bank coefficients.
- **$q$**: number of extra linear bank directions solved for online (corrections); $q=R$ uses the
  whole bank.
- **POD**: proper orthogonal decomposition, the optimal linear basis of given rank for a snapshot set.
- **variable projection**: eliminating the linear coefficients in closed form so only the basis is
  trained.
- **$\rho$**: held-out ratio of quadrature residual error to residual norm, the EQ certification metric.
- **dev6 / hold64 / dev12 / common256 / fresh256**: held-out parameter cohorts defined above.

## Amendment A1 — 2026-09-20, before any job, from the Codex design audit

Audit: `checks/codex-design-audit-r2.txt` (r1 could not read files: the Codex sandbox cannot spawn
a shell on this box, so r2 inlined DESIGN.md, `bf_core.py` and the `sep_common` excerpt). Twelve
findings; dispositions:

1. *POD failing is not a rank lower bound* — accepted, stop rule reworded above.
2. *POD is the optimum of the unregularised full-training objective, not of the ridge minibatch
   loss* — accepted. Every arm now reports `train_full` (all training snapshots, QR floor, no
   ridge); the optimality check is `pod R` rms ≤ learned rms on `train_full` at equal $R$, and the
   text above should be read as "lower bound for that objective".
3. *stop-gradient on the column norms* — accepted, removed; the normalisation is differentiated.
4. *ridge/Cholesky guarantees overstated* — accepted: loss finiteness at every log step, parameter
   finiteness after training, and the ridge-free QR probe floor logged beside the ridge loss every
   10 log steps (that pair is the ridge-sensitivity check).
5–6. *Gram POD validation; SVD rotation broke nested prefixes* — accepted; **6 was a real bug**.
   POD now uses order-preserving QR, and each `pod`/`pod_sub` rank must satisfy the known-answer
   identity (mean-square floor on its own snapshots = discarded eigenvalue mass / count) to 1e-3.
7. *cohort roles* — accepted: dev6/hold64/dev12/common256/fresh256 are **validation** cohorts (they
   drive promotion). Lane-success numbers in Phase 2 are additionally scored ONCE on confirmation
   cohorts reserved now and not opened in Phase 1: Burgers `params_draw(20260922, 32)`, Poisson
   `source_params(20260922, 256)`.
8. *warm-start check vs the historical failure* — accepted: added the assertion that the incumbent
   head zero-padded to the new width decodes to the bit-identical field at the concatenated start.
9. *controls* — the random-bank floor is reported, not asserted; `training_took` is a label.
10. *rank* — promotion additionally requires `rank_valid`; cost proxies use the effective rank.
11. *online cost* — the Phase-1 cost columns are proxies and are labelled so; Phase 2 measures.
12. *B is trained* — intentional and as in `sep_common.train_autodecoder` (only `out_scale` is
   frozen); final synchronisation and finiteness checks added.
