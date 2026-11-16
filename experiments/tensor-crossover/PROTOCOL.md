# tensor-crossover: when off-mesh quadrature of the advection term saves storage or time

Pre-registration, written 2026-11-15 before any job of this lane ran. Lane `exp/2026-11-15-tensor-crossover`
(worktree `worktrees/2026-11-15-tensor-crossover`, forked from `main` @`274a3a9af`; code-only mirror
`origin/codeonly/exp/2026-11-15-tensor-crossover`). Cluster namespace `/cluster/tufts/paralab/tawal01/tcross-20261115/`.
At most 3 lane GPU jobs at once, one GPU each, `gpu` partition, A100 80GB for every timed job (one GPU type for the whole
figure). Later changes are dated amendments at the end, with the reason.

## 1. Question (JCP Figure 3)

For the frozen Burgers 2D and 3D off-mesh reduced models, at which bank width $r$ does the off-mesh quadrature of the
nonlinear (advection) term become cheaper than exact contraction with a precomputed coefficient tensor, (a) in stored
operator bytes, (b) in time for one nonlinear-term evaluation, (c) in time for a full reduced query? Or does no crossover
occur in the tested range?

Existing evidence (context, not re-used as data): Burgers 3D $R'=512$ tensor 4.30 GB vs 0.34 GB Gauss advection arrays
(`reports/2026-10-01-burgers3d-offmesh-quadrature.md`); NS 3D $R=64$ tensor contraction 1.17–1.43× faster than quadrature at
the plateau (`exp/2026-11-05-ns3d-offmesh-quadrature`, report `reports/2026-11-05-ns3d-offmesh-quadrature.md`). The timing
design below copies the NS lane's: same allocation, warm-up excluded, interleaved repeats, plus a kernel microbenchmark
(a `lax.scan` of chained Jacobian evaluations, every entry feeding the next iteration).

## 2. The two paths compute the same reduced model

Both paths evaluate the same tested advection term, the off-mesh quadrature of the continuum term with rule $(x_q,w_q)_{q\le m}$:

- **quadrature** (as deployed): $N(c) = P^\top\big((Bc)\odot(Dc)\big)$, Jacobian $J(c)=P^\top(\mathrm{diag}(Dc)B+\mathrm{diag}(Bc)D)$,
  with $B=\hat G(x_q)$, $D$ the derivative block ($u_x+u_y[+u_z]$) and $P_{qm}=$ (scaled) $w_q\psi_m(x_q)$.
  Stored: $B, D \in \mathbb R^{m\times r}$, $P\in\mathbb R^{m\times M}$.
- **tensor**: the same quadratic form precomputed once, $T_{mij}=\sum_q P_{qm}(B_{qi}D_{qj}+D_{qi}B_{qj})$, $N(c)=\tfrac12 J(c)c$,
  $J(c)_{mi}=\sum_j T_{mij}c_j$. Stored: $T\in\mathbb R^{M\times r\times r}$.

Because $T$ is built from the same rule, the two paths are algebraically identical and must give the same reduced trajectory
up to round-off; this is checked (§5, gate G-eq). The storage and contraction cost of $T$ do not depend on which rule (or the
mesh stencil, as in the incumbent October tensor) generated it, so the tensor side of the figure is the cost of *any*
precomputed quadratic tensor of this shape. Everything else in a query (projection of $u_0$, linear operator $A$, LM solver,
decoding of the output fields) is shared and identical between the paths.

## 3. Frozen inputs (vendored byte-identically, `vendor/PROVENANCE.json`)

Copied from `exp/2026-11-05-burgers-quadrature-plateau` @`a2eae048a` (which vendored them from the October lanes):

- 2D: checkpoint dn256b + `rotation_R512.npz` (ordered 512-column rotated bank), `qcore.py` (point form, `make_linear_query`,
  `offmesh_rule` = Hari's generators), solver settings of `acc`: backward Euler $\Delta t=0.005$, LM gtol $10^{-3}$, step budget 600,
  trust = `trust_linear(r)`, cold start `build_cold_linear(r)`. Width $r$ = first $r$ columns of the ordered rotated bank
  (as `acc` $r=384$ and `fast` $r=128$ were); tests $M=4r$ (as `acc` 1536 and `fast` 512). Mesh $L=256$ (the model's
  native mesh).
- 3D: `model_M2/bank.pkl` (ordered 512-column bank), `offmesh.py` (`offmesh_tables`, `contract_offmesh`, `make_fsc_rule`),
  `tables.py` (projection, decode, linear tables). $\Delta t=0.01$, gtol $10^{-3}$, trust $=0.05\times$ `coefficient_rms_spread[r]`.
  Width $r$ = first $r$ bank columns (as $R'=256$ in October); tests $M=$ `complete_M(n, 4r)`. Mesh $n=65$ nodes ($64^3$ cells).
- Rules: 2D generated in-job by `qcore.offmesh_rule` and checked against the committed `rules2d.npz` (round-off, plateau gate
  G-R); 3D from the committed `rules3d.npz` with per-rule sha256 (`rules3d_manifest.json`).

Primary meshes are the smallest of each dimension because there the nonlinear term is the largest share of a query; the
kernel and storage results do not depend on the mesh. A second mesh is run only if GPU time allows (amendment if so).

## 4. Widths and rules

Widths $r \in \{64, 96, 128, 192, 256, 384, 512\}$ in both dimensions (512 is the full bank; no wider bank exists).

Rule per width, **never chosen by timing**:

- Primary family = the family of the sealed-closeout frozen primary rule (`worktrees/2026-11-15-jcp-sealed-closeout`
  `F1.json`/`DECISIONS.md`): 2D Fibonacci lattice (frozen `fib987` at $r=384$), 3D CBC rank-1 lattice (frozen `lat4096` at $r=512$).
  Secondary family = Gauss tensor (frozen 2D `gauss45` at $r=384$, 3D `gl20` at $r=512$).
- At the frozen width the rule is the frozen rule.
- At every other width: the first qualifying rung of the family's existing ladder (the plateau lane's ladders, `rules2d/3d`).
  *Qualification* is reference-free, on development cases only (2D `dev6` = `engines.params_draw(7090702,4)` +
  `params_draw(911702,2)`; 3D `dev4` = seed 923651 × 4): for rung $k$ let
  $d_k = \max_{\text{cases}}\max_{t>0}\lVert F_k(t)-F_{\text{conv}}(t)\rVert/\lVert F_{\text{conv}}(t)\rVert$, the decoded-field distance from
  the family's top rung (2D `fib46368` / `gauss256`; 3D `lat32768` / `gl32`) at the same width. The tolerance $\tau$ of a family is
  $d$ of its frozen rule at the frozen width on the same dev cases (computed in the same job). The qualifying rung is the smallest
  rung $k$ such that $d_{k'} \le \tau$ for $k$ and every larger rung $k'$ of the family (suffix rule). If even the smallest rung
  qualifies, the selection is marked *ladder-floor* (the true minimum may be smaller, which would favour quadrature).
- Sensitivity: the frozen rule itself at every width (fixed $m$), timed alongside.
- No test, held-out, sealed or OOD cohort is touched by this lane.

## 5. Measurements (one allocation per dimension; every width's paths are timed in the same allocation)

For each width, after the rules are qualified:

1. **G-tab** tensor vs quadrature Jacobian on 4 random coefficient vectors: max relative difference $\le 10^{-12}$.
2. **G-eq** trajectory agreement on every dev case: decoded fields at the five output times and all internal coefficients of the
   tensor path vs the quadrature path built from the same rule; max over cases and times of
   $\lVert F_T-F_Q\rVert/\lVert F_Q\rVert \le 10^{-8}$ (registered tolerance; the actual values are reported), for the primary and
   the secondary rule. LM iteration counts are reported (equal expected).
3. **Full query time**, tensor vs quadrature (primary, secondary, frozen-rule sensitivity), float64, highest matmul precision:
   all subjects compiled and run once (warm-up, excluded), 30 s GPU burn-in, then 5 interleaved repetitions in a seeded
   shuffled order over the dev cases with a 0.1 s burn before each call; each call ends with `block_until_ready`. Reported:
   median, IQR, min, max, n, per-repetition block drift. Timed outputs must equal the accuracy-pass outputs (rel $\le 10^{-12}$).
4. **Nonlinear-term time alone**: a jitted `lax.scan` of $K=200$ chained Jacobian (and value) evaluations at a reached state,
   $c\leftarrow c+10^{-12}\sum_m J_{m\cdot}$ so no evaluation can be skipped; 5 interleaved repetitions as above; time per
   evaluation = total / $K$.
5. **Stored-operator bytes**: tensor = bytes of $T$; quadrature = bytes of $B, D, P$ (the advection arrays only). Shared arrays
   (bank on the mesh, $A$, $\lambda$, Gram factor, cold-start tables) are reported once per width, separately.
6. **Device memory**: per compiled query and per compiled kernel, XLA `memory_analysis()` (argument, output, temp bytes) — the
   device memory that executable needs; plus the process-wide `peak_bytes_in_use` after each width. Stored bytes and device
   memory are never mixed in one number.
7. **Tensor build time** (offline) is recorded separately; it is not part of a query.
8. **OOM**: any `RESOURCE_EXHAUSTED` while building or running the tensor path at a width is caught, recorded as `OOM` for
   that width (with the attempted bytes), and the job continues with the quadrature path.

## 6. Crossover definitions (decided now)

For each dimension and for each measure (storage; kernel time; full-query time), using the primary rule per width:

- ratio $\rho(r)$ = tensor / quadrature (bytes or median time). Quadrature *wins* at $r$ if $\rho(r)>1$; for times the win is
  *separated* if the quadrature 75th percentile is below the tensor 25th percentile, otherwise it is a *tie*.
- Crossover width $r^\times$ = the smallest tested width from which quadrature wins (separated, for times) at that and every
  larger tested width (an OOM tensor counts as a quadrature win). Reported with its bracket (previous tested width, $r^\times$]
  and a descriptive log-log interpolation of $\rho=1$ between the bracket points. If quadrature never wins, "no crossover in
  64–512"; if it wins at every width, "quadrature cheaper throughout 64–512".
- The same is reported, as secondary, for the Gauss family and for the fixed frozen rule.
- Flop/byte model (for the reader, not a gate): tensor $J$ costs $2Mr^2$ flop and reads $8Mr^2$ bytes; quadrature $J$ costs about
  $2mMr$ (dominant $P^\top$ product) + $4mr$ flop and reads $8m(2r+M)$ bytes.

## 7. Outputs and stopping

`results/<job>/result.json` (pulled), `results/crossover.json` (assembled by `make_figure.py` from the result JSONs),
`fig3_crossover.png` (time and storage vs width, 2D and 3D), `RESULTS.md` generated from the JSON. Stop after one complete job
per dimension; a failed infrastructure run may be resubmitted with the same committed config (failures are accounted). If the
3D job cannot fit all widths in its wall time, it is split by width into separate jobs (each width's two paths stay in one
allocation).

## Glossary

- **width $r$**: number of bank columns (reduced coefficients) used by the reduced model; the bank is ordered, so width $r$ is its
  first $r$ columns.
- **$M$**: number of sine test functions of the Petrov–Galerkin residual ($\approx 4r$).
- **$m$**: number of quadrature points of the off-mesh rule.
- **tensor path**: nonlinear term evaluated by contracting a precomputed $M\times r\times r$ coefficient array.
- **quadrature path**: nonlinear term evaluated from the bank and its derivatives at the $m$ rule points.
- **rung / ladder / family**: one rule size / the list of sizes of a rule family / Fibonacci, CBC lattice, Gauss tensor.
- **frozen rule**: the rule fixed by the sealed closeout before it opened the sealed cohort.
- **dev cases**: development cases never used for any reported accuracy claim.
- **IQR**: interquartile range (75th minus 25th percentile).

## Amendments

**A1 (2026-11-16, ~00:10).** Second meshes (the option of §3): 2D $L=1024$ (`configs/c2d1024.json`) and 3D $n=129$, $128^3$
cells (`configs/c3d129.json`), identical configs except the mesh, each its own job with its own in-job qualification and
$\tau$. Decided while the primary jobs were running and after their first widths had been logged (r=384/512 and r=64), because
the primary jobs took ~10 min instead of the ~1–2 h budgeted, leaving GPU time. Purpose: show how the shared grid-bound part of a
query (projection, decoding) dilutes the nonlinear-term difference on finer meshes. They do not change any primary definition;
the figure's primary panels stay the smallest meshes; the finer meshes are a separate figure.
