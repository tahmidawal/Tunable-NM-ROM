# DESIGN — quadrature-burgers3d: Hari's off-mesh quadrature on the full-scale Burgers 3D model

Branch `exp/2026-10-01-quadrature-burgers3d` (forked from `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`), worktree
`worktrees/2026-10-01-quadrature-burgers3d`, lane directory `experiments/quadrature-burgers3d/`. Cluster namespace
`/cluster/tufts/paralab/tawal01/quad3d_20261001/`, one sub-directory per job. Lane approved by the user on 2026-10-01.
Backups: commit small and often, `bash experiments/quadrature-burgers3d/sync_github.sh` after every commit (the
`origin/codeonly/` mirror); the `exp/` branch is never pushed. This file is written before any GPU job. Amendments are
appended, dated and labelled, and never rewrite earlier text.

## 1. Question

Hari's study (`external/quadrature-study-2026-09-30/quadrature/`, `results/SUMMARY.md` section 9, `REPORT3D.md`)
replaced the mesh-based nonlinear term of the reduced model by a fixed classical quadrature of the *continuum* term,
evaluated by decoding the coordinate-network bank and its analytic gradient at the rule's points. In 3D it found that
CBC rank-1 lattices beat tensor Gauss by one to two orders at equal point count, that Smolyak diverges, and that the
off-mesh rules reproduce the dense solve against a refined reference, mesh-invariantly, at a cost flat in the mesh.
His 3D bank was weak ($R = 128$ on 64 trajectories, 40 % worst floor), so his absolute errors carry no information.
This lane repeats the study on our strong frozen Burgers 3D model and asks:

1. With a strong bank, does the lattice still beat Gauss at equal $m$, and what $m$ reaches the incumbent tensor's
   accuracy?
2. Do off-mesh rules reproduce the tensor / dense solve against the refined reference, mesh-invariantly from 64³ to
   256³?
3. Cost: is the off-mesh query cheaper than the $R'^2 M$ quadratic tensor? Tensor memory and time against the
   rule's $m(2R' + M)$ at $R' = 512$ and $256$, and the effect on the speedup against Newton–BiCGStab.
4. (Optional, only if budget remains and everything else is done; not planned.) Can an off-mesh rule let the k-head be
   revisited? Nothing is retrained in this lane without a written amendment first.

## 2. The model (frozen, nothing trained here)

The burgers3d-retry model **M2**: span-only coordinate-network bank $G(x) = \mu(x)\,g_\phi(x)$, $R = 512$, MLP width
1024, trained on 1536 trajectories at 33/65/129 nodes, rotated once into the importance-ordered bank
$\hat G = G T$. `inputs/model_M2/bank.pkl` sha256 `6687f259947ec08a986732db157db69c66f1110ab28125a029de17b4aa389af9`
(identical to `burgers3d-retry/inputs/model_M2/bank.pkl`, branch `exp/2026-09-23-burgers3d-retry` @ `642ab587`). The
head is not used (it did not generalise). Vendored code, byte-identical copies from that commit, in `vendor/`
(`vendor/PROVENANCE.json`): `burgers3d-span/common.py` (family, FOM, DST, LM), `burgers3d-retry/tables.py` (lean
per-mesh tables, `make_fsc`), `paper-b3d/vendor/b3d_common.py` (the blob family and stencils).

PDE: $u_t + u\,(u_x + u_y + u_z) = \nu\,\Delta u$ on $(0,1)^3$, $u = 0$ on the walls, 1–3 Gaussian blobs,
$\nu \in [0.01, 0.1]$ log-uniform, $t \in [0, 0.25]$, outputs at $t = 0, 0.05, \dots, 0.25$. Mesh $n$ = nodes per axis
including walls (65, 129, 257 = 64³, 128³, 256³ cells); $N = n - 1$.

Reduced model (unchanged except for the advection term): $u = \hat G_{[:, :R']}\, c$, LSPG on the $M$ lowest discrete
sine tests $\Phi$ (mesh-orthonormal), $M$ = $4R'$ completed to the end of its eigen-shell, row scaling
$S = 1/(1 + \Delta t\,\nu\lambda)$:

$$ r(c) = S \odot \Big( A c - A c_{\rm prev} + \Delta t \big( a(c) + \nu \lambda \odot A c \big) \Big), \qquad A = \Phi^\top \hat G . $$

Solver: burgers3d-retry `make_fsc` (cached-predictor fixed-sweep Levenberg–Marquardt; adaptive LM, budget 50, on the
first 3 steps, then one sweep per step; $g_{\rm tol} = 10^{-3}$; trust $0.05\times$ the coefficient RMS spread).
`offmesh.make_fsc_rule` is that function with the advection rule passed in; gate G4 checks it reproduces the vendor
function for the tensor rule.

## 3. Arms (identical at every mesh; $R' \in \{512, 256\}$; $\Delta t = 0.01$ everywhere)

The advection term $a(c) \in \mathbb{R}^M$ and its Jacobian are the only things that change. Linear terms ($A$,
$\lambda$, the projection of $u_0$, the decoding of outputs) stay the exact discrete operators of the mesh (Hari's
"hybrid").

| arm | advection $a(c)$ | role |
|---|---|---|
| `tensor` | $a_m = \tfrac12 c^\top T^{\rm sym}_m c$, $T_{m,ij} = \Phi_m^\top(\hat G_i \odot D^- \hat G_j)$ | the incumbent |
| `dense` | $\Phi^\top a_{\rm upwind}(\hat G c)$ on every interior node, Jacobian by JVP per column | mesh truth of the rule; 64³ all cases, 128³ first 4 cases, not at 256³ (cost) |
| `lat4096`, `lat8192`, `lat16384`, `lat32768` | off-mesh, CBC rank-1 lattice ($P_2$, weights 1), random shift seed 0 | the lattice ladder |
| `kor16381` | off-mesh, Korobov lattice, $n = 16381$ | one Korobov size |
| `gl16`, `gl24`, `gl32` | off-mesh, tensor Gauss–Legendre $16^3, 24^3, 32^3$ | Gauss ladder ($m$ = 4096, 13824, 32768) |
| `sob16384` | off-mesh, scrambled Sobol, seed 0 | the $1/m$ reference |
| `lat256` | off-mesh, CBC lattice, 256 points | **must-fail control** (far too few points) |
| `smol8` | off-mesh, Smolyak Clenshaw–Curtis level 8 (2559 points, negative weights) | **must-fail control** |

The off-mesh term (Hari's "point" form):

$$ a_m(c) = N^{3/2} \sum_{q=1}^{m} w_q\, \psi_m(x_q)\, u(x_q)\,\big(u_x + u_y + u_z\big)(x_q), \qquad
\psi_m(x) = 2^{3/2} \prod_{a} \sin(\pi k_{m,a} x_a), $$

with $u(x_q) = B_q c$, $B = \hat G(x_q)$, and $(u_x + u_y + u_z)(x_q) = D_q c$, $D = (\partial_x + \partial_y +
\partial_z)\hat G(x_q)$ computed exactly by one forward-mode JVP with tangent $(1,1,1)$. With $P_{qm} = N^{3/2} w_q
\psi_m(x_q)$, $a(c) = P^\top \big( (Bc) \odot (Dc) \big)$ and the Jacobian is
$J_u(c) = P^\top\big(\operatorname{diag}(Dc)\,B + \operatorname{diag}(Bc)\,D\big)$. Every rule is a symmetric quadratic
form in $c$, so $a(c) = \tfrac12 J_u(c)\,c$ and $J_u$ is linear in $c$; the cached predictor stays exact. (The dense
upwind rule is piecewise quadratic, so its predictor re-evaluates $J_u$ at each candidate.)

The rules are generated once (`rules.py`, NumPy/SciPy, Hari's generators copied unchanged) and committed as
`rules/rules.npz` (sha256 in `rules/rules.json`); the CBC vectors reproduce Hari's for $n = 4096$ and $32768$
($z = (1,1557,1741)$, $(1,12031,7247)$). Continuum target: Gauss $80^3$ (512 000 points); convergence check: Gauss
$64^3$.

**Important difference from the incumbent.** The off-mesh term is a quadrature of the continuum advection; the tensor
and the FOM use the mesh's first-order backward/upwind difference. They differ by the stencil's $O(h)$ consistency
gap. So an off-mesh solve is *not* expected to reproduce the tensor solve on the same grid; the comparison that
matters is against the refined reference, which neither discretisation matches exactly.

## 4. References

- **Same-grid reference** (as burgers3d-retry): Newton–BiCGStab at the mesh, $\Delta t = 0.005$, ntol $10^{-10}$,
  ltol $10^{-11}$, in the panel job.
- **Refined reference**: the same FOM at **513 nodes** (512³ cells, 133 M unknowns), $\Delta t = 0.0025$, computed once
  per cohort by `refjob.py`. Kept only on the **65-node lattice** $x = k/64$, $k = 1..63$ per axis (63³ nodes, exact
  nodes of every mesh of the lane, so no interpolation). Tolerances: chosen from the probe of section 9 as the loosest
  of $(10^{-6}, 10^{-7})$, $(10^{-8}, 10^{-9})$, $(10^{-10}, 10^{-11})$ whose lattice difference from the tightest is
  $\le 10^{-5}$ (relative to $\lVert u_0 \rVert$), provided a case costs $\le 240$ s on the GPU; if 513 nodes is
  infeasible in time or memory, the fallback is **385 nodes** ($\Delta t = 0.0025$), recorded as an amendment.
- The refined reference's own error is estimated by the probe (257-node same-grid reference against it on one case)
  and reported beside every refined number.

## 5. Metrics

Errors are worst over the evolved output times $t = 0.05..0.25$ of
$\lVert u(t) - u_{\rm ref}(t)\rVert_2 / \lVert u_{\rm ref}(0)\rVert_2$, then worst (and median) over the cohort.

- **PRIMARY**
  - `err_refined`: against the refined reference on the 65-node lattice. ROM fields there are $\hat G(x)\,c$ exactly
    (the lattice nodes are mesh nodes); FOM fields are the mesh solution at those nodes.
  - `dist_tensor` / `dist_dense` / `dist_conv`: distance of a rollout from the tensor rollout, the dense rollout, and the
    converged off-mesh rollout (`lat32768`) of the same $R'$, mesh and case, on the full grid, in the field metric
    $\lVert \hat G (c_1 - c_2)\rVert = \lVert L^\top (c_1 - c_2)\rVert$ ($L$ the Cholesky factor of the mesh Gram),
    normalised by $\lVert u_{\rm ref}(0)\rVert$.
- **SECONDARY**: `err_same_grid` on every interior node against the same-grid reference.
- **ρ** (paper eq. 13), $\rho = \lVert a_{\rm rule}(c) - a_{\rm target}(c)\rVert / \lVert a_{\rm target}(c)\rVert$
  over the arm's $M$ tests, on reached states: the tensor solve's accepted states on certification draws
  923811–923813 (8 cases each, 24 cases, 26 states per case at $\Delta t = 0.01$), worst / median / p90 over the
  evolved states $k \ge 1$ (headline) and separately at $k = 0$. Two targets: **continuum** (Gauss $80^3$ of the
  continuum term) and **mesh** (sign-upwind stencil on every interior node). Reported for every off-mesh rule, the tensor,
  and the dense stencil (whose continuum ρ is the stencil's $O(h)$ gap).
- **Mesh invariance**: cross-mesh distance of an arm's rollouts, $\max_t \lVert u_{n_1}(t) - u_{n_2}(t)\rVert /
  \lVert u_0\rVert$ on the 65-node lattice (same bank, same case), for consecutive meshes; computed offline from the
  saved coefficients.
- **Solver**: LM iterations per query (median, max), exit reasons (4 stationary, 0 non-stationary, 3 non-finite).
- **Cost**: ms per query (A–B–A, the retry timing contract, first 16 cases × 3 repetitions), ms per advection-Jacobian
  evaluation (microbenchmark, median of 50), bytes (tensor $8MR'^2$; off-mesh $8m(2R' + M)$) and flops per
  Jacobian. The FOM grid is timed in the same allocation: $\Delta t \in \{0.005, 0.01, 0.025\}$ ×
  ntol $\in \{10^{-2}, 10^{-3}\}$, ltol $0.1$ (contains the three settings the retry lane selected).
- **Speedups** against Newton–BiCGStab, two rules: (a) *same-grid rule* (the retry lane's paper rule): fastest FOM
  setting whose worst same-grid error ≤ the arm's; (b) *refined rule*: fastest FOM setting whose worst refined error ≤
  the arm's ("none" if no setting at that mesh is as accurate).

## 6. Gates (must pass at every mesh; a failed gate stops the verdict that depends on it)

| gate | test | bar |
|---|---|---|
| table gates | Gram condition, tensor vs direct backward advection (retry) | $\le 10^8$; $\le 10^{-10}$ |
| G1 | analytic $(1,1,1)$-derivative of $\hat G$ vs centred FD ($h = 10^{-5}$), 32 points | $\le 10^{-6}$ rel |
| G2 | off-mesh assembly fed the mesh nodes, weights $N^{-3}$, and $D^-$ in place of $D$ reproduces the tensor Jacobian (32 columns, 64 tests) | $\le 10^{-12}$ rel |
| G3 | $J_u$ formula vs `jax.jacfwd` of $a(c)$; $a = \tfrac12 J_u c$ | $\le 10^{-12}$ rel |
| G4 | `make_fsc_rule` with the tensor = vendor `make_fsc` (fields, coefficients) | $\le 10^{-12}$ rel |
| continuum target | ρ of Gauss $64^3$ against Gauss $80^3$, worst over states | $\le 10^{-5}$; otherwise continuum ρ below 10× its value is reported as unresolved |
| reference | same-grid reference residual | $< 10^{-9}$ |
| timing | drift A2/A1 within 1.10; neighbour ratios two-sided in $[1/1.10, 1.10]$; timed outputs equal the quick run ($\le 10^{-12}$) | retry R1-5 |

**Must-fail controls** (checked at one real mesh, 64³ validation, before any verdict is read, then at every mesh):

- `lat256`: continuum ρ worst $> 0.116$ and `dist_conv` worst $> 10^{-3}$.
- `smol8`: continuum ρ worst $> 0.116$ and `dist_conv` worst $> 10^{-3}$.
- The certificate ρ of the tensor against the mesh target must reproduce the retry lane's order ($\le 10^{-2}$); the
  dense stencil's continuum ρ must be $> 0.116$ at 64³ if the $O(h)$ gap is what separates the two families (if it is
  not, the "gap" explanation is withdrawn).
- Audit controls (NumPy audit): a swapped reference case and a 1 % perturbation of a recorded error must both be
  detected.

A control that does not fire on real data makes its gate non-discriminating; the verdict that rests on it is withheld
and the failure is reported.

## 7. Pre-registered rules (validation cohort 923801 × 64, per mesh and per $R'$)

- **Eligible off-mesh arm**: every output finite, no reason-3 exit, non-stationary exits $\le 1\%$ of all steps, and
  continuum ρ worst ($k \ge 1$) $\le 0.116$ at that mesh and $R'$. Controls are never selectable.
- **Converged off-mesh rollout** = `lat32768`, valid at a mesh/$R'$ only if `dist(gl32, lat32768)` worst
  $\le 10^{-3}$; otherwise `dist_conv` verdicts at that mesh are withheld.
- **Selected off-mesh setting** = the eligible non-control off-mesh arm with the smallest validation median ms whose
  worst `dist_conv` $\le 10^{-3}$ (quadrature error at most 0.1 percentage point of the field); if none qualifies,
  `lat32768`. Written to `selection.json` (sha256 in the lab log) before the held-out jobs are staged.
- **(i) lattice vs Gauss**: "the lattice beats Gauss" iff the continuum ρ worst of `lat4096` < `gl16` and of
  `lat32768` < `gl32` for both $R'$ at all three meshes; the medians and the ~16k pair (`lat16384`, `kor16381` vs
  `gl24`, unequal $m$) reported beside. "$m$ reaching the tensor": the smallest $m$ of the CBC ladder whose validation
  worst `err_refined` $\le$ the tensor's at that mesh/$R'$ (reported, with the smallest $m$ meeting `dist_conv`
  $\le 10^{-3}$).
- **(ii) reproduces / mesh-invariant**: reported as `err_refined(selected) − err_refined(tensor)` and `dist_tensor` per
  mesh; "mesh-invariant" iff the selected arm's held-out worst `err_refined` varies by at most 10 % (max/min $\le 1.10$)
  across 64³/128³/256³ at fixed $R'$; the tensor's ratio is reported beside it, with the cross-mesh distances.
- **(iii) cost**: per mesh/$R'$, ms of the selected arm vs the tensor (same allocation), Jacobian ms, bytes, flops, and
  both speedups (section 5).

## 8. Cohorts

| cohort | seed | size | use |
|---|---|---|---|
| validation | 923801 | 64 | rule selection (retry lane's validation cohort; the retry lane used it for its own settings) |
| certification draws | 923811, 923812, 923813 | 8 each | ρ on reached states |
| held-out | 923901 | 32 | opened once at the end with everything frozen |

The held-out cohort is the retry lane's sealed cohort. The retry lane opened it once for its two frozen *tensor*
settings; no off-mesh arm has ever run on it and no choice in this lane uses it. It is kept (rather than a fresh
seed) so the held-out numbers here are directly comparable with the incumbent's published row; this is stated beside
every held-out number.

## 9. Jobs (at most ~10, one directory each under the namespace)

| job | what | GPU |
|---|---|---|
| `smoke1` | `refjob.py --mode probe` (513 nodes, three tolerances, one probe case 923651; 257-node same-grid comparison) then `qpanel.py` at 65 nodes on 4 validation-seed-disjoint probe cases (923651) with every rule and gate, `stop_after_rho` false | H200 |
| `ref1` | refined references, cohorts 923801 × 64 and 923901 × 32 (held-out refs are written but read only by the held-out jobs) | H200 or A100-80G |
| `val65`, `val129`, `val257` | validation panels (sections 3, 5, 6) | H200 |
| `ho65`, `ho129`, `ho257` | held-out panels, frozen config | H200 |

Every panel runs on an H200 (the 257-node tables need ~80 GB; one GPU type keeps rows comparable with the retry
lane). If the panel finishes before the refined reference exists it waits up to 1.5 h, then defers the refined errors
to `refine_q.py` (NumPy, offline, from the saved coefficients and 65-lattice FOM fields); `refine_q.py` runs for every
panel regardless and must agree with in-job values to $10^{-10}$.

## 10. Audits and reporting

- `audit_q.py` (NumPy/SciPy, no JAX): refined and same-grid restricted errors recomputed from saved coefficients /
  fields with a NumPy bank (≤ 1e-10); ρ ingredients of saved states recomputed (off-mesh rule with a NumPy forward
  derivative, scipy DST mesh target, NumPy Gauss $64^3$ continuum target); the selection recomputed independently;
  the audit controls of section 6.
- Codex (`codex exec`) design audit of this file and the code before the experiment jobs; results audit of the final
  report. Both kept in `results/`.
- Report `reports/2026-10-01-burgers3d-offmesh-quadrature.md` on main, generated by a script from the run JSONs (no
  hand-typed numbers), title + status line, LaTeX, mermaid, glossary.

## R1 — revisions after the independent design audit (2026-10-01 ~23:59 EDT, before any GPU job; override §§4–10)

Codex audit `results/codex-design-audit-2026-10-01.md`: verdict "do not launch yet"; off-mesh mathematics, G2, the dense
Jacobian, distances and lattice indices CORRECT; 4 blockers, 6 majors, 5 minors. Disposition:

1. **Refined-reference memory (blocker) — fixed.** The vendor FOM stacks every time step (107 GB at 513 nodes) and
   captures the $(n-2)^3$ spectral array as a constant. `offmesh.make_fom_lean` keeps only the six output states
   (nested scans) and takes the 1D eigenvalues as an argument; same Newton/BiCGStab arithmetic. Gate (in `refjob.py`,
   every run): lean = vendor FOM at 65 nodes (fields $\le 10^{-13}$, Newton counts equal). The smoke job records the
   peak device memory at 513 nodes.
2. **Reference acceptance (blocker) — fixed.** A case is accepted only if finite, no Newton cap hit and every step's
   final relative residual $\le$ ntol; a cohort's `.done` marker (JSON: sha256, $n$, $\Delta t$, tolerances,
   accepted) is written only if every case is accepted. The panel asserts the marker's sha256 equals the file's, that
   $n$, $\Delta t$, tolerances equal `expected_ref` of its config, and that the initial fields agree with its own
   ($\le 10^{-12}$). Tolerances: set in `make_configs.py` from the smoke probe by the §4 rule (default
   $(10^{-8}, 10^{-9})$), recorded as R2 if changed.
3. **Selection fallback (blocker) — fixed.** No unconditional fallback: "no qualifying setting" is a possible outcome
   (`selected = null`). The converged rollout is valid only if `lat32768` and `gl32` are both eligible and their
   distance is $\le 10^{-3}$; otherwise no selection and `dist_conv` verdicts are withheld at that mesh/$R'$.
   **Exit classes:** reason 4 (stationary) and 1 (residual tolerance) count as converged; 0 (budget / one fixed sweep
   not stationary) and 2 (tiny step) count as non-stationary toward the 1 % limit; 3 (damping exhausted or non-finite)
   disqualifies. Ties in ms: lexicographic arm name.
4. **Pipeline completeness (blocker) — fixed.** `refine_q.py` (NumPy refined errors and cross-mesh distances),
   `select_q.py` (this section), `audit_q.py` (NumPy audit, independent selection) and `npcore.py` exist, and were
   checked locally on a CPU smoke at 33 nodes. Held-out configs are generated only once `selection.json` exists and
   carry its sha256; the panel refuses to start if the file differs.
5. **Timing (major) — fixed.** Empty neighbour groups are recorded as *untested* (not 1.0) and do not count as a pass of
   that check; the timed-output determinism check adds full-field checksums (sum and sum of squares, relative
   $\le 10^{-12}$) to the 16³ lattice comparison; microbenchmarks burn in 5 calls per arm and keep all 50 timings. Dense
   arms are not timed (their ms is from the quick runs, flagged). Total query cost stays grid-dependent (projection and
   decoding); only the advection term is mesh-independent — stated wherever cost is discussed.
6. **G4 (major) — strengthened.** Both deployed ranks; fields and coefficients $\le 10^{-12}$ and iteration counts,
   exit reasons and rejections identical — all asserted.
7. **Certification rollouts (major).** Their finiteness and exit counts are recorded; a non-finite one aborts.
   Restatement: ρ is measured on *tensor-reached* states, which certifies the rules on those states, not on each
   off-mesh solve's own trajectory. The test space differs slightly by mesh ($M = 2052/1027$ at 65 nodes,
   $2049/1024$ at 129); rule comparisons are within a mesh.
8. **Unresolved ρ (major).** If both members of a lattice/Gauss pair have continuum ρ below 10× the Gauss
   $64^3$-vs-$80^3$ check value, that comparison is reported as unresolved, not as a win.
9. **Controls (major) — restated.** `lat256` / `smol8` are *empirical* bad-rule expectations, not fault injections; if
   one does not fail on real data that is reported as a finding about the problem's sensitivity, and the
   discriminating power of the corresponding threshold is questioned in the report. The genuine fault-injection controls
   are the audit's swapped reference and perturbed record. The dense-continuum-ρ check tests only whether the
   continuum/upwind gap exceeds 0.116; a smaller value does not disprove an $O(h)$ gap.
10. **(ii) restated.** "Mesh-invariant" is judged on the cross-mesh field distance of the same arm (not only on equal
    worst errors) plus the 1.10 ratio of worst refined errors; the tensor's values are reported beside. If the selected
    arm differs by mesh, invariance is reported for each fixed arm and for the selection policy separately.
11. **Cohort wording (restatement).** The held-out cohort 923901 is "evaluated once by this lane after the selection is
    frozen"; it is not a fresh unopened set (the retry lane evaluated its tensor settings on it). The same-job tensor
    baseline at $\Delta t = 0.01$, $R' \in \{512, 256\}$ is distinguished from the historical published row, whose
    64³/128³ settings used $\Delta t = 0.005$ and $R' = 192$.
12. **Minors.** Smolyak weights sum to 0.99753 (boundary points dropped), not one; they are not renormalised. One shift /
    scramble seed per rule: results are for those rules, not a distribution over shifts; one Sobol size cannot show a
    rate. The 257-vs-513 probe is a refinement discrepancy on one case, not an error bound for the reference.
