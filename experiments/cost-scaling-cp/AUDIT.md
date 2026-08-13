# Audit: is the ViT-CP NM-ROM online solve m-local or n-touching?

**Verdict: m-local.** The Gauss-Newton iteration touches no tensor whose shape
contains n = N^d. n enters the online path only in two places outside the
iteration loop (final decode, F_eq preparation), both avoidable or one-shot.
The measured per-iteration cost should therefore be flat in N, and the XLA
FLOP count identical across meshes at fixed k. The timing harness verifies
this empirically.

## The online path (poisson package)

Entry: `NMROMSolver.solve(F_eq)` in `poisson/src/tunable_rom_poisson/solver/nm_rom.py`.

Shapes flowing through one GN-LM iteration (`body`, nm_rom.py:76-93), with
k = latent_dim, R = rank (128), H = hidden (256), m = n_eq EQ nodes, s = 2d+1
stencil width (5 in 2D):

| op | code | shapes |
|---|---|---|
| decoder MLP | `_mlp_apply` nm_rom.py:50-57 | z(k) -> W1(k,H) -> W2(H,H) -> W_rank(H,R) + skip W_direct(k,R) -> h(R) |
| stencil values | `u_at_stencil` nm_rom.py:59-63 | h(R) @ v_eq_st(R, m*s) -> (m, s) |
| residual | nm_rom.py:65-69 | (m,) arithmetic; dx is a scalar |
| Jacobian | nm_rom.py:79 `jax.jacfwd` | J (m, k) — k forward-mode passes through the above |
| normal eqs | nm_rom.py:80-84 | JtW(k,m) @ J(m,k) -> H(k,k); solve k x k |
| line search | nm_rom.py:86-91 | 4 residual evals, each (m,) |

The largest online tensor is `v_eq_st` with R * m * s entries (128 * 640 * 5
~ 4.1e5 floats) — independent of N. **No online tensor has n = N^2 entries.**

`v_eq_stencil` is precomputed OFFLINE by `build_v_eq`
(`poisson/src/tunable_rom_poisson/eq/nnls.py:44-60`): it gathers the CP factor
columns W_x[:, ix] * W_y[:, iy] at the m*s stencil nodes. That gather is
O(R*m*s) work on (R,N) factors — done once per model, not per solve.

## Where n DOES appear online (both outside the GN loop)

1. **Final decode** — `solver.decode(z)` (nm_rom.py:106-109) materialises the
   full field via `einsum('r,ri,rj->ij', h, W_x, W_y)`: O(R * N^2), once per
   solve, after the latent iteration converges, and only if the caller wants
   the full field (run_rom.py does, to compute rel-L2 against the FOM). The
   *latent solve itself* never calls it. The harness times it separately.
2. **F_eq preparation** — `run_rom.py:86-88` builds the full source field
   `F_full` (N^2) and gathers `F_full[eq_flat]`. Implementation convenience:
   the source is analytic and could be evaluated at the m nodes directly.
   One-shot per parameter, not per iteration; excluded from solve timing
   (the solver receives F_eq, shape (m,)).

Also notable: **the encoder is never used online.** Poisson's solve is
cold-start z=0 (nm_rom.py:71-74); run_rom.py never calls encode. The ViT
encoder is a training-time object.

## Offline stages that ARE n-dependent (allowed)

- EQ selection: NNLS design matrix over all strictly-interior nodes,
  (n_samples, (N-2)^2) (`nnls.py:compute_eq_weights`).
- `build_v_eq` gather (above), decoder training, data generation.

## Heat package

Same structure (`heat/src/tunable_rom_heat/solver/nm_rom.py`): residual and
Jacobian at EQ stencil nodes via precomputed `v_eq_stencil (rank, n_eq*s)`
(lines 33, 70, 85-86, 107), jacfwd -> (n_eq, k), k x k solve. The time loop
adds a factor num_steps but no n. Warm-started instead of cold-started.

## Harness design notes (time_cp_rom.py)

- Uses the REAL `NMROMSolver` + `build_v_eq` + `LinearCPDecoder` +
  `PoissonFOM.cg_solve` from poisson/src — no re-implementation.
- Random decoder weights of the exact architecture shapes (cost depends on
  shapes, not values). Config template: poisson2d_n64.yaml (rank=128,
  hidden=256); N and k overridden per cell.
- EQ nodes: m fixed random strictly-interior nodes with uniform weights.
  The repo's NNLS is offline and selects a data-dependent m >= min_eq_points
  (=640 in the poisson2d configs); for a cost measurement m must be HELD
  FIXED across cells, so the harness pins m = 640 (M_EQ env). The online
  code path is identical either way. (The directive's 100/64 defaults are
  the heat configs; poisson's are 200/640 — went with poisson's 640.)
- Fixed iteration count: gn_rel_tol=0.0 makes the while_loop's cond depend
  only on itr < gn_max_iters (=GN_ITERS), no repo-code modification.
  Iterations-to-tolerance with random weights is not meaningful and is not
  recorded.
- FLOPs: `jax.jit(solver.solve).lower(F_eq).compile().cost_analysis()` — the
  compiler's count for the whole fixed-iteration solve; divided by GN_ITERS
  it is reported per iteration (includes the one-time init residual/J0, a
  constant offset identical across cells).
- The final decode and the FOM CG solve are timed as separate series (the
  expected O(n) contrast lines).
