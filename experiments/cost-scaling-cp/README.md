# Online-cost scaling: the mesh is not in the formula (2026-08-13)

Hari's question: when the ROM solves a PDE, does the online work depend on the
mesh size n, or only on the latent dimension k? Answer, measured two ways on
one A100: **only on k.** The figure is `cost_scaling.png` / `.pdf`.

Two arms, both timed on the same GPU (pax106, A100 80GB PCIe):

- **ViT-CP ROM (this directory)** — the repo's actual online solve path
  (`poisson/src/.../solver/nm_rom.py`), called directly with random weights of
  the correct shapes (cost depends on shapes, not values). See `AUDIT.md`:
  the GN residual/Jacobian flow through a precomputed `(rank, m*5)` EQ-stencil
  gather; the full n-dim decode happens once, after the solve, outside the loop.
- **Coord-net ROM skeleton** (sibling worktree
  `2026-08-13-cost-scaling-coordnet`) — the GN/EQ arithmetic the future
  coordinate-decoder ROM will do: FD stencil residual at m EQ nodes via 5m
  pointwise decoder evaluations, jacfwd over z, k x k LM solve. n-freedom is
  asserted in code (no online tensor shape contains n).

## The evidence

**Wall-clock flatness across n (k fixed):** per-GN-iteration medians vary by
at most 2.1% (CP) and 0.44% (coord-net) across N = 32 -> 512 — a 256x range
in node count — at every k in {2,4,8,16,32}. Meanwhile the FOM CG solve grows
~n^0.94 over the same range (0.20 ms -> 31.1 ms).

**Compiler FLOPs (the stronger form):** XLA `cost_analysis()` FLOPs per
compiled GN iteration are bit-identical across all N at every fixed k, in both
arms. n appears in no online tensor shape, so the compiled program cannot do
n-dependent work. (Footnote: the CP arm's FLOP counts at k=2 vs k=4 are
non-monotone — 3.12e5 vs 1.67e5 — an XLA fusion-accounting artifact; wall
time is monotone in k. Coord-net FLOPs are monotone: 8.4e8 -> 8.9e9.)

**k-dependence (N=512):** per-iteration time grows k^0.12 (CP) / k^0.48
(coord-net) over k = 2 -> 32. Exponents are shallow because at these sizes an
A100 is latency-dominated, not FLOP-dominated; the claim that matters is that
the cost curve rides on k while n never enters.

| arm | k=2 | k=4 | k=8 | k=16 | k=32 (us/GN-iter, N=512) |
|---|---|---|---|---|---|
| ViT-CP ROM | 155 | 168 | 178 | 191 | 219 |
| coord-net ROM | 220 | 257 | 342 | 502 | 842 |

CP absolute times are lower because its online decoder is an MLP + tiny gather
against precomputed factors (m=640, rank=128); the coord-net evaluates a 5x256
FiLM trunk at 5m points (m=100). Both are 4-6 orders below the FOM at N=512.

## Protocol (defensibility)

- Fixed GN_ITERS=10 (no early stopping); JIT compiled once per cell, >=3
  warm-up solves excluded; median + IQR over 30 timed solves with
  `jax.block_until_ready`; `JAX_DEFAULT_MATMUL_PRECISION=highest`;
  `jax_backend=gpu` verified in-log (preflight + script).
- All 25 cells of each arm ran SEQUENTIALLY inside one Slurm job on one A100
  (cross-N comparisons must share a device); both arms landed on the same node
  (pax106). Jobs: 2345393 (coord-net), 2345628 (CP; 2345392 died on the
  then-missing flax in the cluster venv — fixed with --no-deps installs of
  flax 0.12.8 + msgpack/typing_extensions/rich/pyyaml, jax verified unchanged
  at 0.10.2).
- EQ budget fixed per arm across all cells (CP m=640 = poisson configs'
  min_eq_points; coord-net m=100). GPU: NVIDIA A100 80GB PCIe. Local GB10
  smoke JSONs are kept for provenance but excluded from the figure.
- Raw data: `cp_timing_pax106.json` (includes per-N FOM medians and the
  one-shot decode times), sibling `coordnet_timing_pax106.json`.

## Files

| path | what |
|---|---|
| `AUDIT.md` | shape-level audit of the repo's online solve path (m-local verdict) |
| `time_cp_rom.py` | timing harness calling the real solver |
| `cp_timing_pax106.json` | cluster sweep (25 cells + FOM series), the figure's CP data |
| `figure_cost_scaling.py` | builds `cost_scaling.png`/`.pdf` from both arms' JSONs |
| `smoke/` | local GB10 smoke artifacts |
