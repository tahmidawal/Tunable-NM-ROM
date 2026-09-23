# poisson-bank-knob — progress (kept current)

- Quick local 1024² ladder: `QUICK.md` (commit ad71eba1). Parity 1.4e-13 / 5.2e-13; NumPy audit PASS.
- Cluster: `pbkC` = job 4197114 (256², 1024², 2048², H200); `pbkD` = job 4197115 (4096², H200). Namespace
  `/cluster/tufts/paralab/tawal01/pbank_20260923/`. The earlier 4196680/4196682 were cancelled while pending (A1).
- Branch is **local only**: the parent history carries 69 GB of unpushed blobs (mr-heat2d archive parts).

## `pbk_core.py` interface (stable; sibling lane pbank3_* reuses it)

`make_rotation(params, codes, train_draws, intervals) -> (T, L, info)`;
`build_banks(params, n, rows_per_chunk, eval_rows, maxmode, Tm, edges, truth, keep_orig, keep_rot32) -> dict(orig, rot, rot32, T, S, Rg, info, floors)`;
`decode_blocks(rot, a, n, nb)`; `nblocks(edges, Rp)`; `make_trunc_lean(ops, engine, cfg, count, nb, n)` + `trunc_query(...)`;
`make_trunc_linear(Bprime, n, nb)` + `trunc_linear_query(...)`; `arm_q_max(Rp, K, cap=256)`.
These are 2D-square specific: the sine-product contraction and the bank evaluation on the (n-1)² grid.
Nothing has changed since commit ad71eba1.
