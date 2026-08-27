# inherited_qf — frozen separable-Poisson decoders for the quadrature-free cell

Byte-copies of the frozen checkpoints consumed by `sep_poisson_qf.py`
(exp/2026-08-27-b1d-poissonqf, part B: quadrature-free exact weak residual at
N=128/256/512). **No training happens anywhere in this cell** — these decoders
are used exactly as trained by their source runs. `CHECKSUMS.sha256` covers all
three files.

| file | provenance (source of the byte-copy) | trained by | cluster job |
|---|---|---|---|
| `sep_poisson_N128_K16_R96.pkl` | `worktrees/2026-08-23-sepdec-n128/experiments/separable-decoder/runs/sepdec_n128_j1/out/` | `sep_poisson.py` (N=128, K=16, R=96, M=64, m=256, steps=60000, seed 0, fresh_seed 1, arch n_ff=128) | 2825804 (A100, pax105) |
| `sep_poisson_N256_K16_R64.pkl` | in-tree `runs/n256_j1/out/` (this worktree, inherited from exp/2026-08-23-sepdec-n256 via the nodes-mm base branch) | `sep_poisson.py` (N=256, K=16, R=64, M=64, m=256, steps=100000, seed 0, fresh_seed 777, default arch) | 2825729 (A100, pax106) |
| `sep_poisson_N512_K16_R64_nff128_ffs4.pkl` | `worktrees/2026-08-23-sepdec-n512/experiments/separable-decoder/runs/sepdec_n512_j1/out/` | `sep_poisson.py` (N=512, K=16, R=64, M=64, m=256, steps=100000, seed 0, fresh_seed 20260823, n_ff=128 ff_scale=4) | 2825500 (A100, pax106) |

Every checkpoint pickle stores `params` (decoder weights), `Z_tr` (512 training
latents) and `cfg` (the full training configuration, echoed into the QF cell's
output JSON). All are K=16, M=64 (weak alpha=1 modes), incumbent EQ budget
m=256 = 4M.

The N=64 smoke checkpoint used by the local smoke test is the in-tree
`runs/sepdec_r1/out/sep_poisson_N64_K16_R64.pkl` (job 2802238, H200) and is not
duplicated here.
