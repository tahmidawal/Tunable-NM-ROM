# b-ladder-top — the top of the correction ladder, and the combined envelope

Two questions on the frozen Burgers checkpoint `sep_hfit_dense_mid_N256_dense.pkl`
(SHA256 `18f0266ae6f0…`, $K=16$, $R=512$) at 256 intervals, on the same six opened
development cases the head ablation, the audited ladder (job `3713867`) and the
cheap-corrections cell (job `3734098`) used.

**Q1.** Make the $q=256$ rung converge. Three isolated fixes: a coarse-to-fine cascade
warm start, column equilibration of the augmented Jacobian, and a damping/trust schedule
for the correction block decoupled from the latent block.

**Q2.** Price the whole inference-time envelope in one job: the ladder × quadrature ×
evolution tolerance, against same-job full-order controls, POD-LSPG, and the trained FNO.

The predeclared protocol, the equations, the routine calls and the amendments are in
[`DESIGN.md`](DESIGN.md). Read it before the report.

## Layout

| path | what it is |
| --- | --- |
| `DESIGN.md` | predeclared design, gates, acceptance, falsification, routine calls, amendments |
| `topfix.py` | the three fixes, isolated; `base` is the cheap-corrections block solver itself |
| `q1_top.py` | Q1 driver: the convergence sweep, the conditioning probe, the per-rung quadrature |
| `q2_envelope.py` | Q2 driver: the ladder, POD-LSPG, the FOM controls, the FNO cohort |
| `fno_panel.py` | the FNO timing phase, the no-audit lane's `timing.py` protocol replicated |
| `smoke_top.py` | local smoke; sub-mesh, small $q$ |
| `audit_top.py` | independent NumPy audit; no JAX, no GPU |
| `config-q1.json`, `config-q2.json` | the two sweeps |
| `cluster/` | staging, collection and archive-chunking helpers |
| `checks/` | smoke and audit JSONs |
| `artifacts/` | checksum-collected raw archives as bounded Git chunks |
| `reports/` | the source-generated report, its figure and its generator |

## Reproducing

```bash
# local smoke (GB10)
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun /home/tahmid/Dev/.venv/bin/python \
  experiments/b-ladder-top/smoke_top.py experiments/b-ladder-top/checks/smoke-top.json

# stage, submit, collect, audit (Tufts; namespace b_ladder_top_20260916)
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/b-ladder-top/cluster/stage_q1.py btq101
rsync -a experiments/b-ladder-top/runs/btq101/ \
  tufts-login:/cluster/tufts/paralab/tawal01/b_ladder_top_20260916/btq101/
ssh tufts-login 'cd /cluster/tufts/paralab/tawal01/b_ladder_top_20260916/btq101 && sbatch run.sbatch'
$PY experiments/b-ladder-top/cluster/collect.py btq101
$PY experiments/b-ladder-top/audit_top.py \
  experiments/b-ladder-top/runs/btq101/archive/output/result.json --mode q1 \
  --fields experiments/b-ladder-top/runs/btq101/archive/output \
  --out experiments/b-ladder-top/checks/btq101-audit.json
$PY experiments/b-ladder-top/cluster/preserve_archive.py btq101
```

`stage_q2.py` is the same shape and additionally stages the trained FNO checkpoint and
its three architecture modules out of band from the `2026-09-14-no-audit` worktree; those
files are under a `.gitignore`d tree there and have no Git object, so their SHA256 and the
source worktree's HEAD are recorded in the attempt's `PROVENANCE.json` instead.

## Restoring an archive

Each `artifacts/<attempt>/` holds the verified `collection.tar.gz` as 48 MB chunks plus
`archive.json`, `SHA256SUMS` and the uncompressed `result.json`. See the `README.md`
written beside the chunks for the restore commands.
