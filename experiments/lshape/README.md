# lshape — Poisson on the L-shaped domain (paper table T18)

Pre-registration and every criterion: `DESIGN.md`. Report and machine-readable rows are
generated into `reports/` by `reports/generate_lshape.py`; no number is typed by hand.

## Layout

| file | role |
|---|---|
| `lsh_core.py` | geometry, 5-point operator (two assemblies), SuperLU / GPU-CG / ILU-PCG solvers, boundary factors, singular columns, banks, floors, oracles, weak operators, POD |
| `lsh_fit.py` | joint / bank / head training phases (the parent recipe with the factor injected) |
| `lsh_train.py` | job 1: gates, cohorts, bank sweep, heads, correction bases |
| `lsh_solve.py` | jobs 2–4: the frozen solve, one or more meshes, every subject timed in one process |
| `lsh_audit_np.py` | independent NumPy/SciPy audit (imports neither JAX nor a driver) |
| `gate_square.py` | G-FOM-6: reproduces the parent's audited square bank floors with this cell's machinery |
| `cluster/stage.py`, `collect.py`, `preserve_archive.py`, `make_models.py` | staging, checksum collection, Git archiving, model list for the solve jobs |
| `config-train.json`, `config-solve.json` | every knob |

## Local smoke (N = 32, sub-minute, through the shared slot helper)

```bash
cd worktrees/2026-09-17-lshape
source /etc/profile.d/jax-mem.sh
export PYTHONPATH=experiments/separable-decoder:experiments/head-ablation:experiments/mr-burgers2d:experiments/lshape
PY=/home/tahmid/Dev/.venv/bin/python
JAX_DEFAULT_MATMUL_PRECISION=highest <slot-helper> jaxrun $PY experiments/lshape/lsh_train.py \
  --config experiments/lshape/config-train.json --out <scratch>/train --smoke
$PY experiments/lshape/cluster/make_models.py <scratch>/train --stage <scratch>/staged
JAX_DEFAULT_MATMUL_PRECISION=highest <slot-helper> jaxrun $PY experiments/lshape/lsh_solve.py \
  --config experiments/lshape/config-solve.json --models <scratch>/staged/models.json --out <scratch>/solve --smoke
```

## Cluster

```bash
$PY experiments/lshape/cluster/stage.py train lsh01 --hours 10
rsync -a experiments/lshape/runs/lsh01/ tufts-login:/cluster/tufts/paralab/tawal01/lshape_20260917/lsh01/
ssh tufts-login 'squeue -u tawal01; cd /cluster/tufts/paralab/tawal01/lshape_20260917/lsh01 && sbatch run.sbatch; squeue -u tawal01'
$PY experiments/lshape/cluster/collect.py lsh01
$PY experiments/lshape/lsh_audit_np.py train experiments/lshape/runs/lsh01/archive/output --out experiments/lshape/runs/lsh01/audit.json
$PY experiments/lshape/cluster/preserve_archive.py lsh01
# solve jobs: stage the head checkpoints produced by lsh01
$PY experiments/lshape/cluster/make_models.py experiments/lshape/runs/lsh01/archive/output --stage experiments/lshape/checkpoints/lsh01
$PY experiments/lshape/cluster/stage.py solve lsh02 --intervals 64 128 --extra experiments/lshape/checkpoints/lsh01/*
```
