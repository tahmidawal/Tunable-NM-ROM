# Training the Burgers head

Can *training alone* — same separable architecture, same frozen bank — move the Burgers head's
best-found reconstruction from the incumbent's 2.5447 % toward the bank's 0.3918 % projection
floor at 256 intervals, and does the online solve follow? The predeclared protocol is
[DESIGN.md](DESIGN.md); project state stays in the canonical
[lab log](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md).

Three levers, each isolated with the others at the incumbent recipe, then the best combination:
data density, training objective, and capacity ($K$, and conditionally the bank rank $R$).
Nothing about the architecture changes — no mixtures, no FiLM, no new feature families.

## Layout

| file | what it is |
| --- | --- |
| `common.py` | the frozen bank's algebra, the streaming data generator and projector, and the ROM's weak residual written in the whitened head parameterisation |
| `train.py` | the training driver: every arm, the pre-registered selection, and the emitted frozen checkpoints |
| `evaluate.py` | the evaluation driver; every arm runs through the UNCHANGED `head-ablation/arms.py` machinery, with the incumbent as arm (a) in the same job |
| `smoke_train.py`, `smoke_eval.py` | local fidelity smokes |
| `audit_train.py`, `audit_eval.py` | independent NumPy audits; no JAX, no driver import |
| `reports/generate_b_head_train.py` | the one report, generated from the audited JSONs |
| `cluster/` | staging, checksum collection and the Git-trackable chunked archive |
| `checks/` | retained smoke and audit evidence |
| `artifacts/<attempt>/` | the checksum-verified raw job archive, in bounded chunks (the trained checkpoints are inside it) |

`runs/` is Git-ignored scratch for staging and collection; `artifacts/` is the retained record.

## Gates

Nothing is believed before all of these pass.

1. `smoke_train.py`: the whitening reparameterisation is exact; identity $(\ast)$ holds against
   fields the FOM actually produced; `common.weak_residual` **is** `arms.weak_dense`; and
   `train.fit_ext` at $\beta=\gamma=0$ reproduces `sep_hfit.fit` step for step.
2. `smoke_eval.py`: the incumbent through this cell's evaluation path reproduces the consolidated
   saved Burgers case and agrees with `accuracy_paths.make_rom`.
3. In the evaluation job, `incumbent_eq` must reproduce abl01's `a_neural_eq` on all six cases to
   $10^{-9}$ relative.
4. `audit_eval.py` recomputes every error and every same-grid discrepancy from the saved fields.

```bash
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" experiments/b-head-train/smoke_train.py \
  experiments/b-head-train/checks/smoke-train.json
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" experiments/b-head-train/smoke_eval.py \
  experiments/b-head-train/checks/smoke-eval.json
```

## Running a cluster attempt

One job per directory, `gpu` partition only, A100 (excluding `pax007`), and the batch script
asserts `jax_backend=gpu` before doing any work. Check `squeue` before and after every submission.

```bash
PY=/home/tahmid/Dev/.venv/bin/python
NS=/cluster/tufts/paralab/tawal01/b_head_train_20260916

# 1. training
"$PY" experiments/b-head-train/cluster/stage_train.py train01
scp -r experiments/b-head-train/runs/train01 tufts-login:$NS/
ssh tufts-login "cd $NS/train01 && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
"$PY" experiments/b-head-train/cluster/collect.py train01
"$PY" experiments/b-head-train/audit_train.py \
  experiments/b-head-train/runs/train01/archive/output/result.json \
  --checkpoints experiments/b-head-train/runs/train01/archive/output \
  --out experiments/b-head-train/checks/train01-audit.json

# 2. evaluation, from the checkpoints that training emitted
"$PY" experiments/b-head-train/cluster/stage_eval.py eval01 \
  experiments/b-head-train/runs/train01/archive/output
scp -r experiments/b-head-train/runs/eval01 tufts-login:$NS/
ssh tufts-login "cd $NS/eval01 && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
"$PY" experiments/b-head-train/cluster/collect.py eval01
"$PY" experiments/b-head-train/audit_eval.py \
  experiments/b-head-train/runs/eval01/archive/output/result.json \
  --fields experiments/b-head-train/runs/eval01/archive/output \
  --train experiments/b-head-train/runs/train01/archive/output/result.json \
  --out experiments/b-head-train/checks/eval01-audit.json

# 3. preserve and clean up the exact remote attempt directories
"$PY" experiments/b-head-train/cluster/preserve_archive.py train01
"$PY" experiments/b-head-train/cluster/preserve_archive.py eval01
ssh tufts-login "rm -rf $NS/train01 $NS/eval01"
```

## Plain-language glossary

- **Bank:** the fixed set of spatial fields the reduced state is built from; here a coordinate
  network cached as a matrix on the grid.
- **Head:** the small network turning the solved coordinates into bank coefficients — the object
  being trained.
- **Auto-decoder codes:** one latent vector per training snapshot, optimised jointly with the head
  instead of produced by an encoder.
- **Span floor:** the best any coefficients at all could do in that bank; a floor no head can beat.
- **Best-found reconstruction:** the best that checkpoint's own manifold can do on a reference
  field with no PDE involved.
- **Held-out representation oracle:** the same quantity on trajectories no arm ever fitted; the
  only selection statistic, declared before any run.
- **Same-grid error:** difference from the converged full-order solve on the same mesh, so it
  contains no discretisation error.
- **Frozen-bank vs joint arm:** a frozen-bank arm retrains only the head and the codes, so every
  such arm shares one span floor; a joint arm also moves the bank and has its own floor.
