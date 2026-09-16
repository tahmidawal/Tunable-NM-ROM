# The prior dial

Is "trust in the neural prior" a usable inference-time accuracy/cost knob on one frozen
checkpoint? Head ablation asked whether the nonlinear head earns its place; the correction
ladder asked what extra *parameterization* buys. This cell changes neither: the full bank
coefficient vector is free at every setting, and the only knob is how hard the solver is
pulled back towards the head's own prediction.

$$\min_{z\in\mathbb R^{K},\,c\in\mathbb R^{R}}\ \big\|r_w(c)\big\|_2^2\;+\;\lambda\,\big\|R_G\,(c-h_\theta(z))\big\|_2^2$$

$\lambda\to\infty$ is head-ablation arm (a) exactly; $\lambda\to0$ with $M\ge R$ is the
free-bank arm (d). The predeclared protocol, the declared scaling of $\lambda$, the gates and
the pre-registered acceptance criteria are in [DESIGN.md](DESIGN.md); project state stays in
the canonical [lab log](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md).

## Layout

| file | what it is |
| --- | --- |
| `prior.py` | the penalized weak residual, the block trust region, the complete Burgers query |
| `dial.py` | the Burgers 2D driver |
| `poisson_dial.py` | the Poisson 2D driver; $c$ is eliminated exactly from one thin SVD |
| `smoke_prior.py` | the local fidelity gate and the $\lambda$-grid/metric probe |
| `audit_dial.py`, `audit_poisson_dial.py` | independent NumPy audits; no JAX, no GPU |
| `reports/generate_prior_dial.py` | the one report for both PDEs, generated from the raw JSON |
| `cluster/` | staging, checksum collection and the Git-trackable chunked archive |
| `checks/` | retained smoke and audit evidence |
| `artifacts/<attempt>/` | the checksum-verified raw job archive, in bounded chunks |

`runs/` is Git-ignored scratch; `artifacts/` is the retained record.

## The one thing to know before reading any number

The initializer is arm (a)'s and starts the correction at $y=0$, so the $t=0$ output field is
the head's compression of the supplied field and **cannot** depend on $\lambda$. On these
cases that compression error is also the largest of the six output times, so the worst
same-grid error over *all* times is pinned by construction. The report keeps that as the
pre-registered primary metric and reports the worst over the **evolved** times beside it.

## Gates

1. `smoke_prior.py` runs $\lambda=\infty$ **through the new code path** on the archived
   operators: it must match the consolidated saved Burgers case to $10^{-12}$ relative and
   agree with the incumbent `accuracy_paths.make_rom` to $10^{-12}$. It is in fact
   bit-identical.
2. In the Burgers job, `M64_eq_laminf` must reproduce `abl01`'s `a_neural_eq` to $10^{-9}$.
3. In the Poisson job, `laminf` must reproduce `pabl01`'s `a_neural` to $10^{-9}$.
4. The NumPy audits recompute every reported error from the saved output fields.

```bash
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" experiments/prior-dial/smoke_prior.py \
  experiments/prior-dial/checks/smoke-prior.json
```

## Running a cluster attempt

One job per directory, `gpu` partition only, and the batch script asserts `jax_backend=gpu`
before doing any work. Check `squeue` before and after every submission.

```bash
PY=/home/tahmid/Dev/.venv/bin/python
"$PY" experiments/prior-dial/cluster/stage_burgers.py <attempt>   # nested layout
"$PY" experiments/prior-dial/cluster/stage_poisson.py <attempt>   # flat layout
scp -r experiments/prior-dial/runs/<attempt> \
  tufts-login:/cluster/tufts/paralab/tawal01/prior_dial_20260915/
ssh tufts-login "cd /cluster/tufts/paralab/tawal01/prior_dial_20260915/<attempt> && \
  sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
```

Collect, audit, archive, then delete only that exact remote attempt directory:

```bash
"$PY" experiments/prior-dial/cluster/collect.py <attempt>
"$PY" experiments/prior-dial/audit_dial.py \
  experiments/prior-dial/runs/<attempt>/archive/output/result.json \
  --out experiments/prior-dial/checks/<attempt>-audit.json \
  --fields experiments/prior-dial/runs/<attempt>/archive/output
"$PY" experiments/prior-dial/cluster/preserve_archive.py <attempt>
ssh tufts-login "rm -rf /cluster/tufts/paralab/tawal01/prior_dial_20260915/<attempt>"
```

## Plain-language glossary

- **Prior:** the trained head's prediction of the bank coefficients. The dial decides how
  much the online solver is allowed to disagree with it.
- **$\lambda_{\rm rel}$:** the dimensionless knob. Large means "trust the head"; small means
  "trust the equations".
- **Bank:** the frozen set of spatial fields the reduced state is built from.
- **Test modes ($M$):** the smooth functions the PDE residual is averaged against. With the
  bank free, $M<R$ means fewer equations than unknowns — *underdetermined*, and not the free
  bank however small $\lambda$ gets.
- **Same-grid error:** difference from the converged full-order solve on the same mesh, which
  isolates reduction error from discretization error.
- **Completed / early-stopped:** the shared stopping rule terminated everywhere / it did not.
  An early-stopped point is never relabelled as a converged one.
- **Non-dominated:** a setting no other setting beats on both error and cost at once.
- **Development / final cohort:** cases usable for method selection / cases kept unopened.
