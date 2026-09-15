# Head ablation at matched latent dimension

Does the nonlinear coefficient map earn its place at one frozen spatial bank? This cell
answers reviewer 5mgh's demand for a linear POD-Galerkin/DEIM baseline with the same knobs,
and the follow-up objection that the rebuttal's wins came from a frozen POD basis plus the
solver rather than from the manifold. The predeclared protocol is [DESIGN.md](DESIGN.md);
project state stays in the canonical
[lab log](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md).

Every arm writes the reduced state as $u(z)=B\,h(z)$ and solves the same overdetermined weak
residual for $z$. Only the pair $(B,h)$ changes: (a) the retained neural head, (b) a linear
map, (c) a quadratic manifold, (d) unrestricted bank coefficients, (e) a classical POD-LSPG
basis at a ladder of ranks. Test modes, time discretization, initializer policy, stopping
rule and output contract are shared.

## Layout

| file | what it is |
| --- | --- |
| `arms.py` | banks, coefficient maps, map fitting, reduced operators, the shared solver and the complete query |
| `ablation.py` | the Burgers 2D driver |
| `poisson_ablation.py` | the Poisson 2D driver; the Poisson weak residual is exactly $Bh(z)-f_m$ |
| `smoke_arms.py`, `smoke_poisson.py` | local fidelity smokes, under a minute per process |
| `audit_ablation.py`, `audit_poisson.py` | independent NumPy audits; no JAX, no GPU |
| `reports/generate_head_ablation.py` | the one report for both PDEs, generated from the raw JSON |
| `make_lab_entry.py` | generates the lab-log entry from the audited JSONs |
| `cluster/` | staging, checksum collection and the Git-trackable chunked archive |
| `checks/` | retained smoke and audit evidence |
| `artifacts/<attempt>/` | the checksum-verified raw job archive, in bounded chunks |

`runs/` is Git-ignored scratch for staging and collection; `artifacts/` is the retained record.

## Gates

Nothing is believed before both of these pass.

1. `smoke_arms.py` part 1 replays the consolidated saved Burgers case through the *generic*
   arm machinery with the archived operators. It must match the saved fields to $10^{-8}$
   relative and agree with the incumbent `accuracy_paths.make_rom` to $10^{-12}$, so arm (a)
   is the retained solver rather than a re-implementation of it.
2. In the Burgers job, `a_neural_eq` must reproduce the retained multiresolution campaign's
   `frozen_stationary` rollout errors on the same cases and meshes.

```bash
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" experiments/head-ablation/smoke_arms.py \
  experiments/head-ablation/checks/smoke-arms.json
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" experiments/head-ablation/smoke_poisson.py \
  experiments/head-ablation/checks/smoke-poisson.json
```

## Running a cluster attempt

One job per directory, `gpu` partition only, and the batch script asserts `jax_backend=gpu`
before doing any work. Check `squeue` before and after every submission.

```bash
PY=/home/tahmid/Dev/.venv/bin/python
"$PY" experiments/head-ablation/cluster/stage.py <attempt>          # Burgers, nested layout
"$PY" experiments/head-ablation/cluster/stage_poisson.py <attempt>  # Poisson, flat layout
scp -r experiments/head-ablation/runs/<attempt> \
  tufts-login:/cluster/tufts/paralab/tawal01/headabl_20260914/
ssh tufts-login "cd /cluster/tufts/paralab/tawal01/headabl_20260914/<attempt> && \
  sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
```

Collect, audit, archive and then delete only that exact remote attempt directory:

```bash
"$PY" experiments/head-ablation/cluster/collect.py <attempt>
"$PY" experiments/head-ablation/audit_ablation.py \
  experiments/head-ablation/runs/<attempt>/archive/output/result.json \
  --out experiments/head-ablation/checks/<attempt>-audit.json \
  --fields experiments/head-ablation/runs/<attempt>/archive/output
"$PY" experiments/head-ablation/cluster/preserve_archive.py <attempt>
ssh tufts-login "rm -rf /cluster/tufts/paralab/tawal01/headabl_20260914/<attempt>"
```

## Plain-language glossary

- **Arm:** one configuration under test; everything except the named difference is held fixed.
- **Bank:** the fixed set of spatial fields the reduced state is built from.
- **Coefficient map / head:** the function turning the few solved coordinates into bank
  coefficients. This is the object under ablation.
- **Latent dimension:** how many numbers the online solver actually solves for.
- **Test modes:** the smooth functions the PDE residual is averaged against; there must be
  more of them than solved unknowns or the objective collapses.
- **Empirical quadrature:** a learned weighted subset of grid points standing in for a full
  grid sum; "dense" means no such approximation was used.
- **POD / POD-LSPG:** the classical linear snapshot basis / finding its coefficients by
  least-squares minimisation of the projected residual.
- **Bank projection error:** the best any coefficients at all could do in that bank — a floor.
- **Best-found reconstruction error:** the best that arm's own manifold can do on the reference
  field with no PDE involved, separating representation from dynamics.
- **Rollout error:** the error of the real online solve against the refined reference.
- **Development / final cohort:** cases usable for method selection / cases kept unopened.
