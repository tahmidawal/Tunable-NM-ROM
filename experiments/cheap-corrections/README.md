# Making the correction ladder converge cheaply

The audited fixed-weight correction ladder (`experiments/head-ablation`, job `3713867`) showed
that solved error on Burgers falls monotonically with the number $q$ of extra fixed bank
directions, but that only $q\le16$ converges under the shared stopping rule and that cost grows
about $28\times$. This cell changes exactly the three things responsible — the solver, the test
count and the quadrature — on the **same** checkpoint, the **same** nested directions and the
**same** reachable set, and prices each one separately. The predeclared protocol, including the
amendments made after the local smoke, is [DESIGN.md](DESIGN.md); project state stays in the
canonical [lab log](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md).

## Layout

| file | what it is |
| --- | --- |
| `varpro.py` | four solver variants over one residual, the joint stationarity accounting, the bounded NNLS quadrature fitter and the enriched quadrature codes |
| `directions.py` | the audited nested direction fit, reproduced, plus a flattened single-`vmap` path so the compilation-bound setup cost can be measured |
| `cheap_ladder.py` | the Burgers 2D driver |
| `poisson_ladder.py` | the Poisson 2D driver; the Poisson residual is linear in the coefficients, so its corrections are eliminated exactly |
| `smoke_cheap.py` | the local fidelity smoke, under a minute, one process |
| `probe_cost.py` | a local cost probe at the real mesh with proxy directions; chooses the ladder's solver variant on measured cost |
| `audit_cheap.py`, `audit_poisson_ladder.py` | independent NumPy audits; no JAX, no GPU |
| `reports/generate_cheap_corrections.py` | the one report for both PDEs, generated from the audited JSON |
| `cluster/` | staging, checksum collection and the Git-trackable chunked archive |
| `checks/` | retained smoke, probe and audit evidence |
| `artifacts/<attempt>/` | the checksum-verified raw job archive, in bounded chunks |

`runs/` is Git-ignored scratch for staging and collection; `artifacts/` is the retained record.

## The three changes

1. **Variable projection.** For fixed $z$ the weak residual is quadratic in the correction
   coefficients $y$, so $y$ is solved by an inner damped Gauss–Newton and the outer
   stationarity-aware LM runs on $z\in\mathbb R^{16}$ alone, at the $q=0$ trust radius, with $y$
   never subject to that radius. Three eliminating variants (`varpro`, `block`, `alt`) are
   measured against the audited `joint` solver at matched $q$, test count and quadrature.
2. **Test count.** $M=4(K+q)$ (retained), $M=2(K+q)$, and a fixed $M=256$ wherever $M>K+q$.
3. **Empirical quadrature per rung.** One nonnegative-least-squares rule per $q$, fitted offline
   on enriched decoder-output advection snapshots, with a walltime-bounded block-greedy fitter.
   The two rules the audited ladder could construct are refitted with the retained exact fitter,
   so they reproduce it.

## Gates

Nothing is believed before these pass; they are listed with their tolerances in `DESIGN.md`.

```bash
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" experiments/cheap-corrections/smoke_cheap.py \
  experiments/cheap-corrections/checks/smoke-cheap.json
```

## Running a cluster attempt

One job per directory, `gpu` partition only, and the batch script asserts `jax_backend=gpu`
before doing any work. Check `squeue` before and after every submission.

```bash
PY=/home/tahmid/Dev/.venv/bin/python
"$PY" experiments/cheap-corrections/cluster/stage_burgers.py <attempt>   # nested layout
"$PY" experiments/cheap-corrections/cluster/stage_poisson.py <attempt>   # flat layout
scp -r experiments/cheap-corrections/runs/<attempt> \
  tufts-login:/cluster/tufts/paralab/tawal01/cheap_corr_20260915/
ssh tufts-login "cd /cluster/tufts/paralab/tawal01/cheap_corr_20260915/<attempt> && \
  sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
```

Collect, audit, archive, then delete only that exact remote attempt directory:

```bash
"$PY" experiments/cheap-corrections/cluster/collect.py <attempt>
"$PY" experiments/cheap-corrections/audit_cheap.py \
  experiments/cheap-corrections/runs/<attempt>/archive/output/result.json \
  --out experiments/cheap-corrections/checks/<attempt>-audit.json \
  --fields experiments/cheap-corrections/runs/<attempt>/archive/output
"$PY" experiments/cheap-corrections/cluster/preserve_archive.py <attempt>
ssh tufts-login "rm -rf /cluster/tufts/paralab/tawal01/cheap_corr_20260915/<attempt>"
```

## Plain-language glossary

- **$q$:** how many extra fixed spatial directions the online solver is allowed to use on top of
  the frozen neural head.
- **Variable projection:** eliminating the coefficients the residual depends on simply, so the
  hard nonlinear search stays small.
- **Trust radius:** a cap on how far one solver step may move. The audited ladder applied the
  $q=0$ cap to the whole augmented step; that is what bound at large $q$.
- **Test modes ($M$):** the smooth functions the PDE residual is averaged against. There must be
  more of them than unknowns.
- **Empirical quadrature ($m$):** a learned weighted subset of grid points standing in for a full
  grid sum. "Dense" means no such approximation.
- **Converged:** every solve stopped for a legitimate reason with no iteration-budget exit, and
  the joint normalized gradient is below $10^{-6}$.
- **Same-grid error:** discrepancy against the converged full-order solve on the same mesh, so it
  excludes that mesh's own discretization error.
- **Development / final cohort:** cases used for measurement / cases kept unopened.
