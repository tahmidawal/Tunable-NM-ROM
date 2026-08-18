# AGENTS.md — operating rules for this repository

**Read `CODEX-START-HERE.md` first.** It has the state of every experiment, what is safe to
quote, the corrections already applied, and the prioritised next steps. This file is only about
*how to run things here without breaking them*.

Every rule below was written after something went wrong. None is stylistic.

---

## Python environments — absolute paths only

Never `python`, never `python3`. There is exactly one environment per machine.

| where | path |
|---|---|
| local box (GB10) | `/home/tahmid/Dev/.venv/bin/python` — py3.12, JAX 0.10.1 CUDA, Flax, Optax |
| Tufts cluster | `/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python` — py3.13, JAX 0.10.2 CUDA 12 |

The system Python has no JAX. The cluster **login shell** has no JAX — only that venv does.

Adding a package locally requires `--no-deps`, or pip will replace `jaxlib` with a CPU-only
build and every result silently becomes wrong:

```bash
/home/tahmid/Dev/.venv/bin/pip install --no-deps <package>
/home/tahmid/Dev/.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
# must print 0.10.1 and a CudaDevice
```

---

## Running jobs

**Everything is f64 and `JAX_DEFAULT_MATMUL_PRECISION=highest`.** Not optional — f32 creates
false error floors around 2e-4 that have been mistaken for real results.

### Local — smoke tests only, under a minute

```bash
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun /home/tahmid/Dev/.venv/bin/python <script>
```

`jaxrun` puts the job in its own cgroup with a 36 GB ceiling, so a runaway kills only itself.
**At most three concurrent `jaxrun` processes** — the box has ~128 GB shared between CPU and GPU.

### Cluster — everything real

Submit from `tufts-login`. Use the `gpu` partition only, never `preempt` (jobs there are killed
mid-run). Model any new job on `experiments/burgers2d-rom-latent-stepping/cluster/`.

Mandatory preflight in every batch script — a node with a failed `cuInit` falls back to CPU
silently and returns plausible, invalid numbers:

```bash
$PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
```

The log must contain `jax_backend=gpu`. If it exits 42, resubmit on a different GPU type.

- **All output to the paralab space.** The cluster home directory is at quota; a job writing
  there dies in about two seconds with no output file.
- **One job per directory.** Two jobs sharing a directory corrupt each other's writes.
- **Check `squeue` before and after every submit.**
- **Regenerate data from seed on the cluster.** `data/` is never synced.
- **`scp` code directly into the paralab path.** Never stage through login `/tmp` — login nodes
  are load-balanced with node-local `/tmp`, and a stale script silently corrupted a round.
- Pull results with checksums, then delete the cluster job directory. The share runs near full,
  and a job can exit 1 with an empty log when it fills — check disk before diagnosing a code bug.

### Cancelling jobs — the one that cost two agents their queue

**Never run bare `scancel`.** `scancel --name=a,b,c` takes one name, not a list; the value
matches nothing, which leaves no effective selector, and a `scancel` with no filter cancels
**every job on the account**. This killed two agents' entire fleets at once.

Use `experiments/cost-to-tolerance/cluster/cancel.sh` with explicit numeric job ids. It refuses
names, globs and user selectors, verifies every id is queued under this account and named
`ctol_*`, and aborts the whole call if any id fails.

---

## Measurement rules

These are what separate a result from a plausible-looking number.

- **Burn in the GPU before every timed block.** A 17% clock-ramp bias after a long host-bound
  NNLS fit manufactured a crossover that does not exist.
- **Never compare wall clock across jobs.** Different panels land on different A100 models; the
  difference alone produced a 3.7× apparent speed-up. Compare within one job on one GPU, or use
  iteration counts, which are hardware-free.
- **Cost and accuracy must come from the same solver invocation.** Assembling them from separate
  runs made an entire earlier table unusable.
- **Persist timing repetition arrays**, or post-selection bias is unbounded and any "best
  configuration" claim has to be withdrawn rather than defended.
- **Name the reference tolerance beside any correction factor.** A 1.56× correction was published
  as 1.16× purely by quoting the tightest of three unlabelled columns.
- **Compare against a like-for-like baseline.** Both classical baselines here were over-solved —
  Burgers ran a fixed 8 Newton iterations per step, Poisson ran CG to the tolerance used to
  *manufacture the truth data*. Speedups against those are inflated.
- **`git -C <staged dir>` on the cluster walks into an unrelated ancestor repo.** Verify
  provenance by content hash against git history, not the working tree.
- **Report medians and outlier counts, not just means.** A mean over 16 cases dominated by 3
  divergent solves produced a false finding that survived several rounds of review.

## Method rules

- Minimise the **weak form** (residual projected onto smooth test modes), never the pointwise
  residual — the discrete Laplacian amplifies grid-scale decoder error by ~N².
- **M > k comfortably.** At `M = k` the objective collapses.
- **m ≈ 4M quadrature points.** At `m = M` the quadrature fit degrades ~45× with worst-row
  error of 8.5e+05, which turns an entire column into an artefact.
- Fit NNLS quadrature weights on **decoder-output** snapshots, never residual snapshots. Refit
  whenever N or M changes.
- **Hyper-reduce the cold start too**, or the online path stays grid-bound.
- Never use random, importance, or off-grid strong-form collocation on localised families —
  every such arm failed on every PDE.
- On Burgers, keep the FOM-exact upwind operator inside the weak advection term.

---

## Git

`main` is the frozen public baseline and its heat ROM rollout is broken (frozen after step 1) —
do not use it as a behavioural reference. Work on a branch; experiments live in dated worktrees
under `worktrees/`, named `YYYY-MM-DD-<slug>` with branch `exp/<same>`. Ask before creating one.

Do not rewrite anything under `best-results/` — it is a frozen archive. Copy a cell into a new
directory before changing architecture, data, or hyperparameters.

---

## Before accepting any GPU result

- the log says `jax_backend=gpu`
- `JAX_DEFAULT_MATMUL_PRECISION=highest` was set and the run is f64
- the job had its own directory and did not race another submission
- data was regenerated from the recorded seed
- no captured-large-constant warnings, OOMs, truncated logs, or disk-full errors
- results pulled with checksums and the cluster directory deleted
- seed, mesh, family, latent dimension, config, commit, GPU type, job id recorded with the result
