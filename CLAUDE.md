# Tunable-NM-ROM — compute, GPU, and experiment operations

Operational rulebook for running this repository on the local GB10 box and the Tufts HPC
cluster. This file is about **how to run jobs safely and reproducibly**. For the model,
equations, configurations, and reported results, read `README.md`, `heat/README.md`,
`poisson/README.md`, and the per-cell documents under `best-results/`.

The repository has two clean packages:

- `heat/`: Heat 2D/3D data generation, ViT-CP training, EQ construction, and ROM rollout.
- `poisson/`: Poisson 2D/3D data generation, ViT-LinearCP training, EQ construction, and ROM solve.

`best-results/` is the frozen experiment archive. Do not casually rewrite those scripts or
reported outputs; make a new experiment directory when extending them.

## Branching — main is the baseline; experiments live in worktrees

`main` is the frozen baseline. Never run or commit experiments directly on it.

Every new experiment starts in its own git worktree on its own branch, pushed to GitHub
(`origin` = https://github.com/tahmidawal/Tunable-NM-ROM.git):

- **Ask the user before creating a worktree** — propose the name and get confirmation first.
- Worktree dirs live in `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/`,
  named `YYYY-MM-DD-<short-descriptive-slug>` — date first so directories sort
  chronologically. The `worktrees/` dir is listed in `.git/info/exclude` so `main` stays
  clean; it is never committed.
- The branch name mirrors the directory: `exp/YYYY-MM-DD-<slug>`.
- Push the branch to origin with `-u` immediately and keep it synced as work progresses.

```bash
cd /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude
git worktree add -b exp/YYYY-MM-DD-<slug> worktrees/YYYY-MM-DD-<slug> main
git -C worktrees/YYYY-MM-DD-<slug> push -u origin exp/YYYY-MM-DD-<slug>
```

Each worktree doubles as its own isolated job/submit directory, satisfying the
one-job-per-directory rule below. Finished experiments stay on their branches as an archive;
merge back to main only deliberately.

## Sessions, worktrees, and the lab log

### One session writes to one worktree

A session works in a single worktree unless told otherwise. Reading across worktrees is fine
and often necessary — comparing cells, assembling reports, auditing another cell's JSONs — but
**write to one tree only**. Two sessions or agents writing into the same tree corrupt each
other's `runs/` and cluster directories.

### Several experiments at once — one worktree each, and ask first

When asked to run more than one experiment concurrently, give each its own subagent and its own
worktree. **Propose the names and get confirmation before creating any of them.** Give each a
distinct cluster namespace as well (`/cluster/tufts/paralab/tawal01/<ns>/`), because the account
is shared and jobs from different experiments must never land in one directory.

**When they finish, ask whether to merge the worktrees into one.** Do not merge on your own
initiative, and do not leave the decision unasked — a finished experiment sitting alone on its
branch is easy to lose track of. Record whichever way it goes in the lab log.

### Starting a new session on new ideas — ask where to start from

Before creating a worktree for new work, ask whether it should branch from the most recent
worktree or from somewhere else. Say which one you would pick and why. Never assume `main`:
`main` is the frozen baseline and its heat ROM rollout is known broken, so branching from it
silently discards every correction made since.

### One canonical `LAB-LOG.md`, on `main`, read and appended by every session

`/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md`. Worktrees share one `.git`, so
that absolute path resolves to main's copy from inside any worktree — there is **no per-worktree
copy**, and no separate status or start-here document. **Read it at the start of a session and
append to it before the session ends; appending is a closing step, not an optional extra.**

The file has two parts: a **Where things stand** block at the top that each session rewrites when
what is true has changed, and an append-only chronology below it, oldest first, `## YYYY-MM-DD`
headers with one `###` per session.

Each entry records:

- what was run, and where it landed (branch, job ids, cluster namespace)
- what was found, with the numbers
- **what was wrong and got retracted** — this matters more than the successes, and it is the
  part that gets silently dropped
- what is left open, and what the next session should pick up

Two failures this rule exists to prevent, both from 17–18 August: a root-cause investigation
that existed only in a `/tmp` scratchpad and was nearly lost, and a set of trained decoders that
were `.gitignore`d and tracked nowhere, so the branch could be read but not rerun. **If work only
exists in a scratchpad or in conversation, it does not exist.**

## Reports

Written-up results live in `reports/` on `main`, beside the lab log. One report per question,
never one per session.

- **Name them `YYYY-MM-DD-<what-it-contains>.md`** — the date is when the *numbers* were
  finalised, not when the prose was last edited, and the slug says what the report is about so
  the filename alone tells a reader whether to open it. Good:
  `2026-08-18-solve-cost-quadrature-and-the-hybrid.md`. Bad: `results2.md`, `final-report.md`.
- **Open every report with a level-1 title and one or two sentences** saying what it covers and
  what state its numbers are in — final, provisional, or superseded.
- **Never hand-type a number into a report.** Generate tables from the run JSONs with a script in
  `reports/`, so a report cannot drift from its data. Every prose number that has gone wrong in
  this project went wrong by being typed by hand.
- **Say which numbers are provisional and why**, inline, next to them — not only in a preamble.
- `*.built.md` files are generated for publishing; edit the source, not the build.

## Routing — which machine

### Tufts is the default; the local GB10 is the fallback

For any real GPU job, use the cluster unless it is a tiny, sub-minute smoke test.

1. Check free Tufts GPUs first. Prefer A100 or H100; H200 and L40S are also acceptable.
2. Use the local GB10 only when no cluster GPU is free.
3. Keep quick smoke tests and debugging runs local.

Do not narrow a sweep because the shared GB10 is slow. Run the intended meshes, seeds,
families, latent dimensions, and comparison arms on Tufts instead.

### Parallelize independent work

- On Tufts, submit independent meshes, PDE families, seeds, latent dimensions, and A/B arms
  simultaneously.
- Every concurrent cluster job must have its **own job/submit directory**. Jobs sharing a
  directory can corrupt each other's `data/`, checkpoint, or `.npz` writes.
- Locally, run at most **three concurrent `jaxrun` jobs**.
- Within one experiment, `generate data -> train -> run ROM` is sequential. Separate
  experiment cells are independent and should overlap.
- Continue useful CPU-side analysis, plotting, documentation, and code preparation while GPU
  jobs run.

## Local GB10 box

The CPU and GPU share roughly 128 GB. Every JAX process must run through `jaxrun`, which gives
the process its own memory-limited cgroup.

### Required Python environment

Always use the environment by absolute path. Never use bare `python` or `python3`.

| Location | Python environment | Key contents |
|---|---|---|
| Local GB10 | `/home/tahmid/Dev/.venv` | Python 3.12, JAX/JAXlib 0.10.1 CUDA, Flax 0.12.8, Optax 0.2.8 |
| Tufts | `/cluster/tufts/paralab/tawal01/ae-research/venv` | Python 3.13.2, JAX 0.10.2 CUDA 12 |

The system Python and the Tufts login shell do not provide the required JAX environment.

If a package must be added to the local environment, install it without dependency resolution:

```bash
/home/tahmid/Dev/.venv/bin/pip install --no-deps <package>
/home/tahmid/Dev/.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
```

`--no-deps` is mandatory. A normal `pip install` can replace `jaxlib` with an incompatible or
CPU-only build. After installation, JAX must still report version 0.10.1 and a CUDA device. Do
not create a second Flax/JAX environment.

### The local launch command

```bash
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest \
  jaxrun /home/tahmid/Dev/.venv/bin/python <module-or-script> ...
```

- `jaxrun` creates a `systemd --user --scope` with a hard 36 GB ceiling and a soft throttle
  near 30 GB. Use `JAXRUN_MAX=64G jaxrun ...` only when one job genuinely needs more memory.
- `/etc/profile.d/jax-mem.sh` sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.25`.
- `JAX_DEFAULT_MATMUL_PRECISION=highest` is required for accuracy-sensitive training, EQ/POD
  work, and ROM evaluation.
- Never run more than three concurrent local `jaxrun` processes. Watch them with
  `systemd-cgtop --user`.
- `earlyoom` starts terminating processes around 15% free memory; it is a last-resort safety
  net, not a memory-management strategy.

## Tufts HPC cluster

### Access and submission

- Use `tufts-login` for availability checks, `squeue`, and submission.
- Do not submit through `tufts` or `tufts-code`; those aliases point to a temporary parking
  compute node.
- Use the `gpu` partition only. Never use `preempt`.
- The helper scripts in the separate `ae-research` repository are the preferred interface:

```text
tufts-free.sh              # free GPUs by type
tufts-submit.sh "<cmd>"    # sync a working tree, submit, and print the job id
tufts-status.sh <jobid>    # queue state and log tail
tufts-pull.sh              # pull results and logs
```

`tufts-submit.sh` accepts `--gpu a100|h100|h200|l40s|auto`, `--time`, `--mem`, and `--cpus`.
Automatic GPU preference is A100, H100, H200, then L40S. If the helpers are unavailable, use
`sbatch` directly while preserving every requirement below; do not invent another SSH or
submission path.

### Mandatory job requirements

- Activate `/cluster/tufts/paralab/tawal01/ae-research/venv` inside the batch job.
- Set `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- Assert that JAX is actually using a GPU before doing any work:

```bash
/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -c \
  "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b == 'gpu' else 42)"
```

The job log must contain `jax_backend=gpu`. A node with a failed `cuInit` can silently fall
back to CPU and produce plausible but invalid timing results. If the preflight exits 42,
resubmit on another GPU type. Allow CPU only for explicitly CPU-only jobs.

- Write logs, data, checkpoints, and job output under the paralab space, never the cluster
  home directory, which is at quota.
- Regenerate data from the seed at the start of each cluster experiment. `data/` is not
  synchronized from the local machine.
- Pull results and logs after completion, then delete finished cluster job directories. Do
  not park checkpoints on the nearly full group share.

### Duplicate-submit race

Check `squeue` before and after every submission and confirm exactly one job is associated
with each job directory. Never submit two jobs from the same directory. Give every mesh,
family, seed, latent dimension, or comparison arm a unique submit/job directory.

### GPU sizing

- A100/H100 are appropriate for ordinary configurations through roughly nc64-scale derived
  experiments.
- nc128-scale derived runs with about 262k DOF and 20k samples can exceed an 80 GB A100. Use
  an H200 with approximately `--mem 240G`.
- Do not close a JIT-compiled function over a large training array. Pass the array as an
  explicit JIT argument; otherwise XLA may embed it as a captured compile-time constant and
  segfault during lowering.
- Disable full-batch L-BFGS polishing (`LBFGS_ITERS=0`) for derived experiments at or above
  nc128/n >= 1M. The closure can consume roughly 245 GB of host memory.

## Running this repository

### Local smoke tests

Run these independently from the repository root. They are deliberately tiny and belong on
the local GB10.

```bash
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python

cd heat
PYTHONPATH=src JAX_DEFAULT_MATMUL_PRECISION=highest \
  jaxrun "$PY" -m pytest tests/ -q

cd ../poisson
PYTHONPATH=src JAX_DEFAULT_MATMUL_PRECISION=highest \
  jaxrun "$PY" -m pytest tests/ -q
```

The smoke tests validate interfaces and the end-to-end pipeline on N=8. They do not validate
converged accuracy.

### Full clean-package experiment

A clean-package run has three true dependencies. Execute them in order within one isolated
cluster job directory. The Heat example is:

```bash
cd heat
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python

JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" -m scripts.generate_data \
  --config configs/heat3d_n64.yaml --out data/heat3d_n64.npz

JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" -m scripts.train \
  --config configs/heat3d_n64.yaml \
  --data data/heat3d_n64.npz --out checkpoints/heat3d_n64.pkl

JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" -m scripts.run_rom \
  --config configs/heat3d_n64.yaml \
  --data data/heat3d_n64.npz --ckpt checkpoints/heat3d_n64.pkl
```

For Poisson, work from `poisson/` and use the corresponding configuration and module paths:

```bash
cd poisson
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python

JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" -m scripts.generate_data \
  --config configs/poisson3d_n64.yaml --out data/poisson3d_n64.npz

JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" -m scripts.train \
  --config configs/poisson3d_n64.yaml \
  --data data/poisson3d_n64.npz --out checkpoints/poisson3d_n64.pkl

JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" -m scripts.run_rom \
  --config configs/poisson3d_n64.yaml \
  --data data/poisson3d_n64.npz --ckpt checkpoints/poisson3d_n64.pkl
```

Do not run these full configurations locally merely because the data is already present.

### Frozen `best-results` experiments

- Read the cell's `README.md` before running it; each cell records its exact scripts and
  configuration.
- Preserve the frozen archive. Copy a cell into a new, clearly named experiment directory
  before changing architecture, data, or hyperparameters.
- Keep each sweep cell isolated so concurrent runs cannot share data or checkpoint paths.
- Record the seed, mesh, family, latent dimension, config, Git commit, GPU type, JAX backend,
  precision setting, job id, and output paths with every result.
- Treat an empty cluster log as a possible disk-full failure before diagnosing a code bug.

## Accuracy-critical inherited experiment rules

`JAX_DEFAULT_MATMUL_PRECISION=highest` applies to this checkout directly. Some later
POD+MLP/decoder experiments derived from this project also use the following environment
flags; the May 13 public scripts in this clean checkout do not currently consume them:

- `GRAM64=1` for `scope2d`: build the POD Gram matrix in f64; f32 can create a false error
  floor around 2e-4. Expected top-k orthonormality deviation is around 1e-8, not 1e-4.
- `DT64=1` for the generator, scope, and trainer: store f64 data when measuring below the
  roughly 1e-7 f32 quantization plateau.
- `FREEZE_WDEC=1` for `dec2d`: keep the linear lift fixed at V-transpose; training the large
  matrix can destroy the POD initialization below roughly 1e-4.

If those derived scripts are brought into this checkout, retain these settings. Do not assume
an environment flag has an effect without confirming that the target script reads it.

## Result-integrity checklist

Before accepting a GPU result, confirm all of the following:

- The intended repository commit and configuration were used.
- The log says `jax_backend=gpu`.
- `JAX_DEFAULT_MATMUL_PRECISION=highest` was set.
- The job used an isolated directory and did not race another submission.
- Data was regenerated from the recorded seed.
- There were no captured-large-constant warnings, OOMs, truncated logs, or disk-full errors.
- Results and logs were pulled locally and the cluster job directory was cleaned up.
