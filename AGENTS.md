# AGENTS.md — operating rules for this repository

**Read `LAB-LOG.md` first.** It is the single canonical record for the project — current state
at the top, then the dated chronology including everything that was retracted. It lives on `main`
at the repository root and every session reads and appends to that one file by absolute path:

```
/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md
```

There is no per-worktree copy and no separate status document. This file carries no state — it is
only about *how to run things here without breaking them*.

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

## Sessions, worktrees, and the lab log

**One session writes to one worktree.** Reading across worktrees is fine and often necessary;
writing to more than one is not. Two agents writing into one tree corrupt each other's `runs/`
and cluster directories.

**Several experiments at once: one worktree each, and ask first.** Give each its own subagent,
its own worktree, and its own cluster namespace (`/cluster/tufts/paralab/tawal01/<ns>/`) —
the account is shared. Propose the names and get confirmation before creating any of them.
When they finish, **ask whether to merge the worktrees**; do not merge unprompted, and do not
leave the question unasked.

**Starting new work: ask where to branch from.** Say which base you would pick and why. Never
assume `main` — it is the frozen baseline with a known-broken heat rollout, so branching from it
silently discards every correction since.

**Append to the canonical `LAB-LOG.md` before your session ends** — the one on `main` at the
repository root, by absolute path, whichever worktree you are in. Add a dated `##` section with a
`###` per session at the bottom, and rewrite the "Where things stand" block at the top if what is
true has changed. Record what was run and where it landed, what was found with numbers, **what
was retracted**, and what is left open. Appending is a closing step, not an optional extra.

If work exists only in a scratchpad or in conversation, it does not exist. Two things were nearly
lost this way on 17–18 August: a root-cause investigation living in `/tmp`, and the trained
decoders, which were `.gitignore`d and tracked on no branch.

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
- **Equations in LaTeX, not ASCII math.** Write display and inline math with `$...$` /
  `$$...$$` in reports, design docs, and `understand/` explainers — GitHub and most
  previewers render it. Unicode-approximated formulas (`Φᵀu`, `R^{2×64}`) are acceptable
  only inside tables or code spans where LaTeX cannot render; never build multi-line
  equations out of ASCII art. Code identifiers stay as backticked code, not math.
- **Diagrams in mermaid, not ASCII box art.** Architecture and flow diagrams go in
  ```mermaid fences (GitHub, VS Code preview, and artifact pages all render them); use
  classDef color-coding to encode roles (e.g. trained / frozen / solved). ASCII drawings
  have already been rejected as unreadable once (2026-08-27).
- `*.built.md` files are generated for publishing; edit the source, not the build.

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
