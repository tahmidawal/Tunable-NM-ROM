# train3d — E2 3D bank recipe for one seed (W1024; incumbent = jcp-wide-bank job j3b)

Code only, no results. This is DESIGN.md §3.1, §3.3 and A1.7. `train2w.py` and `configs/train_r1024.json` are
byte-identical to the j3b job's staged copies (`jcp-wide-bank/runs/j3b/MANIFEST.sha256`). `train2w.py` imports
`Path(__file__).parent.parent/vendor/quad3d/vendor/burgers3d-span/common.py`. From `experiments/jcp-robust/train3d/`
that is the lane's existing vendor copy (same sha256 as j3b), so no edit was needed. `PROVENANCE.json` lists every file.
Checks: `experiments/jcp-robust/train_tests.py` (static checks, plus `--dry3d`).

## Seeds and configs

`make_config.py` writes `configs/train_r1024_s<s>.json` with $\texttt{model\_seed} = 923715 + 1000s$ and
$\texttt{subset\_seed} = 923002 + 1000s$. It asserts that every other key equals the incumbent config; seed 0 is the
incumbent itself. The committed configs are s1 (924715 / 924002), s2 (925715 / 925002) and s3 (926715 / 926002).

**Checkpoint selection (A1.7) is train2w's own and is unchanged.** `train_bank` evaluates the bank-validation cohort
(seed 923751, 96 trajectories) every `checkpoint_every` = 1000 steps, at step 50 and at the last step. It keeps the
network with the lowest worst projection error over the three groups (33, 65, 129). The incumbent selected step
11,000. The selected step is `training.json` → `bank.selected_step` and is copied to `run_record.json`.

## Memory pilot (smoke, DESIGN §6)

`run_seed.sh pilot <outdir>` runs `configs/train_r1024_pilot.json`, which uses the seed-1 seeds with `bank_steps` =
200 and `warmup` = 100. The warmup change is forced: `optax.warmup_cosine_decay_schedule` raises for `bank_steps`
$\le$ `warmup` (500 in the recipe, checked locally). Neither key changes an array shape. The pilot generates all data,
runs 200 bank steps, then the ordering, the 129³ full-grid floors and the comparison-bank floors, so it reaches every
production memory peak (longer bank training allocates no further arrays).

Recorded every run (pilot and production):
- device peak: `memory.json` (`jax.devices()[0].memory_stats()` after the run, via `memwrap.py`, which executes the
  unchanged `train2w.py` with `runpy` as `__main__`), and `run_record.json` → `device_peak_GiB`;
- process `ru_maxrss` (`memory.json`), `/usr/bin/time -v` (`time_v.txt`), and 30 s RSS samples (`rss.log`);
- the authoritative host number, `sacct -j <jobid> --format=JobID,MaxRSS,ReqMem,Elapsed`, which the job script runs
  after the step finishes (or the lane agent runs afterwards).

Production gate (DESIGN §3.3): host RSS × 1.25 ≤ requested memory, and device peak ≤ 120 GB on an H200 (143,771 MiB).

## Job environment (copied from j3b `run.sbatch`)

H200, `--mem=420G`, `--cpus-per-task=8`; the incumbent ran 6,660 s in total, 4,104 s of it in bank training.
Environment: `JAX_ENABLE_X64=true`, `JAX_DEFAULT_MATMUL_PRECISION=highest`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`,
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=8`, and `TMPDIR`/`XDG_CACHE_HOME` under the output directory. `run_seed.sh` sets
all of these. It `cd`s to the job root, because `compare_bank` is the repo-relative path
`experiments/jcp-wide-bank/vendor/quad3d/inputs/model_M2`. That key is kept verbatim, so the script links that path
inside the job root to the lane's vendored `vendor/quad3d/inputs/model_M2` and checks the bank's sha256 `6687f259…`.
The comparison bank only produces the diagnostic `compare_bank_floors_full`.

## Outputs (`<outdir>`)

- `bank.pkl` (params, rotation $T$, $R_G$, $V$, cfg)
- `training.json`, `training.log`, `head_data.npz` (500 MB; no heads are trained, since `head_variants` = [])
- `memory.json`, `rss.log`, `time_v.txt`, `train.out`/`.err`
- `device.txt`, `run_record.json`, `OUTPUTS.sha256`, `SEED-COMPLETE` (written last)

## Glossary

- **W1024 bank**: the 3D coordinate-network bank with 1,024 columns, trained on 33³, 65³ and 129³ snapshots.
- **model_seed / subset_seed**: the network-initialisation and minibatch seed, and the seed of the fixed random 129³
  point subset.
- **selected step**: the training step whose bank had the lowest worst bank-validation projection error.
- **MaxRSS**: the peak resident host memory of the job, as Slurm reports it.
