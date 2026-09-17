# p-linear

The Poisson 2D correction ladder run to $q = R$ on the best checkpoint, with POD-LSPG to
$k' = 512$, the direct DST solve and CG all timed in the same job: the paper's linear-case
figure. Pre-registration: [`DESIGN.md`](DESIGN.md).

## Layout

| file | what it is |
|---|---|
| `DESIGN.md` | pre-registered question, criterion, arms, gates, falsification, job-3 plan |
| `plin_core.py` | basis extension to $R$ directions, test-count rules, augmented best-found oracle, non-dominated set |
| `plin_solve.py` | jobs 1–2 driver: the ladder and every comparator at one mesh |
| `plin_head.py` | job 3 driver: head-capacity arms on the frozen bank, evaluated and solved in-job |
| `directions.py` | verbatim copy of `cheap-corrections/directions.py` (commit `d90e09aa`), for the ccpoi01 gate only |
| `config-1024.json`, `config-256.json`, `config-head.json` | the configurations |
| `checkpoints/` | `pbh02`'s frozen primaries and bases, SHA-verified against its `models.json` |
| `references/` | per-case cross-job fidelity references extracted from `pbh02` and `ccpoi01` by `make_references.py` |
| `audit_np.py` | independent NumPy/SciPy audit; imports neither the driver nor JAX |
| `cluster/` | flat Git-verified staging, checksum collection, chunked archives |
| `checks/` | local smoke records |
| `reports/` | the generated report, `summary.json`, figures, and their generator |

## Running

```bash
# local smoke (64 intervals, one repetition; must pass the pbh02 gates at 64)
source /etc/profile.d/jax-mem.sh
E=experiments; export PYTHONPATH=$PWD/$E/p-linear:$PWD/$E/p-bank-head:$PWD/$E/multiresolution-poisson:$PWD/$E/head-ablation:$PWD/$E/mr-burgers2d
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun /home/tahmid/Dev/.venv/bin/python \
  experiments/p-linear/plin_solve.py --config experiments/p-linear/config-1024.json \
  --out experiments/p-linear/checks/smoke64 --smoke

# cluster
python experiments/p-linear/cluster/stage.py solve plin1024 --config config-1024.json
python experiments/p-linear/cluster/stage.py solve plin256  --config config-256.json
python experiments/p-linear/cluster/stage.py head  plhead1  --config config-head.json
```

Namespace `/cluster/tufts/paralab/tawal01/p_linear_20260917/`, one attempt directory per
job, `squeue` checked before and after every submit.
