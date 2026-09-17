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
| `checks/` | local smoke records; `oracle_fix_check.py` + `2026-09-17-oracle-fix-check-64.json` (DESIGN A10 fix verified), `2026-09-17-oracle-metric-scale.json` (the measured $V^\top V$ scale per mesh), the generated lab-log entry |
| `reports/` | the generated report, `summary.json`, `verdicts.json`, figures, their generator, and `generate_entries.py` (writes `self-audit-report.md` and the lab-log entry from the same JSONs) |

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
# the 1024 resubmit after the A100-40GB OOM (DESIGN A7)
python experiments/p-linear/cluster/stage.py solve plin1024b --config config-1024.json --gpu h200 --mem 240G
```

Namespace `/cluster/tufts/paralab/tawal01/p_linear_20260917/`, one attempt directory per
job, `squeue` checked before and after every submit.

## State (2026-09-17)

Jobs: `plin256` = 3780692 (A100-PCIE-40GB), `plin1024` = 3780691 (**retracted**, OOM in the
untimed dense oracle), `plin1024b` = 3783813 (**H200**), `plhead1` = 3783883 (A100-PCIE-40GB).
All three completed jobs are collected, audited, chunked under `artifacts/`, and their remote
directories deleted. The untimed augmented-oracle column for $q > 0$ in both ladder jobs is
retracted (DESIGN §A10; fixed in `plin_core.py`, not re-run). Nothing else is retracted.
