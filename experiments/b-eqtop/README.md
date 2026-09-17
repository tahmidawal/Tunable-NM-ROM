# b-eqtop — certifying the top rungs of the Burgers EQ ladder

`q-ridge` left the EQ correction ladder with one violation: at $q = 128$ and $q = 256$ no
empirical-quadrature rule with $m \le 2048$ reached the held-out bar $\rho_{\max} \le 0.116$,
and the top rung regressed. This lane grows $m$ to 6144 under three declared fit arms
(the incumbent construction with a larger pool; 64 fit states; 64 fit states with rows
scaled so the objective is $\sum_s \rho_s^2$), certifies every rule by held-out $\rho$,
rebuilds the ladder with the cheapest certified rule per rung, and measures the
$\rho$-vs-$m$ law at every rung. Read [`DESIGN.md`](DESIGN.md) first.

| path | what it is |
| --- | --- |
| `DESIGN.md` | pre-registered design, bars, gates, pass/fail, falsification, amendments |
| `eqtop.py` | design construction (two scalings), exact QR compression, the chain scheduler, NumPy field-path $\rho$, archived-rule reconstruction |
| `fitworker.py` | CPU-only worker running `varpro.bounded_nnls` verbatim (gated bitwise in the smoke) |
| `q_eqtop.py` | the job driver (phases 1–2 are `q-ridge/q_eqcert.py` verbatim) |
| `make_config.py` → `config-j1.json`, `config-j2.json` | the two jobs; constants read from the parent config and the qrg304 archive |
| `import_rules.py` → `rules/qrg304/` | qrg304's 18 rules, checksum-verified against its `OUTPUTS.sha256` |
| `make_comparators.py` → `checks/comparators.json` | qrg304's per-arm metrics for the fidelity gates |
| `smoke_eqtop.py` → `checks/smoke-eqtop.json` | local smoke (64 intervals) with the parent-baseline reproduction |
| `audit_eqtop.py` | independent NumPy audit, no JAX |
| `cluster/` | staging, collection, archive chunking |
| `reports/generate_eqtop.py` → `reports/2026-09-1x-b-eqtop.md`, `reports/summary.json` | the source-generated report |
| `artifacts/` | checksum-collected raw archives as bounded Git chunks |

## Reproducing

```bash
PY=/home/tahmid/Dev/.venv/bin/python
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" experiments/b-eqtop/smoke_eqtop.py experiments/b-eqtop/checks/smoke-eqtop.json
"$PY" experiments/b-eqtop/make_config.py
"$PY" experiments/b-eqtop/cluster/stage.py bet101 config-j1.json
rsync -a experiments/b-eqtop/runs/bet101/ tufts-login:/cluster/tufts/paralab/tawal01/b_eqtop_20260917/bet101/
ssh tufts-login 'cd /cluster/tufts/paralab/tawal01/b_eqtop_20260917/bet101 && sbatch run.sbatch'
"$PY" experiments/b-eqtop/cluster/collect.py bet101
"$PY" experiments/b-eqtop/audit_eqtop.py experiments/b-eqtop/runs/bet101/archive/output/result.json \
  --fields experiments/b-eqtop/runs/bet101/archive/output --bank experiments/b-eqtop/runs/bet101/archive/output/bank_G.npz \
  --out experiments/b-eqtop/checks/bet101-audit.json
"$PY" experiments/b-eqtop/cluster/preserve_archive.py bet101
"$PY" experiments/b-eqtop/reports/generate_eqtop.py --j1 experiments/b-eqtop/checks/bet101-audit.json \
  --j2 experiments/b-eqtop/checks/bet201-audit.json --out experiments/b-eqtop/reports/2026-09-17-b-eqtop
```
