# q-ridge — is the Burgers evolved-times regression test-space overfitting?

`b-ladder-top` left the $q=16$ regression on the worst-over-evolved-times metric
unexplained. The hypothesis under test here is that the extra correction unknowns
**overfit the $M$ weak test equations**: the solve lowers the projected residual by moving
along directions the tests barely observe, which raises the field error.

Three probes, each isolated, on the frozen Burgers checkpoint
`sep_hfit_dense_mid_N256_dense.pkl` (SHA256 `18f0266ae6f0…`, $K=16$, $R=512$) at 256
intervals, on the same six opened development cases, under `b-ladder-top`'s budget-600
block-damped variable-projection contract and the **old** direction rule.

- **R1** a field-metric ridge $\lambda\|y\|_W^2$ on the correction block, with
  $\lambda = \lambda_{\mathrm{rel}}\sigma_q^2$ dimensionless.
- **R2** more tests at fixed $q$: $M \in \{4,8,16\}(K+q)$.
- **R3** the control: the exactly-integrated weak residual on **held-out** test modes,
  computed post hoc in NumPy from the retained per-step bank coefficients.

The predeclared design, the equations, the gates, the pass criterion and the falsification
clause are in [`DESIGN.md`](DESIGN.md). Read it before the report.

## Layout

| path | what it is |
| --- | --- |
| `DESIGN.md` | predeclared design, gates, pass/falsification, amendments |
| `ridge.py` | the ridge as an augmented residual; $\lambda = 0$ branches to the retained solver itself |
| `q_ridge.py` | the driver for both jobs |
| `r3.py` | NumPy modes / advection / weak residual — the audit imports no JAX |
| `make_configs.py`, `make_comparators.py` | generate the configs and the cross-job comparator table from the retained archives |
| `smoke_ridge.py` | local smoke, 64 intervals, eight gates |
| `audit_ridge.py` | independent NumPy audit; no JAX, no GPU |
| `config-r1.json`, `config-r2.json` | the two sweeps |
| `cluster/` | staging, collection and archive-chunking helpers |
| `checks/` | smoke, comparator and audit JSONs |
| `artifacts/` | checksum-collected raw archives as bounded Git chunks |
| `reports/` | the source-generated report, its LaTeX twin, its figure and its generator |

## Reproducing

```bash
PY=/home/tahmid/Dev/.venv/bin/python

# local smoke (GB10)
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" \
  experiments/q-ridge/smoke_ridge.py experiments/q-ridge/checks/smoke-ridge.json

# regenerate the configs and the comparator table (both read the retained archives)
"$PY" experiments/q-ridge/make_configs.py
"$PY" experiments/q-ridge/make_comparators.py

# stage, submit, collect, audit (Tufts; namespace q_ridge_20260916)
"$PY" experiments/q-ridge/cluster/stage.py qrg101 config-r1.json
rsync -a experiments/q-ridge/runs/qrg101/ \
  tufts-login:/cluster/tufts/paralab/tawal01/q_ridge_20260916/qrg101/
ssh tufts-login 'cd /cluster/tufts/paralab/tawal01/q_ridge_20260916/qrg101 && sbatch run.sbatch'
"$PY" experiments/q-ridge/cluster/collect.py qrg101
"$PY" experiments/q-ridge/audit_ridge.py \
  experiments/q-ridge/runs/qrg101/archive/output/result.json --mode r1 \
  --fields experiments/q-ridge/runs/qrg101/archive/output \
  --bank experiments/q-ridge/runs/qrg101/archive/output/bank_G.npz \
  --out experiments/q-ridge/checks/qrg101-audit.json
"$PY" experiments/q-ridge/cluster/preserve_archive.py qrg101

# the report, both formats and the figure
"$PY" experiments/q-ridge/reports/generate_q_ridge.py \
  --r1 experiments/q-ridge/checks/qrg101-audit.json \
  --r2 experiments/q-ridge/checks/qrg201-audit.json \
  --out experiments/q-ridge/reports/2026-09-16-q-ridge
```

`qrg201` is the same shape with `config-r2.json` and `--mode r2`.

## Restoring an archive

Each `artifacts/<attempt>/` holds the verified `collection.tar.gz` as 48 MB chunks plus
`archive.json`, `SHA256SUMS` and the uncompressed `result.json`; see the `README.md`
written beside the chunks. `output/bank_G.npz` is deliberately **not** in those chunks —
see `EXCLUDED-FROM-ARCHIVE.txt` and `BANK-SHA256.txt`.
