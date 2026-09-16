# q-trajdirs — the dense correction ladder, and trajectory-fitted directions as its control

> **Redirected on 2026-09-16, mid-flight.** The lane opened on the trajectory-directions
> question below. While its job was running, the `q-diag` lane established that the
> $q=16$ evolved-metric regression is the **empirical quadrature**, not the directions.
> The **primary question is now**: *does the DENSE correction ladder, incumbent
> directions, per-step budget 600, satisfy the knob criterion on both metrics inside one
> allocation?* The trajectory-directions job was let finish and is reported as a control
> of that verdict. See section 11 of [`DESIGN.md`](DESIGN.md); nothing above that section
> was edited.

## The original question, kept as the control

One question on the frozen Burgers checkpoint `sep_hfit_dense_mid_N256_dense.pkl`
(SHA256 `18f0266ae6f0…`, $K=16$, $R=512$) at 256 intervals, on the same six opened
development cases the head ablation, the audited ladder, the cheap-corrections cell and
`b-ladder-top` used.

> Does fitting the correction directions $C_q$ to the ROM's **trajectory** error,
> instead of the head's **static** reconstruction residual, make the ladder monotone on
> the worst-over-evolved-times metric ($t>0$) while keeping it monotone on
> worst-over-all-times?

Everything except $C_q$ is the `b-ladder-top` Q1-B contract verbatim: the checkpoint, the
bank, the head, the weak objective, the initializer, the block-damped variable-projection
solver at per-step budget 600, the output contract, the mesh, the time step and the
cohort. The predeclared protocol, the equations, the cohorts, the gates, the pre-registered
pass and the three falsification statements are in [`DESIGN.md`](DESIGN.md). Read it
before the report.

## The three direction sets

| set | residual decomposed | cohort |
| --- | --- | --- |
| `old` | $\eta_i-h_\theta(z^\*_i)$, the head's best static fit to a snapshot | 1024 seeded bank-coefficient snapshots |
| `traj` | $c^\*_{j,t}-h_\theta(z_{j,t})$, the $q=0$ ROM's own trajectory error at every internal step against the same-mesh `fft_tight` FOM | 32 training trajectories |
| `prac` | the same quantity | 6 training trajectories (the practitioner's rule) |

All three go through the **same** POD function (`trajdirs.pod_from_residual`), verified in
the local smoke to reproduce `directions.audited` bitwise when handed the incumbent's own
residual, so the sets differ only in which residual matrix is decomposed.

## Layout

| path | what it is |
| --- | --- |
| `DESIGN.md` | predeclared design, gates, pass, falsification, routine calls, and the 2026-09-16 redirect |
| `config-dense.json` | the redirected primary job: two DENSE ladders, no quadrature anywhere |
| `dense_from_archives.py` | Part 1 of the redirect: the dense ladder from the existing archives, no GPU |
| `trajdirs.py` | the trajectory direction rules, the shared POD, cross-capture and principal angles |
| `qtd_run.py` | the driver: three direction sets over eight declared ladders in one allocation |
| `config-qtd.json` | the sweep, the cohorts, the seeds and the cross-job expectations |
| `smoke_qtd.py` | local smoke; 64 intervals, tiny $q$ |
| `audit_qtd.py` | independent NumPy audit; no JAX, no GPU, no driver import |
| `cluster/` | staging, checksum collection and archive chunking |
| `checks/` | smoke and audit JSONs |
| `artifacts/` | checksum-collected raw archive as bounded Git chunks |
| `reports/` | the source-generated report, its figure and its generator |
| `make_lab_entry.py` | the dated LAB-LOG entry, generated from the audits |

## Reproducing

```bash
# local smoke (GB10)
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun /home/tahmid/Dev/.venv/bin/python \
  experiments/q-trajdirs/smoke_qtd.py experiments/q-trajdirs/checks/smoke-qtd.json

# stage, submit, collect, audit (Tufts; namespace q_trajdirs_20260916)
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/q-trajdirs/cluster/stage_qtd.py qtd01   # control (config-qtd.json)
$PY experiments/q-trajdirs/cluster/stage_qtd.py qtd02 experiments/q-trajdirs/config-dense.json
ssh tufts-login 'mkdir -p /cluster/tufts/paralab/tawal01/q_trajdirs_20260916/qtd01'
rsync -a experiments/q-trajdirs/runs/qtd01/ \
  tufts-login:/cluster/tufts/paralab/tawal01/q_trajdirs_20260916/qtd01/
ssh tufts-login 'cd /cluster/tufts/paralab/tawal01/q_trajdirs_20260916/qtd01 && sbatch run.sbatch'
$PY experiments/q-trajdirs/cluster/collect.py qtd01
$PY experiments/q-trajdirs/audit_qtd.py \
  experiments/q-trajdirs/runs/qtd01/archive/output/result.json \
  --fields experiments/q-trajdirs/runs/qtd01/archive/output \
  --out experiments/q-trajdirs/checks/qtd01-audit.json
$PY experiments/q-trajdirs/cluster/preserve_archive.py qtd01
# Part 1 of the redirect: no GPU, from the restored comparator archives
$PY experiments/q-trajdirs/dense_from_archives.py \
  --archives <dir holding cclad01/ btq101/ btq102/ btq201/ restored> \
  --out experiments/q-trajdirs/checks/dense-from-archives.json

$PY experiments/q-trajdirs/reports/generate_dense_ladder.py \
  --audit experiments/q-trajdirs/checks/qtd02-audit.json \
  --result experiments/q-trajdirs/artifacts/qtd02/result.json \
  --archives experiments/q-trajdirs/checks/dense-from-archives.json \
  --control-audit experiments/q-trajdirs/checks/qtd01-audit.json \
  --out experiments/q-trajdirs/reports/2026-09-16-dense-correction-ladder.md \
  --figure experiments/q-trajdirs/reports/2026-09-16-dense-correction-ladder
```

## Restoring the archive

Each `artifacts/<attempt>/` holds the verified `collection.tar.gz` as 48 MB chunks plus
`archive.json`, `SHA256SUMS` and the uncompressed `result.json`. See the `README.md`
written beside the chunks for the restore commands.
