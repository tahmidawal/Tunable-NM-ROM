# q-diag — diagnosis of the $q=16$ evolved-metric regression

Read-only lane. No cluster job, no GPU: every number is recomputed in NumPy from the already
audited archives of `b-ladder-top` (`btq101`, `btq102`, `btq201`), `cheap-corrections`
(`cclad01`) and `head-ablation` (`qlad01`), plus the frozen Burgers checkpoint
(SHA256 `18f0266a…`) that those jobs used.

- `DESIGN.md` — the pre-registered question, the four candidate causes and the decision rule
  for each, committed before any number was computed.
- `reports/2026-09-16-q16-regression-diagnosis.md` — the report, generated end to end by
  `reports/generate_q_diag.py` from `checks/*.json`.
- `checks/` — the intermediate products every table reads.

## Reproducing

```bash
SP=<scratch>              # somewhere with ~2.5 GB free
PY=/home/tahmid/Dev/.venv/bin/python
CK=.../separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl

# 1. restore the five archives (each verifies against its archive.json sha256)
for J in btq101:b-ladder-top btq102:b-ladder-top btq201:b-ladder-top \
         cclad01:cheap-corrections qlad01:head-ablation; do
  A=${J%%:*}; E=${J##*:}; mkdir -p $SP/restore/$A
  cat experiments/$E/artifacts/$A/collection.tar.gz.part* > $SP/restore/$A.tar.gz
  tar -xzf $SP/restore/$A.tar.gz -C $SP/restore/$A
done

# 2. the four stages
$PY experiments/q-diag/per_time.py        --restore $SP/restore --out experiments/q-diag/checks/per-time.json
$PY experiments/q-diag/reference_norms.py --restore $SP/restore --out experiments/q-diag/checks/reference-norms.json
$PY experiments/q-diag/quadrature.py      --restore $SP/restore --out experiments/q-diag/checks/quadrature.json
$PY experiments/q-diag/local_defect.py    --restore $SP/restore --job cclad01 --arms <list> \
                                          --out experiments/q-diag/checks/local-defect-cclad01.json
$PY experiments/q-diag/heldout_tests.py   --restore $SP/restore --job cclad01 --checkpoint $CK \
                                          --qs 0,16,32,64,128 --out experiments/q-diag/checks/heldout-tests.json

# 3. the report
$PY experiments/q-diag/reports/generate_q_diag.py \
    --checks experiments/q-diag/checks \
    --out experiments/q-diag/reports/2026-09-16-q16-regression-diagnosis.md
```

`heldout_tests.py` re-implements the decoder, bank, test modes and weak residual in NumPy and
is gated on reproducing the cluster jobs' own recorded per-step residual norms
($3.4\times10^{-11}$ relative); `local_defect.py` re-implements the full-order step and is
gated on propagating `fft_tight` back onto itself ($4\times10^{-9}$ relative).
