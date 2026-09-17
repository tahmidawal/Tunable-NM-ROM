# lsh01 — RETRACTED attempt (DESIGN.md §A3)

Job `3780148`, A100 on `pax144`, 59:56 elapsed, state FAILED. It completed the operator
gates, all six bank arms, the bank selection and seven of eight head arms, then aborted on
`assert binfo['orthogonality_error'] < 1e-8` (measured 1.8455e-08) — a threshold inherited
from the parent cell's 32-column correction basis and never rescaled for this cell's 128
columns. Per column the value is 1.63e-09, i.e. round-off; nothing was numerically wrong.

**No number from this attempt is reported anywhere.** It is rerun identically, with the
rescaled gate, as `lsh02`.

What is kept here: the partial `result.json` (gzipped), both job logs, the staged-source
provenance and manifests, and the sbatch. **What is deliberately NOT kept: the 67 MB of
trained checkpoints and correction bases.** They are superseded by `lsh02`'s, which are
produced by the same code from the same seeds, and the group share is 92 % full while this
repository is already very large. `OUTPUTS.sha256` was generated post hoc by hand, because
the job aborted before the sbatch line that writes it; `MANIFEST.sha256` verified on the
remote before collection.

```bash
gunzip -c result.json.gz | sha256sum   # f2667c5459d54379c07ae0aad87289c71f5f60d9a0b12366f30c3c6cdadac88c
```
