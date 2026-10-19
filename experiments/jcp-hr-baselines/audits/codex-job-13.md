1. **CORRECT** — Live `sacct`: **5932490 COMPLETED**, exit **0:0**, elapsed **02:49:31**. Ledger reconciles `s3d64b` at **2.8253 GPU-h**. Checksum-verified `stageb.json`: **53/53** members complete, **180/180** timing pairs complete with `outputs_ok=true`, **9/9 G_ours=true**, all **50** phases OK. Live queue contains no HR-baselines retry; no completion failure requires one. Results audit remains pending.

2. **CORRECT** — Recomputed ledger total including `s2d256a`’s reservation: **19.2122 + 4 = 23.2122 GPU-h ≤ 30**.

3. **WRONG as stated; nonblocking** — Bundle unchanged: **63/63 manifest entries match**, no extra files, manifest unchanged from its recorded commit. However, `stageb.py` is **not identical** to `s3d64b`: it predates A44.1 and lacks `canon_dt`, using `float(dt)` instead. That is the entire source difference. All frozen 2D members use **0.005**; `float("0.005") == 0.05/10`, so this omission does **not** affect this job.

4. **CORRECT** — After retrying SSH, read-only remote hashing verified **83/83 dependencies**, including `tune.json`, against the staged frozen hashes under `/cluster/tufts/paralab/tawal01/jcp-hrb/t2d256a/output`. This matches `s2d256a.STAGE.txt`.

**SUBMIT-s2d256a**