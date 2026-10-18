1. **CORRECT — bundle integrity.** At recorded commit `c9cc52fad8ad4c015fe6d1433c63649126de6592`, all 59 non-config provenance files equal their Git blobs. The config equals the committed config plus only the injected `commit` key, including the expected serialization. All 63 manifest entries verify; no missing or extra payload files.

2. **CORRECT — permitted changes only.** Against `s2d1024a`, payload differences are exactly `stageb.py` and the config’s injected commit. Staged `stageb.py` equals the version approved by code audit 32. Frozen membership and its `.sha256` are byte-identical; membership SHA256 remains `e463836bf8d0d309295e6ed1ac099743323e914f7578a0a0a3eaaeae5637c7b9`.

3. **CORRECT — remote dependencies.** `s2d1024b.PREFLIGHT.txt` reports 73/73 verified. Independent read-only SSH verification also returned **73/73 matching**, zero mismatches.

4. **CORRECT — batch script.** `run.sbatch` retains GPU partition, one H200, 200G, 03:30:00, isolated paralab paths, prescribed venv, x64, highest precision, GPU preflight exit 42, and `JCPREF_STORE`. Driver deadline is 206 minutes plus 30-second kill allowance; internal budget is 195 minutes.

5. **CORRECT — ledger.** Exactly one unbound 3.5-hour reservation exists for `s2d1024b`. Previous attempt’s 2.1294 GPU-h matches archived accounting of 02:07:46 on one GPU. Total charged/reserved is **22.878 ≤ 30 GPU-h**, including `t3d64b`’s 12-hour reservation.

**SUBMIT-s2d1024b**