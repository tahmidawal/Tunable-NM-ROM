1. **CORRECT** — A43.1 exactly describes the sole `freeze_stageb.py` change from `s2d1024b`: `block × ncase/6` becomes `block × (ncase + 2 + ncase × n_members)/32`. No staged job manifest includes this offline tool. Both bundles contain identical sets of **41 Python sources**, with identical hashes.

2. **CORRECT** — `tune.json` confirms **1,056 complete**, **72 construction-failed**, **836 Q1 passes**, and **9/9 G-ours passes**. Rank-1,024 ECSW reports the **11.95 GiB append-allocation OOM**; adequacy fails at ratios **1.178151934** and **1.584390820**, consistent with A43.2’s restrictions. The ledger confirms job **5542231**, H200, **7.5028 GPU-h**, total spend **16.163 GPU-h**, and the **8.5 h** reservation: **24.663 ≤ 30 GPU-h**. A43.3 agrees with A30.1’s defer order.

3. **CORRECT** — All **63/63 manifest entries** verify; no extra files except the manifest itself. The bundle has no differences from staging commit `3fa9100f2`, preceding audit 8. Manifest SHA-256 remains `c976f09d9fcf69bf8e0dda8996556bf4d882d25a16fec144e5f0fa09ae35e2bb`.

**SUBMIT-s3d64a**