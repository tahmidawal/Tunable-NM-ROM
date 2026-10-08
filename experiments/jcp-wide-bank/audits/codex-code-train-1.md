**Verdict: training code and configs pass the requested structural checks; the training staging wrapper NEEDS-FIX before J3.** No files modified, including the lab log. I ran the filter unit test and read-only provenance/config/shell checks; I did not run training.

Other activity changed files during this audit. The staging findings below refer to the working copy containing the sidecar and `mkdir -p "$TASK_ROOT/output"`; HEAD still lacks the sidecar.

1. **CORRECT — exactly two registered code changes.**  
   `train2_original.py` is byte-identical to `58d83d09b:experiments/burgers3d-retry/train2.py`, SHA256 `6e3d9a0d4a7bbb0382f9e6b805570c5915078a16c4af27fa6e6809c9b262e78c`.

   The diff contains the required explanatory header and exactly two executable changes: the import path at `train2w.py:44` and comparison filtering/reporting at `train2w.py:451–457`. No other behavioral edits. **Fix: none.**

2. **CORRECT — both vendored imports resolve in the staged layout.**  
   `train2w.py:44` resolves to `vendor/quad3d/vendor/burgers3d-span/common.py`. That file’s `common.py:34–36` resolves its sibling `paper-b3d/vendor/b3d_common.py`.

   `cluster/stage.py:28–29` stages all three files at these exact relative paths. Both dependencies match their pinned source-commit bytes. **Fix: none.**

3. **CORRECT — R=1024 config changes only registered keys.**  
   Compared programmatically against the specified Git blob. Exactly eight keys differ: `bank_rank`, `bank_width`, `modes`, `ladder`, `pod_reference_ranks`, `model_seed`, `head_variants`, and `compare_bank`. Values match A1/A2; no added or removed keys.

   References: `train3d/configs/train_r1024.json:31–45`, `:57–80`. **Fix: none.**

4. **CORRECT — no demonstrated R=1024 dimensional or indexing defect.**  
   Relevant checks:

   - `train2w.py:124`: 2048 modes fit within **18,432 snapshots per mesh**.
   - `:108`: width 2048 and output rank 1024 are config-driven.
   - `:383–406`: identity matrices, QR factors, coefficient ordering and rotation scale with the configured rank. A float64 `np.eye(1024)` is only 8 MiB.
   - `:410`, `:440–442`: cumulative energies and spreads cover every registered ladder entry through 1024.
   - `:311–321`: one unpivoted QR supports nested-prefix floors when those prefixes are full rank.
   - `:451–457`: comparison floors stop at the comparison bank’s column count; wider ranks are explicitly unavailable. Both banks use the same `full_val[n]`, as A2-B3 requires. New-bank floors contain the 512/1024 values needed for A2-B4.
   - `:467–469`: head targets remain computed and saved even though head training is skipped, as registered.

   **Limitation:** QR floors are not rank-revealing. A numerically deficient bank can produce misleading floors; dimensional correctness does not establish B2′ or memory feasibility. **Fix/action:** enforce the registered stability gates before accepting downstream results; no additional trainer change is established by this audit.

5. **CORRECT — smoke config and comparison-filter test.**  
   `train_smoke_local.json:2–55` consistently specifies mesh/order mesh 17, white group `"17"`, all required saved steps, 96 modes for **8 × 12 = 96 snapshots**, rank 64, width 128, and warmup 5 within 20 steps.

   Batch size 128 exceeding 96 snapshots is valid: `train2w.py:200` samples with replacement. `test_filter.py:25–30` executes the deployed filter block using a 32-column fake bank, verifies rank 64 is unavailable, and verifies only rank 16 reaches the floor function. **Test passed. Fix: none.**

6. **CORRECT — B2′ decision quantities are already available in `training.json`.**  
   `train2w.py:409–411` records `ordering.inverse_check`, exactly the requested Frobenius residual. `:437–438` computes singular values of **RG** and records `bank.condition_65`.

   For this 1024-column bank, finite `condition_65 < 1e10` is equivalent to the registered strict condition  
   \(\sigma_{\min}/\sigma_{\max}>10^{-10}\). Thus **bank.pkl is unnecessary to decide the full-rank threshold**.

   An explicit numerical-rank count and RG singular-value vector are **not** written to JSON. Also, `ordering.singular_values` belongs to **Arows**, not RG. If the actual rank count is required, compute it afterwards from `bank.pkl["RG"]`; `RG`, `V`, and `rotation` are saved at `:443–444`.

   **Required downstream action:** reject missing/null/nonfinite metrics, require the condition threshold and inverse check ≤ `1e-8`, and enforce J4’s separate mesh-Gram gates. `complete=True` alone is not bank acceptance.

7. **NEEDS-FIX — training command is correct, but A4-J3 instrumentation is incomplete.**  
   `cluster/stage.py:34` correctly runs from the staged root, supplies both required arguments, and wraps Python with `/usr/bin/time -v`. Its statistics go to the Slurm stderr log. The rendered command passes `bash -n`.

   **The RSS sidecar is present in the working copy**, at `:35–39`; it is absent from HEAD. Specific problems:

   - **Sampler startup race:** `:36` executes `C=$(pgrep -P $TPID | head -1)` under `set -euo pipefail` (`:88`). Before `time` forks Python, `pgrep` can return 1 and terminate the background sampler. Its exit status is never checked. **Fix:** tolerate/retry “child not yet present,” retain the sampler PID, and supervise its exit.
   - **Missing phase timestamps:** RSS samples are timestamped, but `train2w.py:354–356` prints training messages without timestamps; `:479` saves the same untimestamped messages. `ORDER`, `FLOORS`, and group messages therefore cannot reliably be aligned with RSS samples as A4-J3 specifies. **Fix:** timestamp training stdout in the external wrapper, preserving the two-change trainer contract.
   - **Incorrect J3 defaults:** `stage.py:49–50` defaults to A100-80G/128G, not H200/420G. **Fix:** supply `--gpu h200 --mem 420G` explicitly, or enforce train-specific defaults.
   - **Uncommitted wrapper:** `:61–62` rejects the current dirty lane. **Fix:** finalize and commit the reviewed wrapper before staging.
   - **Accounting collection:** the wrapper does not collect `sacct` MaxRSS. **Action:** collect it after job completion alongside the logs, as registered.

| Item | Verdict | Concrete fix/action |
|---|---|---|
| 1. Two-change provenance | CORRECT | None |
| 2. Staged import chain | CORRECT | None |
| 3. Registered config differences | CORRECT | None |
| 4. Wide-rank code paths | CORRECT | Validate numerical stability and actual resource use |
| 5. Smoke config/unit test | CORRECT | None; filter test passed |
| 6. B2′ quantities | CORRECT | Gate JSON metrics; use saved RG only for explicit rank/spectrum |
| 7. Staging/time/RSS | NEEDS-FIX | Supervise sampler, timestamp phases, request H200/420G, commit wrapper, collect accounting |