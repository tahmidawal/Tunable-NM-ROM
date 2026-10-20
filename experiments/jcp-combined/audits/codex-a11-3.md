**Overall: CORRECT for the revised A11.3 plan. SUBMIT `smk3ext2.json` on H200. HOLD `c3extW1024` until the new smoke passes its machinery gate and a valid resource certificate passes staging verification. No blocking defect found in this round.**

1. **`select3d.py` — CORRECT.** Round-2 defects are fixed:
   - Reference identity uses `sha256`, `n`, and `dt`, allowing different staged paths.
   - Extension acceptance explicitly requires W1024.
   - Successor arrays used in recomputation pass the same explicit shape check as gl80: production `(64, 6, 1024)`.
   - Population, numerical contract, four-arm gl80 reproduction, and independently recomputed A10 pair-status agreement remain enforced.

2. **`inventory.py::smoke_gates` — CORRECT.** Read-only, in-memory tests of the actual function confirmed:
   - Complete local extension evidence passes.
   - Empty pairs, one retained row, a missing axis arm, and duplicate rows fail.
   - A complete two-rule inventory passes.
   - Unjustified omission of a third rule fails; omission after an A10 pass succeeds.

   The timing-panel exemption is appropriate. Machinery acceptance requires complete, valid evidence; it does **not** require the scientific convergence axis to pass.

3. **Synthetic machinery evidence — CORRECT, limited scope.** `sel-fakeext_a.json` exists; both recorded input hashes match. All four gl80 arrays reproduce exactly, and gl96 equals gl80 multiplied by `1+1e-9`. The fixture uses a different reference path. Its saved output reports matching recomputed statuses, convergence at gl80, and selection of `lat65536`.

   `fakeext_b` exists and its arrays match `mkfake.py`, but **`sel-fakeext_b.json` is absent**. Consequently, there is no saved successful gl96-selection result to credit. That historical branch is unnecessary for A11.3.

4. **A11 wording and A11.3 — CORRECT when read chronologically.** A11.3 supersedes A11.2’s gl112 escalation and original smoke plan. The earlier wording corrections hold: case 5 alone exceeds the fixed-BE bar, other cases are at least approximately 15× below it, and W768’s original axis remains unresolved. Selection is reapplied and remains explicitly post hoc.

   The failure archive supports A11.3: GPU/f64/highest are logged; gl96 produced eight valid rows covering four arms and two cases; gl112 failed in `test_block` while allocating **42.88 GiB**. The result is incomplete and fails the current smoke gate. The ledger records this accurately. The memory conclusion applies to this implementation and allocation.

5. **Configs and removal of planned gl112 — CORRECT.** Both JSON configs exactly match their generator definitions. The inventory contains:
   - `smk3ext2`: two cases, gl80/gl96, four arms — **16 rollouts**.
   - `c3extW1024`: 64 cases, identical rules/arms — **512 rollouts**.
   
   Neither schedules gl112 or timing panels. Passing the sole extension pair yields gl80; otherwise selection remains unavailable. Freeze propagates the effective post-hoc rule, and budget generation maps **gl80 → gl96**. Dormant gl96→gl112 support and retained rule files do not schedule gl112 under these configs.

**Production gate:** Yes, `c3extW1024` may be submitted once `smk3ext2` passes machinery checks and its resource certificate verifies memory, runtime, GPU, configuration, and source bindings. Scientific convergence on the two smoke cases is not an additional submission requirement.

No files were created, modified, or appended.