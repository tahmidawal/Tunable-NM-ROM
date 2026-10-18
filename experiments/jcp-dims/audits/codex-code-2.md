**NEEDS-FIX.** The reported crash and several numerical defects are fixed, but the smoke workload, certification identity, and memory contracts remain incomplete.

This audit was read-only. No files changed, jobs submitted, GPU used, or scratch files created. I ran the tests with bytecode writes disabled and called their test functions directly, bypassing the file-writing entry point.

Evidence gathered:

- **86/86 CPU checks passed.**
- All **17 regenerated configuration files match** the existing files, using intercepted in-memory writes.
- Tiny compiled adaptive LSPG and GAL queries succeeded. Both returned `W.shape == (6, 2)` and `fields.shape == (1, 6, 2)`.
- The square diagnostics and both certifiers compiled and executed successfully. I found no traced-Python-control-flow failure in these checks.
- Several adversarial acceptance checks failed, detailed below.

File references are relative to `experiments/jcp-dims/`.

**Round-1 findings, individually**

| # | Verdict | Evidence and remaining issue |
|---|---|---|
| 1 | **CORRECT** | `dtime.py:36` imports `dacc`; anchors at `dtime.py:179` use its runner. `dacc.py:85` selects square-aware tolerances. |
| 2 | **NEEDS-FIX** | Missing certification stores now produce rejecting links through `d3run.py:366`, `:397`, `:428`. However, `dstats.py:278` checks only entries supplied in the certification map; an empty map passes. The link also does not establish successful numerical certification; see below. |
| 3 | **CORRECT** | `d3run.py:88` decodes inside the jitted query. `W[:, None]` correctly converts `(6, R′)` to `(6, 1, R′)`, matching `s3common.py:155`. Confirmed by executing both adaptive forms. |
| 4 | **CORRECT** | `dstats.py:116`, `:130`, `:141` apply missing-evidence precedence to the intersection. Both round-1 counterexamples are now covered and pass. |
| 5 | **NEEDS-FIX** | Per-subject keys are recorded at `d3run.py:393`, `:430`; `keep_wall` is requested at `:231`. But the certification-to-scoring link at `:454–458` compares only `W` and `W_all`, ignoring FS/QN execution records. |
| 6 | **CORRECT** | `dacc.py:280` uses `blk.get('setting')`; fixgate dispatch occurs separately at `:295–298`. The previous `KeyError('setting')` is removed. |
| 7 | **NEEDS-FIX** | The lingering local-variable defects are fixed by `dtime.py:230` and `d3run.py:387`. However, `d3run.py:416–418` clears only `cx.queries`; adaptive compiled queries remain in `ad.q` (`:73–89`). The prescribed compiled-query cleanup is incomplete. |
| 8 | **NEEDS-FIX** | Heavy accuracy queries and heavy 2D GAL are added (`make_configs.py:179–184`, `:212–214`). But `smk3` still lacks the required W1024/gl64 **timing-panel** block (`make_configs3d.py:80–85`). Peak flags do not implement the memory rejection contract. |
| 9 | **NEEDS-FIX** | Two-draw/two-rung 2D coverage, distinct archive tags, reference-A execution, and square diagnostics are added. However, the two-rung W512 lattice smoke still uses only cases `[0,1]` (`make_configs3d.py:69`), versus cases 0–3 in `DESIGN.md:352`. |
| 10 | **NEEDS-FIX** | The new manifest improves 2D identity substantially. Its 3D entries still omit explicit tolerances, budget, weighting and trust; `rounds` blocks are skipped, leaving `q3r4` unrepresented. See `manifest_cells.py:113–131`. Deferred timing configs are not themselves a smoke blocker. |
| 11 | **CORRECT** | Driver dispatch and smoke paths now match the lane (`resources.py:57`, `:87`, `:108`, `:169`). Staging checks exact runs/config hashes (`cluster/stage.py:43–50`) and restricts GPU selection to H200 (`:37`). These are estimates, not proven upper bounds. |
| 12 | **CORRECT** | `ledger.py:68–69` rejects negative/nonfinite optional entries; rounds 5–8 are mandatory at `:34`, with recorded waivers. The previous `b2m=-100` counterexample is rejected. |
| 13 | **NEEDS-FIX** | Fixture pins, empty state-gate rejection and uncertainty agreement are fixed (`dacc.py:314`, `d3run.py:284`, `:310`, `dstats.py:162`, `:256`). But `PROVENANCE.json:85–90` still labels rewritten `resources.py` unchanged and supplies its old hash. I verified the mismatch. |
| 14 | **NEEDS-FIX** | Gaussian/2D MC tests and point-plus-weight Smolyak comparison are added. All 86 checks pass. They still miss empty certification maps, mutated FS/QN execution records, nonfinite adaptive decoded fields, and incomplete heavy-panel coverage. `test_dims.py:216–224` tests supplied false links, not absent links. |

**Ranked outstanding findings**

1. **FS/QN certification identity remains weaker than the registered payload identity.**

   At [d3run.py:458](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/d3run.py:458), equality of `W` and `W_all` suffices. Changing `reason`, iteration counts, or other execution records does not invalidate this link.

   I reproduced `scored_equals_certified(...) == True` after changing an execution-record value while keeping both coefficient arrays fixed. The complete payload hash subsequently compares timed runs against the separate scoring run; it does not repair the missing full-payload link to the earlier certification run.

   Store and compare the certification invocation’s full payload hash. Preserve its numerical validity/certificate outcome separately: the current function also returns `True` when the stored trajectory belongs to a failed run.

2. **`smk3` still does not exercise the required heaviest timing-block sequence.**

   `make_configs3d.py:59–64` executes heavy accuracy ladders, but the panel at `:80–85` contains W1024/lat262144 only. **W1024/gl64 with persistent reference A is absent**, despite `DESIGN.md:637–643`.

   Building/executing gl64 earlier does not test its timing-block lifetime with A resident. Also restore cases 0–3 for the registered two-rung W512 lattice smoke.

3. **Memory handling remains incomplete.**

   - Adaptive compiled-query caches survive panel blocks: `d3run.py:73–89`, `:416–418`.
   - `dacc.py:290` and `d3run.py:519` merely record `over_120GB`. They continue and ultimately write completion.
   - `dtime.py:299–318` checks release relative to baseline but has no explicit 120 GB rejection.
   - Resource readers require `complete`, not accepted memory evidence (`resources.py:59`, `:89`, `:110`).

   The existing flags are useful evidence, but do not yet implement “not reached (memory).” Deferred scoring must not let completion substitute for this gate.

4. **Adaptive validity ignores its newly added decoded fields.**

   [d3run.py:103](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/d3run.py:103) checks coefficients, states, exits and stationarity records, but not `fields`, `w0` or `scale`.

   Reproduced: finite coefficients/states with `fields=inf` return **`ad_valid=True`**. Check the new output contract consistently with `p1common.run_valid`, which checks decoded fields.

5. **`panel_accept` accepts missing certification evidence.**

   Reproduced directly:

   ```python
   panel_accept(
       [dict(B="b", ratio=.5, drift=1., hashes_ok=True)],
       {},
       {"b": 1},
   )["valid"]  # True
   ```

   `dstats.py:278` cannot detect omitted keys. Require the complete expected subject/case key set, including A.

   The current `run_panel` normally populates those keys, so this is a decision-function hole rather than a reproduction of the old missing-store driver bug.

6. **The identity manifest and provenance remain incomplete.**

   `manifest_cells.py:113` excludes round members entirely; `:126–131` lacks several promised 3D numerical parameters. Fix the `resources.py` provenance entry at `PROVENANCE.json:85–90`.

   Separately, `make_round.py:24` silently tolerates missing lane round labels. Before using rounds 5–8, require predecessor evidence so a missing analysis artifact cannot regenerate candidates from stale qualification state.

7. **Regression coverage still does not protect these boundaries.**

   Add targeted checks for full certification-payload linkage, missing certification keys, decoded-field finiteness, and the exact required smoke subject/case inventory. The passing arithmetic tests do not establish those integration properties.

**Deferred work and smoke readiness**

The current `dacc`, `dtime`, and `d3run` execution paths **do not depend on the unwritten analysis/scoring code or `t2`/`t3s`/`t3c` configs to collect smoke or accuracy outputs**. Deferring those files is reasonable. Acceptance and scientific qualification of those outputs do depend on subsequent audited scoring; later rounds also depend on generated round labels.

I found no new reproducible interface or tracing crash in the bounded CPU checks. Nevertheless, the current three-smoke package does not yet demonstrate the registered workload and acceptance contracts.

**Overall: NEEDS-FIX.**