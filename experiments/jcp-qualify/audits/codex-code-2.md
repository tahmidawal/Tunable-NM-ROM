**NEEDS-FIX — not ready for cluster smoke.**

Paths below are relative to `experiments/jcp-qualify/`. All 21 `test_q` fixtures passed when invoked without their file-writing runner; additional CPU checks exposed failures.

Ranked must-fix items:

1. **WRONG — memory pilot still understates timing’s footprint.** `qtime.py:87–96` builds both successors together; `qpilot.py:55–67` builds/drops them individually. Moreover, `qtime.py:199–204` retains reference A’s successors until its later R384 subjects finish, overlapping other settings. The per-setting drop improves memory but does not satisfy A2.4.

2. **WRONG — reservation inventory remains unenforced.** `ledger.py:67–81` accepts `{}`, missing mandatory jobs, and negative/non-finite projections. No validated A3–A5 lifecycle enforces fallback replacement and timing enlargement. Pending entries disappear only at bind (`:85`), so `PENDING_SUBMIT`/`AMBIGUOUS` attempts remain double-counted. CPU example: a 2-hour pending submission counts as 4 hours against the next submission.

3. **WRONG — fallback timing can be attached to the original configuration.** Accuracy keys now include `|fb`, but front/role/timing-export consumers omit it: `analyze.py:389`, `:416`; `front2d.py:71`. `front2d.py:22` also crashes on fallback keys—CPU-confirmed `ValueError`. Split accuracy and repeated timing attempts remain undiscovered (`analyze.py:174`, `:312`).

4. **WRONG — the registered 3D archive cannot satisfy the new offline reader.** `s3ctx.py:168–169` overwrites arrays by setting/rule/spec. In `configs/q3d.json:90–95`, `W256|gl24|fs|CN|0.0125|K1` is first an eight-case check, then a 64-case successor. `q3offline.py:62` requires the archived population to equal eight exactly and aborts. A config-only replay confirmed this collision.

5. **WRONG — reproduction failure still permits H-acc-r verdicts.** `build_2d` clears `numerically_qualified` (`analyze.py:195–196`), but `case_qualified` ignores that flag and reads only Q1/Q2 (`:139–142`); `h_acc_r` consumes it (`:245–248`). Smoke reproduction also omits manifest equality (`smoke_accept.py:46–51`); the added combined comparator omits tolerances (`analyze.py:162–165`).

6. **NEEDS-FIX — 3D evidence validation remains incomplete.** `analyze.py:495` checks ID lists, not corresponding array lengths; a synthetic one-case evidence array claiming 64 IDs passed with the other gates satisfied. `offline3d` accepts an empty rho list through vacuous `zip` (`:523–524`). Smoke merely reports offline-blocked labels (`smoke_accept.py:161–163`). Require explicit shapes/populations and gate offline agreement.

Prior-audit disposition:

| Prior item | Status | Evidence |
|---|---|---|
| 1. Timing memory | **PARTIALLY** | Per-setting release added, but blocker 1 remains. |
| 2. Pilot discovery | **RESOLVED** | `smoke_accept.py:144` selects `qpilot_pilot`. |
| 3. Ledger | **PARTIALLY** | Missing-map rejection and bind removal added; blocker 2 remains. |
| 4. Non-finite certificates | **PARTIALLY** | Offline/GAL fixes: `qscore.py:33–48`, `qcert.py:81–95`. In-job LSPG still calls unsafe `p1common.gcert_ok` (`:317–320`) from `qacc.py:184` and `qtime.py:146`; finite-vs-infinity still returns true. |
| 5. Smoke controls | **RESOLVED** | `smoke_accept.py:61–64,132–139,170`; `test_q.py:46–49`; `make_configs.py:114` now uses dev6. |
| 6. Gates block verdicts | **PARTIALLY** | TH06/front blocking added (`analyze.py:262–276,461–463`); blocker 5 remains. |
| 7. Fallback representation | **PARTIALLY** | Full38 fallback generation fixed (`make_configs.py:66–77`); blocker 3 remains. |
| 8. Pooled ext22 Q1 | **RESOLVED** | `analyze.py:102–108` excludes standalone certificate predicates and checks the union. |
| 9. 3D evidence | **PARTIALLY** | Missing reproduction/own invalidity now block (`analyze.py:498–505`); blockers 4 and 6 remain. |
| 10. Open candidates | **RESOLVED** | `analyze.py:459–460` retains candidates open on either panel. |
| 11. Role uncertainty/eligibility | **RESOLVED** | Defining-case uncertainty and timing qualification: `analyze.py:135–136,416–432`. Fallback identity remains separately broken. |
| Missing EXPORT | **RESOLVED** | `analyze.py:582–598`. |
| Missing cited P1 transitions | **RESOLVED** | `analyze.py:577–579`, including the 16-case qualification. |
| Resource provenance | **RESOLVED** | `PROVENANCE.json:86–91` identifies replacement rather than unchanged copying. |

Other changed-code findings:

- **NEEDS-FIX — `resources.py`:** fallback pricing ignores quadrature rule (`:52–56,83–85`), assigning unchanged per-case estimates to larger rules. Reprice or explicitly justify a conservative fallback allowance.
- **NEEDS-FIX — `front2d.py`:** metadata comes only from the original accuracy directory (`:26–32`); exported timing includes certification hashes but no per-triplet `hashes_ok` evidence (`:82–85`), required by A1.6.
- **NEEDS-FIX — `q3offline.py`:** zero denominator always produces NaN (`:72`), unlike `q3run.py:34–35`, which returns infinity when the numerator is nonzero.
- **CORRECT — `qacc.py` control arrays:** required normalization, successor validity, and IDs are now saved (`:314–317`).
- **CORRECT — fallback generator and added front fixture:** full38 fallback arms and actual candidate filtering are exercised (`make_configs.py:66–77`; `test_q.py:46–49`). Coverage does not catch the integration failures above.

**Read-only breach:** one test import unexpectedly executed the references lane’s `make_configs.py`, rewriting its configs. Both worktrees subsequently showed clean Git status, but writes did occur. No LAB-LOG writes, cluster submissions, or GPU runs occurred.