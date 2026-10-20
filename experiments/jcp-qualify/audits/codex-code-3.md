**NEEDS-FIX** at `f57178774`. Reviewed the full lane diff from `a971467ec`. Paths below are relative to `experiments/jcp-qualify/`.

Ranked must-fix items before cluster smoke:

1. **WRONG — reservation lifecycle remains unenforced.** `ledger.py:85–90` accepts an inventory containing mandatory job names but **no fallback allowances**; A4/A5 replacement and timing enlargement remain unchecked. Newly, popping at reserve (`:78`) loses the pending reservation after release/failure (`:115,121`); subsequent submissions only check that the map exists (`:70`). Reservation hours themselves still accept negative/NaN values (`:69–74`).

2. **WRONG — pilot still misses retained fallback tables.** Timing retains every original/fallback setting’s main tables (`qtime.py:81–100,201–215`). The pilot represents fallback mains as temporary roles and drops them (`qpilot.py:62–78`), so multiple retained fallbacks are never measured together. Grouping A’s subjects fixes its earlier successor overlap, but does not establish A2.4’s full footprint.

3. **NEEDS-FIX — fallback timing remains unpriced.** Accuracy now scales by rule size (`resources.py:64–68`), but timing certification and triplet projections still ignore rule size (`:91–98`). Enlarging a fallback rule leaves its timing estimate unchanged.

4. **NEEDS-FIX — attempt discovery is not reliable source selection.** Accuracy reruns containing overlapping configurations now abort at `analyze.py:203`; no supersession rule exists. `front2d.py:30–37` selects the last attempt containing the **rank**, without checking the requested arm, so split attempts can export another arm’s source metadata.

5. **NEEDS-FIX — smoke’s P1 reproduction still lacks manifest equality.** Combined checks now compare tolerances and manifests, but the R384/P1 smoke path still checks trajectory agreement alone (`smoke_accept.py:51–53`).

Disposition of every numbered finding in `codex-code-2.md`:

| Item | Status | Judgment / evidence |
|---|---|---|
| 1. Timing memory | **PARTIALLY** | **NEEDS-FIX:** simultaneous successors and A ordering fixed; retained fallback footprint remains wrong. `qpilot.py:51–78`; `qtime.py:110–112`. |
| 2. Reservation inventory | **PARTIALLY** | **WRONG:** mandatory names and finite pending values checked; double-counting fixed, lifecycle still broken. `ledger.py:78–90,118–121`. |
| 3. Fallback identity/discovery | **PARTIALLY** | **NEEDS-FIX:** fallback keys now propagate correctly; split discovery added, overlapping reruns/source selection unresolved. `analyze.py:49–59,188–203,405,434`; `front2d.py:22–37,73`. |
| 4. 3D archive collision | **RESOLVED** | **CORRECT:** member/role-specific keys used consistently by writer, offline reader, reproduction. `q3run.py:44–61`; `q3offline.py:51–64`; `analyze.py:553–555`. |
| 5. Reproduction gates | **PARTIALLY** | **NEEDS-FIX:** H-acc-r now respects reproduction failure; combined tolerance check fixed; P1 smoke gap remains. `analyze.py:152–156,178–179,260–263`. |
| 6. 3D evidence validation | **RESOLVED** | **CORRECT** for reported counterexamples: population lengths, rho shape/count and smoke offline agreement enforced. `analyze.py:512–515,543`; `q3offline.py:53,63`; `smoke_accept.py:165–170`. |

Other findings from code audit 2:

| Item | Status / judgment |
|---|---|
| Non-finite in-job LSPG certificates | **RESOLVED / CORRECT:** explicit finite checks, `qacc.py:184`, `qtime.py:148`. |
| Fallback resource pricing | **PARTIALLY / NEEDS-FIX:** accuracy changed; timing unchanged, as above. |
| FRONT-2D metadata and hashes | **PARTIALLY / NEEDS-FIX:** triplet hashes exported (`analyze.py:368–371`; `front2d.py:87`); exact source selection remains wrong. |
| Offline zero-denominator rho | **RESOLVED / CORRECT:** matches runner’s NaN/∞ distinction, `q3offline.py:74–75`. |
| Control arrays; fallback generator/front fixture | **CORRECT**, unchanged: `qacc.py:314–317`; `make_configs.py:66–76`; `test_q.py:46–49`. |
| Import-path mitigation | **CORRECT:** references path moved behind lane modules, `qscore.py:20–24`; verified imported helper locations. |

Previously resolved findings remain resolved: pilot discovery (`smoke_accept.py:146`), smoke controls (`:63–65,134–141,175`), pooled ext22 (`analyze.py:115–122`), open candidates (`:476–477`), role eligibility/uncertainty (`:434–450`), EXPORT/P1 citations (`:596–617`), resource provenance (`PROVENANCE.json:86–91`).

Small CPU checks confirmed reproduction blocking, fallback keys, and unique 3D storage keys; reproduced acceptance of an inventory without fallback allowances. **No files written, scratch created, cluster jobs submitted, or GPU runs performed.**