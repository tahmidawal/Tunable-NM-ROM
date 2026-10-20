**NOT READY.** Most round-1 fixes work, but control gating remains incomplete, unavailable-comparator reporting crashes, and the narrowed H4.3 interpretation does not fully satisfy A4.4.

CPU checks used a copied lane under `/tmp/speed3d-review-round2-gwpveoho/`. I reran both surviving round-1 selection fixtures, adding the newly required metadata. No repository writes, git changes, submissions, or GPU use occurred.

| Round-1 item | Disposition | Evidence |
|---|---|---|
| Complete selection population | **RESOLVED for H2** | `analyze.py:320–334` constructs the g1–g3 inventory and blocks selection on missing or duplicate configurations. H1 gating remains incomplete below. |
| H1.2/H1.3 evidence classification | **RESOLVED** | Eligibility now distinguishes `unresolved` from scientific `ineligible`. |
| False H1.4 rescue | **RESOLVED** | Fixture with unresolved FS-K2 now returns `unavailable (FS-K2 unavailable (evidence))`. |
| H1.4 3% cost tie | **RESOLVED** | Shared `pick()` selects the more accurate candidate costing 1.02 versus 1.00. |
| H2 comparator admissibility | **RESOLVED**, with new reporting bug | Unselectable comparator returns unavailable without division. Printing that result crashes; see below. |
| Reference-eligible ranked cases | **RESOLVED under A5** | Explicit IDs and η alignment replace positional comparison; solver eligibility still covers all cases. Both honest score producers pin the same reference version and status. |
| Missing B* decomposition | **RESOLVED** | Explicit unavailable decomposition status and reason. |
| K9 gating | **NOT RESOLVED** | Only `selectable` is cleared; H1 ignores that field. Reproduced H1.4 `pass (screening)` with `blocked_by='K9 did not fire'`. |
| FOM case alignment and η | **RESOLVED under A5** | A 63-ranked-case fixture excludes the unrankable case despite its enormous error, and reports `ranked_cases=63`. |
| K10 convergence control | **RESOLVED** | Raw synthetic evidence goes through `validate()`; rejection reason is convergence, not timing. |
| K10 near-reference-floor control | **RESOLVED** | At `e=1e-5`, `η=1e-4`, the real match succeeds and the inaccurate control is rejected. |
| No match versus unavailable | **RESOLVED** | Empty admissible family returns explicitly unavailable; unsuccessful joint comparison no longer asserts accuracy dominance. |
| Comparator/winner label collision | **RESOLVED** | C_W768 winning against C_old produces canonical label `C_W768`, without crashing. |
| Direct B*–W* confirmation | **RESOLVED** | Distinct-winner fixture now emits `['B_C_old', 'W_C_old']`. |
| Winner equals comparator | **RESOLVED** | Self-selection fixture records explicit ratio-one identity results. |
| Unqualified ROMs exported for matching | **NOT RESOLVED fully** | Scientific ineligibility is filtered, but control/evidence blocks are not: `selectable=False, eligibility='eligible'` still receives a match. |
| Matching the actual timed ROM | **RESOLVED for n=65 mesh path; NOT RESOLVED for full registered H4.3** | New panel uses the path whose accuracy was matched. Mesh-specific matched-accuracy qualification remains absent. |
| p1/m1 resource certificates | **RESOLVED as executable estimates** | Both projection functions ran against actual smoke/pilot records with generated fixture configurations. |
| Grid proxy claimed as upper bound | **NOT RESOLVED** | Actual lookup still prices `fs|CN|0.0125|K2` from `fs|CN|0.01|K1`. The output disclaimer improves interpretation, but the module still says “always an upper bound.” |
| Tighter-ADT/FOM projection limitations | **NOT RESOLVED** | Existing projection logic remains unchanged. These are estimation limitations, not independently demonstrated scientific blockers under A5. |

The remaining blocking defects are:

1. **Control failure can still produce scientific verdicts and FOM matches.**  
   [analyze.py:331](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze.py:331) blocks H2 through `selectable`, but `h1()` consults eligibility/evidence instead. Missing population has the same problem. Separately, [fom_match.py:125](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/fom_match.py:125) and `select_s3.mesh()` accept eligibility alone. Carry an explicit claim-admissibility gate through H1 and FOM matching; preserve scientific eligibility separately.

2. **An honestly unavailable comparator crashes the analysis command.**  
   `select()` omits `S_screen` and `S_BE_screen` on its unavailable branch, but [analyze.py:340](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze.py:340) indexes both. Reproduced `KeyError('S_screen')`. The JSON is already written, so this also leaves an artifact from a command that exits unsuccessfully. Return a consistent schema or print optional fields safely.

3. **The proposed H4.3 interpretation needs an explicit scope amendment.**  
   The added [n=65 mesh-path panel](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:477) is scientifically appropriate for that matched ROM. A1.9 permits a cross-resolution FOM comparator. However, [A4.4](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/DESIGN.md:516) explicitly says **“Every matched FOM configuration is timed against the ROM at every mesh”** under H4.3. Those timings still exist, but calling all mesh-free panels merely H4.1/H4.2 diagnostics narrows the registered claim. Either explicitly amend H4.3 to n=65 mesh-path matching, or validate matching against each mesh-specific ROM before reporting its matched-accuracy speed-up.

Reproduction scripts: [check.py](/tmp/speed3d-review-round2-gwpveoho/check.py), [more.py](/tmp/speed3d-review-round2-gwpveoho/more.py). These findings do not establish a need to rerun g1–g3 or f1.

**NOT READY.**