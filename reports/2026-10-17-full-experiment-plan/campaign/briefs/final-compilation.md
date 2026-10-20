# Brief: final compilation of the Burgers 2D/3D campaign (AGENT-PROMPT-burgers.md §8)

Read `briefs/COMMON-BRIEF.md` (the writing, integrity and Codex rules apply; you run **no** cluster jobs).
You write only in `reports/2026-10-17-full-experiment-plan/final/` on `main` (the coordinator commits).
Do not edit `LAB-LOG.md`, `notebook/`, or any lane worktree.

## Deliverable
`final/make_final.py` → `final/report.md` + `final/figs/*.png` + `final/FINAL-EXPORT.json`.
**Every number is read from lane outputs** (each lane's `runs/EXPORT.json`, `report.md`-adjacent JSONs,
`checks/*.json`, `runs/GPU-LEDGER.json`); none is hand-typed. Each number carries its source path in a
machine-readable provenance table (`final/PROVENANCE.json`: number id → file, JSON key path, sha256 of file).
Copy (do not regenerate) lane plots where they are the right figure; make new summary plots from JSON.

## Lanes (all finished; paths under `worktrees/`)
| lane | worktree | branch / last commit / mirror — read from git and the lab log |
|---|---|---|
| jcp-speed3d | 2026-10-14-jcp-speed3d | 3D speed, f1 FOM ladder, m1 mesh panel, H2, K2 3D |
| jcp-p1-scaling | 2026-10-17-jcp-p1-scaling | P1a–P1g, Q3: H-cost-N/r, H-acc-N/r, P1g |
| jcp-qualify | 2026-10-18-jcp-qualify | Q1–Q3, B1, B4, B5, FRONT-2D, 3D Q2 subset |
| jcp-fom2d | 2026-10-18-jcp-fom2d | C1 ladder, C2 crossover, H-comp 2D, K2 2D |
| jcp-hr-baselines | 2026-10-06-jcp-hr-baselines | C3 minimal, H-base |
| jcp-heads | 2026-10-18-jcp-heads | C5 K sweep, linear vs head |
| jcp-dims | 2026-10-18-jcp-dims | A3, A4, B3, B6, 3D front, H2q |
| jcp-budget | 2026-10-18-jcp-budget | B2, B7, B8 |
| jcp-robust | 2026-10-19-jcp-robust | E1, E2, E5 |
| jcp-regimes | 2026-10-19-jcp-regimes | D1 |

## Report structure (AGENT-PROMPT §8.1)
1. Level-1 title; one-paragraph status (final for the development/validation cohorts; sealed/OOD unopened;
   E0 ST conditional Richardson ε_ref; post-hoc labels E0 A10, combined A10/A11/A15 retraction).
2. **Hypothesis verdicts** table: H-cost-N, H-cost-r, H-acc-N, H-acc-r, H-comp (per dimension and per FOM
   family + union), H-base (per dimension/mesh) — pass / fail / inconclusive, key number, figure, source lane.
   Include K2 (fires 2D and 3D) and both readings of "k" (r and K).
3. **Per-priority results with plots:** scaling vs k (r), scaling vs mesh, accuracy–cost front, speed-up vs
   each FOM family, POD baselines, rule families, test space, error budget, long horizons, robustness (E1,
   E2, E5), regimes (D1).
4. **P1g recheck:** how much earlier "flat accuracy" results moved under frozen mode lists.
5. **Retractions and corrections** (consolidated from the lab log entries 2026-10-17 → end and each lane's
   report): e.g. speed3d H2 winners not quadrature-qualified; combined H5 13.6× scope; combined finalist
   labels; "1024² reuse" false for R256; coordinator's 4096² mode-list claim; post-data tolerance changes.
6. **Deferred items** (each lane's deferred list + coordinator deferrals: C3-full, C4, 3D regimes, B8 3D, …).
7. **GPU-hour ledger**: per lane (from GPU-LEDGER.json), total.
8. **Lane list**: branch, last commit, mirror commit, report path — "awaiting the user's merge decision".
9. **Plain-language glossary** at the end.
Never mention the heat POD-128 comparison. LaTeX maths, mermaid diagrams, provisional labels inline.

## Gates
Codex audit of `make_final.py` + `report.md` (`final/audits/codex-final-<n>.md`): recompute numbers from
the lane outputs, check provenance, check every verdict against the lane's own verdict, check nothing
retracted is presented as a result. Fix every WRONG; re-audit until clean. Hand back: report path, audit
verdicts, open issues.
