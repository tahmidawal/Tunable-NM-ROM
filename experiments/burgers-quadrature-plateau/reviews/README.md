# Reviews of this lane

| file | reviewer | object | verdict | disposition |
|---|---|---|---|---|
| codex-design-01.md | Codex CLI 0.166 (gpt-6.1-sol, prompt-enforced read-only; the `-s read-only` sandbox cannot start on this box) | DESIGN.md §§1–11 | NOT-READY (2 blockers, 7 majors) | answered by amendment A1 |
| codex-design-02.md | same | A1 | NOT-READY (blocker 1 partially open; 5 partial; 1 new major) | answered by amendment A2 |
| codex-design-03.md | same | A2 | READY | design released for cluster jobs |
| codex-results-01.md | same | generated report, plateau.py, audit.py vs DESIGN A1–A5 and the result JSONs | READY-WITH-FIXES (0 blockers, 4 majors, 2 minors; all numbers reproduced) | fixed: time-error attribution removed (§2 now "consistent with", controlled Δt study owed); A2.1 tail wording applied in text/tables/plot stars; timing disclosures added (IQR/min–max/outliers, paired per-case ratios, PS + IC-prep sensitivity, full-field and decode); audit G-par failure now fails the audit and the generator withholds failed meshes; matching fails closed for ineligible FOMs / undemonstrated plateaus; A2.4 checks both members; "mesh-independent" → "approximately unchanged over the tested meshes"; operator sentence restored; failed-job logs transcribed in results/FAILED-JOBS.md (logs were deleted before pulling — disclosed). Not done: envelope treatment of the 1 % tail test and of the crossover sign change (Codex rechecked: no classification changes). |
